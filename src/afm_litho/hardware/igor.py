"""COM (Execute2) path into Igor Pro: calibration reads, installers, out-of-band SafePark.

Data (commands, captures, telemetry) rides the SOCKIT bridge instead (bridge_protocol; see
docs/IGOR_PYTHON_COMMS.md §1). Execute2's signature, from Igor's VB example (Utilities.bas):

    IgorApp.Execute2 allowInterrupt, binary, cmd,
                     -> errorCode, errorMsg, history, results

`results` is the readback channel: an `fprintf 0, ...` inside the command writes there.

Two error channels: Igor-level (errorCode/errorMsg, a bad command string) -> IgorError, and
ARC-level (a td_/ir_ return value, an ae_ code that Igor-level success hides) -> ArcError via
arc_call(). Both carry a bounded trace of recent commands. win32com is imported lazily and any
object exposing Execute2 works, so the suite drives this with a fake.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .arc_errors import ArcCode, decode
from .bridge_protocol import note_wire

# Version-independent ProgID -> IgorPro.Application.6 on this lab's machines.
PROGID = "IgorPro.Application"


def _render_trace(trace) -> str:
    if not trace:
        return ""
    lines = "\n".join(f"    {c}" for c in list(trace)[-8:])
    return f"\n  recent commands (most recent last):\n{lines}"


class IgorError(RuntimeError):
    """An Igor command failed at the Execute2 level (a bad command string)."""

    def __init__(self, code: int, message: str, command: str, trace=None):
        self.code = int(code)
        self.message = message or f"Igor error {code}"
        self.command = command
        self.trace = list(trace or [])
        super().__init__(
            f"{self.message} (Igor error {self.code}) while running: "
            f"{command!r}{_render_trace(self.trace)}"
        )


class ArcError(IgorError):
    """A td_/ir_ ARC function returned a documented nonzero error code.

    Subclasses IgorError so `except IgorError` catches both, but carries the
    decoded ArcCode (name, category, description) rather than an Igor message.
    """

    def __init__(self, code: int, call: str, trace=None):
        self.arc: ArcCode = decode(code)
        self.call = call
        self.code = self.arc.code
        self.command = call
        self.trace = list(trace or [])
        RuntimeError.__init__(
            self,
            f"ARC rejected `{call}` -> {self.arc}{_render_trace(self.trace)}"
        )


@dataclass
class IgorResult:
    """The four out-parameters Execute2 returns, in Utilities.bas order."""

    error_code: int
    error_msg: str
    history: str
    results: str


class Igor:
    """Drives one Igor automation server.

    `app` is a live COM object (from connect()) or any stand-in exposing
    Execute2(allowInterrupt, binary, command) -> (code, msg, history, results).
    """

    def __init__(self, app=None, trace_len: int = 64):
        self._app = app
        # A bounded log of commands sent, so an error can show its lead-up.
        self.trace = deque(maxlen=trace_len)

    @property
    def connected(self) -> bool:
        return self._app is not None

    def connect(self, allow_running: bool = True, *, launch: bool = True) -> "Igor":
        """Attach to Igor's automation server (pywin32 imported lazily).

        allow_running reuses an open Igor (Dispatch); False forces a fresh one (DispatchEx).
        launch=False attaches only to a running Igor (GetActiveObject) and raises if none:
        a script that then bails on missing Asylum software must not orphan a blank Igor.
        """
        if self._app is not None:
            return self
        try:
            import win32com.client
        except ImportError as exc:  # instrument PC only
            raise RuntimeError(
                "pywin32 is required to talk to Igor: pip install -e .[hardware]"
            ) from exc
        if not launch:
            try:
                self._app = win32com.client.GetActiveObject(PROGID)
            except Exception as exc:                 # no running Igor -> com_error
                raise RuntimeError(
                    "No running Igor Pro found. Open Igor and load the Asylum "
                    "MFP3D software first, then re-run -- this script attaches to "
                    "your Igor rather than launching a blank one."
                ) from exc
            return self
        factory = (
            win32com.client.Dispatch if allow_running else win32com.client.DispatchEx
        )
        self._app = factory(PROGID)
        return self

    def execute(self, command: str, *, check: bool = True) -> IgorResult:
        """Run one Igor command string and return its four out-parameters.

        pywin32 returns Execute2's [out] parameters as a tuple, in IgorPro.tlb order.
        """
        if self._app is None:
            raise RuntimeError("not connected to Igor; call connect() first")
        self.trace.append(command)
        code, msg, history, results = self._app.Execute2(0, 0, command)
        r = IgorResult(int(code), str(msg or ""), str(history or ""), str(results or ""))
        note_wire("com", command, code=r.error_code)  # after Execute2 returned
        if check and r.error_code != 0:
            raise IgorError(r.error_code, r.error_msg, command, self.trace)
        return r

    def arc_call(self, call: str) -> int:
        """Run a td_/ir_ ARC function; raise a decoded ArcError unless it returns ae_None.

        Execute2 reports success for an ARC function that runs but rejects its arguments,
        so the return value is read back via fprintf. Use execute() for Igor operations
        with no ARC return (Make, LoadWave, fprintf).
        """
        r = self.execute(f'fprintf 0, "%d", ({call})')
        text = r.results.strip()
        code = int(text) if text.lstrip("-").isdigit() else 0
        if code != 0:
            raise ArcError(code, call, self.trace)
        return code

    def value(self, expr: str) -> float:
        """Evaluate a numeric Igor expression via fprintf readback (%.15g: full precision)."""
        r = self.execute(f'fprintf 0, "%.15g", ({expr})')
        text = r.results.strip()
        if not text:
            raise IgorError(0, f"no value returned for {expr!r}", expr, self.trace)
        return float(text)

    def string(self, expr: str) -> str:
        """Evaluate a string Igor expression and return its `results` text."""
        return self.execute(f'fprintf 0, "%s", ({expr})').results


class LazyIgor:
    """A COM Igor handle that attaches on first use, for the out-of-band SafePark.

    Attaches with launch=False (never a blank Igor). Callers pass check=True, so a COM error
    or an Igor-level rejection raises and the trigger falls back to the TCP ABORT
    (docs/IGOR_PYTHON_COMMS.md §5). COM needs CoInitialize on the calling thread: the
    fail-safe thread (failsafe_thread.py) initialises COM and owns its own LazyIgor.
    """

    def __init__(self, igor: "Igor | None" = None):
        self._igor = igor

    def execute(self, command: str, *, check: bool = True) -> IgorResult:
        if self._igor is None:
            self._igor = Igor().connect(launch=False)
        return self._igor.execute(command, check=check)

    def arc_call(self, call: str) -> int:
        """Igor.arc_call on the attached handle: raises unless the ARC function returns 0."""
        if self._igor is None:
            self._igor = Igor().connect(launch=False)
        return self._igor.arc_call(call)
