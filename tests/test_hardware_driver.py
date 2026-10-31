"""Wave hand-off to Igor: temp-file format, Igor paths and the push command."""

from afm_litho.hardware.igor import Igor
from afm_litho.hardware.waves import push_wave, to_igor_path, write_temp_wave


class _DriverFake:
    """Records every command Igor is sent; each Execute2 succeeds with no result."""

    def __init__(self):
        self.commands = []

    def Execute2(self, allow, binary, cmd):
        self.commands.append(cmd)
        return (0, "", "", "")


# --- wave hand-off ----------------------------------------------------------

def test_igor_path_uses_colons():
    assert to_igor_path(r"C:\Data\a\afm_XLitho.txt") == "C:Data:a:afm_XLitho.txt"


def test_temp_wave_has_the_name_header_then_values(tmp_path):
    path = write_temp_wave([1.0, 2.5, -3.0], "XLitho", str(tmp_path))
    with open(path) as fh:
        lines = fh.read().splitlines()
    assert lines[0] == "XLitho_temp"
    assert lines[1:] == ["1", "2.5", "-3"]


def test_push_wave_emits_make_loadwave_copy(tmp_path):
    app = _DriverFake()
    push_wave(Igor(app), "XLitho", [0.0, 1.0], str(tmp_path))
    assert app.commands[0] == "Make/O/N=2 XLitho"
    assert app.commands[1].startswith("LoadWave/O/A/J/D/W/K=0")
    assert app.commands[2] == "XLitho = XLitho_temp"
