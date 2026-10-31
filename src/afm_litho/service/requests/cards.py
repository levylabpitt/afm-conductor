"""The FLEX HELP cards: what each verb's HELP prints and the honest time bound it publishes.

One copy of these numbers and words. A row takes its bound and card text from here, and the
service's verb list (`flex_dispatch.REGISTRY`) is built from the rows in this order, so HELP and
getCapabilities print the same bytes whatever order the rows are gathered in.
"""

from __future__ import annotations

from typing import NamedTuple

from ..host.scan_cmds import mode_settle_s


class Card(NamedTuple):
    params: str
    result: str
    bound_s: float     # the worst case a client should wait; getCapabilities publishes it
    doc: str


CARDS: dict[str, Card] = {
    "ACK": Card(
        params='-',
        result='"ACK"',
        bound_s=0.005,
        doc='Liveness. Reads no hub state, so it answers while a write runs.'),
    "IDN": Card(
        params='-',
        result='{Manufacturer, Model, Serial Number, Firmware, Backend}',
        bound_s=0.02,
        doc=('Identity. `Backend` is the anti-fallback field: hardware | twin | disconnected,'
             ' derived live from the bridge.')),
    "HELP": Card(
        params='{command?: str} | [name]',
        result='[method names] | {method, params, result, bound_s, doc}',
        bound_s=0.02,
        doc='Discovery -- the only mechanism FLEX has.'),
    "getState": Card(
        params='-',
        result='disconnected|idle|approaching|engaged|scanning|writing|measuring|parked|fault',
        bound_s=0.05,
        doc=('The poll verb. Pure in-memory read; never issues a bridge read. See the derivation'
             ' table in the flex_dispatch module docstring.')),
    "getStatus": Card(
        params='-',
        result=('{state, backend, run_id, run_kind, progress, detail, safe_park, commander,'
                ' deadman_armed, bridge}'),
        bound_s=0.05,
        doc=('The richer sibling of getState; getState stays a bare string so it registers'
             ' cleanly as a FLEX Parameter.')),
    "getCapabilities": Card(
        params='-',
        result=('{microscope_id, is_cypher, backend, modes, max_outwave_points, in_banks,'
                ' inline_cap_bytes, method_bounds, twin_only, ports}'),
        bound_s=0.02,
        doc=('Lets a script refuse rather than discover a limit by tripping it. method_bounds is'
             ' what the FLEX driver derives its timeouts from.')),
    "getInstrument": Card(
        params='-',
        result=('{platform, z_out_channel, ..., overrides_tail, last_write_params,'
                ' last_scan_params, xy_cal}'),
        bound_s=0.05,
        doc=('The instrument model (docs/INSTRUMENT_MODEL_SPEC.md section 10), read only -- same'
             ' shape as GET /api/instrument. Served on BOTH sockets.')),
    "getLastPark": Card(
        params='-',
        result='{verified, method, z_out_v, zsensor_v, defl_v, critical, reason, ts} | {}',
        bound_s=0.05,
        doc=('The most recent park outcome (E1), read only -- same shape as GET /api/park/last.'
             ' {} until the first safe_park. Served on BOTH sockets.')),
    "getEstimate": Card(
        params='-',
        result='{total_s, trajectory_s, objects:[{id, seconds, length_um, speed_um_s, enabled}]}',
        bound_s=0.1,
        doc=("The write-time estimate for the loaded design, computed from the service's own"
             ' enabled set (docs/INSTRUMENT_MODEL_SPEC.md sections 4/10). Per-object `seconds` is'
             " trajectory time; `trajectory_s` sums the enabled objects' trajectory and `total_s`"
             ' adds the per-object overhead + margin the write-start estimate charges (one'
             ' estimator). Disabled objects are listed with enabled:false and excluded from both'
             ' totals. Same shape as GET /api/litho/estimate.')),
    "getTelemetry": Card(
        params='-',
        result='{t, age_s, x_um, y_um, zdrive_v, defl_v, sum_v, amp_v, row, backend}',
        bound_s=0.02,
        doc=('The last cached telemetry frame -- never a new instrument round-trip. On the twin'
             " the same shape is built from the twin's own state.")),
    "getPattern": Card(
        params='-',
        result=('{name, objects:[{index, name, id, kind, voltage, speed, points, segments,'
                ' strokes, enabled, written, vertices_um, properties}], frame}'),
        bound_s=0.1,
        doc=('The loaded design as a summary, with point COUNTS rather than points. Each object'
             ' also carries the shared listing block -- `id`, `vertices_um` (the source polyline;'
             " a chiral wire's base axis) and `properties` (voltage, speed_um_s, enabled, and the"
             ' chiral/fill block setObjects edits).')),
    "getWritten": Card(
        params='-',
        result=('{objects:[{index, name, object, segments, voltage, length_um, points, t_start,'
                ' t_end}], run_id, aborted, backend}'),
        bound_s=0.1,
        doc=('What this session has written so far. ONE ENTRY PER STROKE, not per object: a'
             ' filled shape arrives as several entries named "<object>#k"; group by `object` for'
             ' per-object totals. A CHAIN CHUNK IS NOT AN ENTRY -- an over-cap stroke is several'
             ' out-waves but one continuous tip path, so one entry; the extra arms show up in'
             ' `segments`.')),
    "getScanResult": Card(
        params='{run_id?, inline?, decimate?}',
        result=('{run_id, kind, backend, completed, aborted, partial, shape, channels, files,'
                ' stats, preview?, data?}'),
        bound_s=0.2,
        doc=("run_id omitted = the last COMPLETED scan. `completed` is true only after the pump's"
             ' persistence block finished, so the paths in `files` are whole.')),
    "getWriteResult": Card(
        params='{run_id?, inline?, decimate?}',
        result='as getScanResult, kind=write, plus `trace` when inline',
        bound_s=0.2,
        doc=("The written pass's raw capture (UserIn2 measured bias + sensors). With `inline` it"
             ' also carries `trace`: the per-object written path with the tip voltage along it,'
             ' from memory and bounded -- the x-axis a lock-in series gets correlated against. An'
             ' ABORTED run still returns its partial trace.')),
    "acquireControl": Card(
        params='{client, deadman_s?=30, withdraw_on_deadman?=false}',
        result='{token, expires_s, deadman_s, withdraw_on_deadman}',
        bound_s=0.05,
        doc=('Become the single commander. Refused -32040 control_held while an operator holds'
             ' control or another script holds an unexpired token. deadman_s is clamped to [5,'
             ' 300] and can never be disabled.')),
    "releaseControl": Card(
        params='{token}',
        result='{ok, stale}',
        bound_s=0.05,
        doc=('Give up control. NEVER an error: a stale or unknown token returns {ok:true,'
             ' stale:true}, so a script can call it from a finally block.')),
    "heartbeat": Card(
        params='{token}',
        result='{state, expires_s, valid}',
        bound_s=0.05,
        doc=('Refresh the dead-man. Any OTHER request bearing a valid token refreshes it too, so'
             ' a polling loop refreshes for free.')),
    "abort": Card(
        params='{token?, scope?}',
        result='{ok, state, reason, safe_park}',
        bound_s=10.0,
        doc=('TOKEN-FREE emergency stop: cooperative stop of whatever is in flight PLUS a'
             ' verified retract, through the same safe_park the GUI Abort fires. Never the'
             ' non-retracting COM AFM_Abort.')),
    "safePark": Card(
        params='{reason?}',
        result='{ok, state, backend, transport, verified, z_v, critical, verify_timeout}',
        bound_s=10.0,
        doc=('TOKEN-FREE verified retract. Reports verified:false with verify_timeout:true rather'
             ' than blocking on a stalled bridge.')),
    "getMode": Card(
        params='-',
        result='{mode, engaged, gate, backend}',
        bound_s=0.05,
        doc=("The instrument's imaging mode, plus the live engaged/scanning gate that says"
             ' whether setMode would be accepted right now. A pure memory read, so the READ'
             ' socket serves it too.')),
    "setMode": Card(
        params='{token, mode:contact|ac}',
        result='{ok, mode, backend}',
        bound_s=round(mode_settle_s() + 1.0, 3),   # the settle it waits out, plus send and loop
        doc=('Switch contact <-> AC. REFUSED -32021 while the tip is engaged, scanning or'
             ' writing: the vendor switch changes the Z feedback channel and gain sign and does'
             ' NOT withdraw, so with the tip down it is a crash. Withdraw first. Returns only'
             ' after a settle, so a following startApproach cannot engage into a half-applied'
             ' mode change.')),
    "startApproach": Card(
        params='{token, setpoint?, pgain?, igain?, settle_s?, mode?}',
        result='{run_id, est_s, state}',
        bound_s=1.0,
        doc=('START+POLL fine engage: closes the Z feedback loop so the tip tracks the surface.'
             ' Poll getState -> approaching, then engaged. A FAILED engage fires safe_park and'
             ' lands in parked, and the run record carries why.')),
    "withdraw": Card(
        params='{token?}',
        result='{ok, z_v, verified, verify_timeout, backend, state, snapshot}',
        bound_s=6.5,
        doc=('Retract, from any state. TOKEN OPTIONAL and no backend gate, for the same reason'
             ' abort and safePark are token-free: getting the tip OFF the surface must never be'
             ' refused by bookkeeping. `verified` means Z parked within 0.5 V with no read that'
             ' timed out.')),
    "setZGain": Card(
        params='{token, pgain?, igain?, setpoint?, mode?}',
        result='{ok, backend, snapshot}',
        bound_s=1.0,
        doc=('Re-arm the running Z loop with new P/I gains, without withdrawing. HARDWARE ONLY --'
             ' on the twin it answers -32010 rather than reporting a tune it did not make.')),
    "startScan": Card(
        params=('{token, size_um, pixels, lines, line_rate_hz, x_offset_um, y_offset_um,'
                ' angle_deg?, direction?, trace_retrace?, mode?, setpoint?, engage?, pgain?,'
                ' igain?}'),
        result='{run_id, est_s, state, backend, config}',
        bound_s=1.0,
        doc=('START+POLL raster. Poll getState, then getScanResult(run_id) -- whose `completed`'
             ' is gated on persistence, so the paths in `files` are whole. On hardware a'
             ' `setpoint`, `pgain` or `igain` without `engage:true` is REJECTED -32602 rather'
             ' than accepted and ignored. pgain/igain are PANEL gains (the setZGain convention)'
             " and override this scan's default Z gains when given.")),
    "setContinuous": Card(
        params=('{token, enabled, config?, max_frames?, max_seconds?, interval_s?,'
                ' withdraw_on_stop?, rail?}'),
        result=('{ok, enabled, state, max_frames, max_seconds, interval_s, withdraw_on_stop,'
                ' rail}'),
        bound_s=5.0,
        doc=('Arm or stop the auto-repeat imaging loop; the state stays `scanning` for as long as'
             ' it runs and every frame gets its own run_id. `rail` may only TIGHTEN the'
             ' rail-dwell guard, never loosen it. Disabling is never refused for backend reasons'
             ' -- a dead bridge is exactly when you want it stopped.')),
    "loadPattern": Card(
        params='{token, source?|objects?}',
        result=('{name, object_count, objects:[{index, name, id, kind, points, segments, strokes,'
                ' vertices_um, properties, profile?, speed_achieved_um_s?}], total_points,'
                ' bbox_um, segments, strokes, multi_segment_objects, max_outwave_points,'
                ' duration_s}'),
        bound_s=2.0,
        doc=('Compile and validate a design. ARMS NOTHING -- no claim, no thread, no tip motion.'
             ' EXACTLY ONE of `source` or `objects`. `source` is "auto", a design name in'
             ' data/designs, a file path, or an inline SVG document. `objects` orders geometry BY'
             ' PARAMETERS -- [{kind:"chiral", points_um, lambda_um, y_amp_um, v0, v_k, phase_deg,'
             ' hand?, speed_um_s?}] for a sinusoidal wire whose tip voltage is modulated at a'
             ' chosen PHASE to the deviation, and [{kind:"region", shape:"rect", x_um, y_um,'
             ' w_um, h_um, pitch_um, voltage, fill?, speed_um_s?}] for a filled rectangle at a'
             ' SIGNED voltage (an erase is a negative one); `angle_deg` sets the sweep direction'
             ' of a `directional` fill and is refused on the others, which ignore it. A design'
             ' file carries no tip voltage, so neither could be ordered through `source` at all.'
             " A chiral entry's `profile` echoes what was BUILT -- including `y_amp_built_um` and"
             ' `curvature_clamped_fraction`, the magnitude of a curvature clamp and not just its'
             ' boolean -- and any entry carries `speed_achieved_um_s` when the tick quantisation'
             ' floored the speed asked for. NO SIZE REFUSAL ON THE OUT-WAVE CAP: an object over'
             ' it CHAINS. There IS a size policy on `objects` -- per-object extent (2,000 um),'
             ' per-object and whole-pattern compiled-point budgets, and a fill pitch no wider'
             ' than half the shape -- all checked as arithmetic BEFORE anything is rendered, so'
             ' an absurd request is -32602 in milliseconds; a build or compile that still runs'
             " past this verb's own published bound is -32021 refused with the previous pattern"
             ' left untouched. `strokes` is continuous tip paths (one transit and one getWritten'
             ' entry each); `segments` is out-waves armed, which exceeds it only where a stroke'
             ' is over max_outwave_points and is fired as a seamless chunk chain -- tip down and'
             ' bias live across the boundary. Those objects are listed in multi_segment_objects'
             " and named in a WARNING. NO max_segment_points: the compiler's 8,192-point cut is"
             ' opt-in and off (it was a lift mid-stroke).')),
    "startWrite": Card(
        params=('{token, objects?="all"|[idx|id], setpoint, setpoint_reference?="absolute"|"free_air",'
                ' amp_gain?, deadman_s?, twin_speed?}'),
        result='{run_id, est_s, objects, segments, state, backend}',
        bound_s=1.0,
        doc=('START+POLL lithography. `setpoint` (volts) is REQUIRED; `setpoint_reference` is "absolute" (default: used as given) or "free_air" (volts above the fresh free-air deflection). CONTACT-ONLY: on hardware the AC-guard reads the live'
             ' crosspoint and refuses -32021 in AC/tapping, and an unreadable crosspoint refuses'
             ' rather than assuming contact. `objects` is "all" (every ENABLED object, in pattern'
             ' order), or a list mixing integer indices and string ids; an unknown id is -32602.'
             ' Poll getState -> writing, then getWriteResult(run_id). A second start while one'
             ' runs is -32020 busy with the in-flight run_id, never queued.')),
    "abortWrite": Card(
        params='{token?}',
        result='{ok, state, backend, safe_park}',
        bound_s=10.0,
        doc=('TOKEN-FREE stop of a write PLUS a verified retract, through the same safe_park the'
             ' GUI Abort button fires. The partial result stays retrievable: getWriteResult'
             ' reports aborted:true with the objects that did complete, and getWritten lists'
             ' exactly those.')),
    "setObjects": Card(
        params='{token, updates:{id:{property:value}}}',
        result=('{name, object_count, objects:[{index, name, id, kind, points, segments, strokes,'
                ' vertices_um, properties, profile?}], total_points, bbox_um, segments, strokes,'
                ' multi_segment_objects, max_outwave_points, duration_s}'),
        bound_s=2.0,
        doc=('Edit the loaded pattern in place -- a map of {id: {property: value}} merged onto it'
             ' and recompiled. ARMS NOTHING and moves no tip. Settable properties: `voltage`,'
             ' `speed_um_s`, `enabled`, plus `chiral` (a path only) and `fill` (a'
             ' rect/taper/shape only). `chiral: {lambda_um, y_amp_um, v_k, phase_deg, hand}`'
             ' rebuilds the wire through the same chiral_profile the cockpit uses (v0 IS the'
             " object's voltage); merging needs only the fields that change, but turning"
             ' chirality on from a plain path needs all five. `chiral: null` collapses the wire'
             ' back to its plain base axis. `fill` merges likewise. The WHOLE call is atomic: an'
             " unknown id/property, a wrong type, or a property invalid for an object's kind is"
             ' -32602 naming the id and field, with the previous pattern kept -- as it is when a'
             " recompile runs past this verb's bound. REFUSED -32020 busy while a write is in"
             ' flight. Returns the same per-object listing as loadPattern and getPattern.')),
}
