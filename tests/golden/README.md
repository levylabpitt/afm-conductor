# Golden traces

Each `*.golden.json` here is the frozen trace of one scenario in `scenarios.py`, recorded on the
Igor stub (`FakeIgorBridge`) as a Cypher (`microscope_id` 4) with the fixed model in `fixture/`.
Each `*.mfp3d.golden.json` is the same scenario recorded as an MFP-3D (`microscope_id` 1), at the
wire level only. `tests/test_golden_scenarios.py` re-records every scenario and judges it
against its files.

## What a scenario may do

Only FLEX verbs, through `FlexDoor` (the same `handle_message` the ZMQ command socket calls).
No service internals, so the same scenario can judge a later architecture. Every parameter a
scenario relies on is written out.

Two more doors, both outside the service:

- **`http`**, the cockpit's HTTP door (`HttpDoor`), only for what FLEX has no verb for: Show Tip,
  the scan Stop, the relative setpoint write, the plain withdraw. Such a golden is stamped
  `door: flex+http`.
- **`inst`**, the instrument side (`Inst`): set a value Igor would report, make the next move
  land short, park the stub at a point (mid-line, mid-stroke, a chain seam) so that an abort lands
  exactly there, or take COM down. It never touches the service.

A scenario takes the doors it needs by parameter name (`door`, `inst`, `http`).

## The two levels

| Level | Source | `actuation` | `reads` |
|---|---|---|---|
| `wire` | the stub's tap | exact, ordered | sorted distinct reads, UIDs dropped; frozen ⊆ new |
| `igor` | the expander fed the tap's raw transcript | exact, ordered | same rule |

`wire.actuation` also holds `@FILL` (the stub completed a capture: Igor's fill callback runs
there) and COM entries spelled as expander pseudo-verbs (`@SAFEPARK`, `@DOSCANFUNC <name>`).
Waves are `wave(name,n,first,last,sha)`, the same summary and hash at both levels.

The MFP-3D files hold the `wire` level only. The end-of-line level is Cypher-first (the
expander's table is the Cypher's); the MFP-3D wire files exist so that a Phase 1 refactor of
shared code (channel table, dead code) cannot break the MFP-3D silently. The MFP-3D is not
worked on.

`known_defects` lists the defects a golden freezes as today's behaviour (`KNOWN_DEFECTS` in
`scenarios.py`); an intended fix re-freezes the golden and removes the entry, and the comparison
fails if the two disagree. A test keeps the lists honest:

- **defect-1** (the park's overlapping loop-2 commands; SLICES F5) is fixed by slice AB and listed
  in no golden: the park is Igor's one abort function (`AFM_SafePark`: td_Stop once per abort, bias
  0, the -10 V re-arm, a 5 s backup), and the host sends no STOPLOOP or RAMP of Z, only
  `ABORTCONFIRM` once Height reads under 1 V and the `ABORTFAULT` read.

At the `igor` level Igor function names compare case-insensitively (`td_ReadValue` equals
`td_readValue`, SLICES F22); string arguments (channels, parameters, events) stay exact.

Each file also stamps the golden format version, the three `.ipf` hashes and the fixture hashes.
Any of those changing fails the test.

## Scenario catalogue

Door is FLEX unless stated. Every row is also an MFP-3D wire golden. Time is one recording.

| Scenario | What it freezes | Time |
|---|---|---|
| `scan_chained_contact` | 32 px, 2 lines, contact, at the current engagement | 2 s |
| `scan_chained_ac` | the same in AC (error channel Amplitude), instrument already in AC | 2 s |
| `scan_abort_mid_line` | a 4-line chained scan, token-free `abort` after line 1 | 2 s |
| `scan_stop_deferred_withdraw` | the cockpit Stop (`flex+http`): the retract is deferred until the pass halts | 2 s |
| `continuous_two_frames` | continuous imaging, 2 frames, then the loop's safe-park | 3 s |
| `engage_contact_tune_withdraw` | contact engage, live Z-gain tune, withdraw | 3 s |
| `engage_ac_tune_withdraw` | AC engage (negated gain on Amplitude), tune, withdraw | 3 s |
| `mode_switch` | SETMODE AC then Contact while withdrawn | 2 s |
| `show_tip` | SHOWTIP ON and OFF (`flex+http`) | 1 s |
| `connect_reads` | the connect alone: STARTUP, GETGV; no actuation | 1 s |
| `write_refused_in_ac` | startWrite in AC: READXPT, refused, nothing armed | 1 s |
| `safe_park` | token-free `safePark`: COM `AFM_SafePark()`, then the Height confirm | 1 s |
| `safe_park_com_down` | no COM: the park falls back to the wire `ABORT` | 1 s |
| `safe_park_vendor_withdraw` | a park whose verify fails: `DoScanFunc("StopEngageButton")` once | 3 s |
| `litho_single` | one straight wire, absolute setpoint | 11 s |
| `litho_lateral_refused` | `litho_single` with bank 2 refusing the UserIn2 + Lateral pair: UserIn2 armed alone | 11 s |
| `litho_multi_object` | a filled pad (hover moves between its strokes), then a wire (full transit), end-of-write withdraw | 16 s |
| `litho_chain_two_chunks` | one stroke of about 100,000 points: LITHOCHAIN over two chunks | 13 s |
| `write_relative_setpoint` | the cockpit write (`flex+http`): withdraw records free air, `setpoint_relative` becomes absolute | 13 s |
| `abort_in_transit` | two wires; the abort lands during the second wire's transit | 18 s |
| `abort_mid_stroke` | the abort lands while a stroke plays | 15 s |
| `abort_at_chain_seam` | the abort lands between the two chunks of a chain | 18 s |
| `withdraw_flex` | contact engage, then the FLEX `withdraw` (one golden per withdraw door) | 3 s |
| `withdraw_http` | contact engage over FLEX, then the cockpit's `POST /api/withdraw` (`flex+http`) | 3 s |
| `cockpit_abort_mid_stroke` | the cockpit Abort button (`POST /api/safe_park`) while a stroke plays (`flex+http`) | 15 s |
| `cockpit_abort_in_transit` | the cockpit Abort button during the second wire's transit (`flex+http`) | 18 s |
| `litho_abort_http_mid_stroke` | `POST /api/litho/abort` while a stroke plays (`flex+http`) | 15 s |
| `transit_retry` | a transit lands short once, two reads agree, the retry arrives | 12 s |
| `transit_failure_withdraw` | a transit lands short twice: it withdraws and raises, the write fails | 11 s |
| `multi_pass` | wire, chained scan, wire: the SLICES F9 shape, frozen as today | 21 s |
| `interlock_trip` | engage, DetectorSum collapses; the interlock trips after 5 s and the fail-safe thread parks | 6 s |
| `deadman_during_write` | the script goes silent during a stroke; the 5 s dead-man aborts and parks | 20 s |
| `deadman_engaged_withdraw` | silent while engaged with `withdraw_on_deadman`: the dead-man withdraws | 9 s |
| `engaged_idle_cap` | engaged with nothing running: the engaged-idle cap (1 s for the recording) withdraws | 5 s |

Where the catalogue cannot reach a path through a door, it is listed under Gaps below.

## Igor entry points

Every path into Igor. `goldened` names the scenarios whose golden holds it. `below the line` means
the call and its arguments are in the golden but what Igor does inside vendor code is not
modelled. `gap` means no scenario can reach it through a door today (reason in Gaps). The Igor
mirror (`service/mirror.py`) is gone (F16), so it has no row.

| Entry point | Reached by | Status | Scenarios |
|---|---|---|---|
| socket `STARTUP` | the handshake at connect | goldened | `connect_reads` |
| socket `GETVALUE` | every scalar read: guards, verifies, engage polls | goldened | `connect_reads` `engage_contact_tune_withdraw` |
| socket `GETGV` | calibration reads at connect and on refresh | goldened | `connect_reads` |
| socket `READXPT` | the mode discriminator: AC guard of a write, mode reconcile | goldened | `write_refused_in_ac` |
| socket `SETVALUE` | scalar writes; no door sends one since the MFP-3D set-down became a loop re-arm | gap | none |
| socket `SERVOLOOP` | `ir_SetPISLoop`: engage, tune, withdraw, transit and scan loops | goldened | `engage_contact_tune_withdraw` `scan_chained_contact` |
| socket `STOPLOOP` | no door sends one since slice AB (the park has no host Z step) | gap | none |
| socket `RAMP` | the hover move between strokes | goldened | `litho_multi_object` |
| socket `WAVE` | out-waves, in every scan, write and transit | goldened | `scan_chained_contact` |
| socket `WAVEAPP` | a wave over 8,192 points, sent in chunks | goldened | `litho_chain_two_chunks` |
| socket `ARMFIRE` | one stroke: litho convenience arm and fire | goldened | `litho_single` |
| socket `OUTWAVE` | single-channel out-wave bind (the twin only) | gap | none |
| socket `OUTPAIR` | the XY pair: transit, scan | goldened | `scan_chained_contact` |
| socket `INWAVE` | single-channel capture arm; no door sends it since the unchained scan was retired | gap | none |
| socket `INPAIR` | the capture pair: transit, scan | goldened | `scan_chained_contact` |
| socket `FIRE` | fire an event: transit | goldened | `abort_in_transit` |
| socket `SCANCHAIN` | the chained scan | goldened | `scan_chained_contact` `scan_chained_ac` |
| socket `LITHOCHAIN` | an oversize stroke as chunks | goldened | `litho_chain_two_chunks` |
| socket `SHOWTIP` | the Show Tip toggle (display hint; the tip poll always runs) | goldened | `show_tip` |
| socket `SETMODE` | the imaging mode switch | goldened | `mode_switch` |
| socket `ABORT` | the wire fallback when COM `AFM_SafePark()` fails | goldened | `safe_park_com_down` |
| socket `ABORTCONFIRM` | the park read Height under 1 V: Igor clears its abort flag | goldened | `safe_park` `abort_mid_stroke` |
| socket `ABORTFAULT` | the park (and the connect) read Igor's 5 s backup fault, read only | goldened | `safe_park` |
| socket `CLEARFAULT` | the operator's fault clear resets Igor's abort flag and fault | gap | none |
| `AFM_AbortBackup` (AFMabortBackup task) | Igor's own 5 s confirm, only when the host has not confirmed | gap | none |
| COM `AFM_SafePark()` from the ARC | `HardwareARC.abort`: the chained scan driver, a litho run | goldened | `scan_stop_deferred_withdraw` |
| COM `AFM_SafePark()` from the fail-safe | `send_safe_park` on the watchdog thread: Abort, interlock, dead-man | goldened | `safe_park` `interlock_trip` `deadman_during_write` |
| COM `DoScanFunc("StopEngageButton")` | `HardwareARC.vendor_withdraw`, the park's one fallback | goldened | `safe_park_vendor_withdraw` |
| `AFM_LithoRunStart` / `AFM_LithoRunEnd` | run start and end, typed by hand on the instrument PC | gap | none |
| `AFM_LithoRunDryRun` | the diagnostic dry-run switch, by hand | gap | none |
| `MainPopupFunc("ImagingModePopup_0", ...)` | called by `AFM_SetMode` for SETMODE | below the line | `mode_switch` |
| capture-fill callbacks (`AFM_Done`, `AFM_ScanAdvance`, `AFM_LithoAdvance`) | the controller fires them when a capture fills (`@FILL`) | goldened | `scan_chained_contact` `litho_chain_two_chunks` |
| tip poll `AFM_TipTick` (TELEM) | Igor's always-on scalar poll, host-bound | goldened | `interlock_trip` |

## Gaps

Paths no scenario reaches through a door, and findings from recording the catalogue. Nothing here
was fixed (no new features, no bug hunt).

- **Socket `SETVALUE`.** The MFP-3D litho set-down was its only sender; it re-arms loop 2 now
  (slice M2).
- **Socket `OUTWAVE` and `INWAVE`.** No door sends them on hardware since the unchained scan was
  retired (SLICES 1.2b); only the twin binds a single out-wave.
- **`AFM_LithoRunStart`, `AFM_LithoRunEnd`, `AFM_LithoRunDryRun`.** Typed by hand on the
  instrument PC; the service never calls them. The expander models them (`@RUNSTART`, `@RUNEND`,
  `@DRYRUN`); the golden starts at the state `AFM_LithoRunStart` leaves.
- **Hardware dynamics.** The stub's readbacks are static, so anything that needs the loop to move
  over time (a real settle, a real overshoot) is outside the goldens until the stub grows a safety simulation
  (SLICES 0.5b).
- **Result bookkeeping is not in a golden.** An abort that lands between two lines of a chained
  scan reports the run as `completed: true, aborted: false`: the chain driver sets `aborted` only
  when a line arrives after the stop flag (`hw_scan.run_hw_scan_chained`). `scan_abort_mid_line`
  freezes the commands, not that result.
- **The park/worker order.** After an abort during a write, the worker's end-of-run withdraw
  (`SERVOLOOP`) and the fail-safe's `STOPLOOP`/`RAMP` are ordered by the park joining the worker
  before any Z command. The golden holds that order. It broke in 1 recording in 14 until the
  harness waited for the previous recording's pump to finish: that pump clears
  `hub.litho_worker` when it ends, and when it ended after the next write had started it removed
  the new write's handle, so the park skipped its join. The same overlap is possible on a real
  station that starts a write right after another ends.

## Regenerating

Only when a change to the end commands is approved (or a scenario is added):

```
AFM_LITHO_REGEN_GOLDENS=1 ./.venv/Scripts/python.exe -m pytest -q tests/test_golden_scenarios.py
```

Commit the regenerated files on their own, with the reason in the message, and never edit one
by hand (the test checks each file is canonical). Review the diff of `igor.actuation`: that is
what the AFM executes.

`AFM_LITHO_GOLDEN_REPEAT=20` makes the determinism test record each scenario 20 times.
The scenarios that take over 8 s (the litho ones, `multi_pass` and the dead-man) are run with
`AFM_LITHO_GOLDEN_REPEAT=5`.
