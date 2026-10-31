# Wire v3 conformance suite

Judges anything that speaks `docs/migration/plan/phase-5-wire-v3-contract.md`. Run it with
`pytest tests/wire_v3`. Today it runs against the fake Igor (`FakeIgorBridge`, slice W3) and
against `reference_igor.py`, a small model written from the contract so the tests are
themselves tested. Another target: `AFM_LITHO_V3_TARGET=module:factory` (see `target.py`).
Tests marked "sim" need the target to move a simulated clock; on a real rig they are the
checklist to do by hand with the tip in the air.

| Contract section | Test (all in `test_verbs.py` unless noted) |
|---|---|
| 2a HELLO, mismatch | `test_hello_reply_fields`, `test_hello_wrong_proto_is_mismatch_with_same_body`, `test_mismatch_refuses_load_play_set_but_keeps_reads_and_makesafe` |
| 1 rights, 7 safe set | `test_no_hello_means_no_rights_but_the_safe_set_works`, `test_v2_safe_set_still_answers`, `test_second_connection_is_not_the_host`, `test_clearfault_is_host_only` |
| 2a reads, STATUS | `test_status_fields`, `test_read_value_gv_xpt`, `test_read_flag_abortfault_is_read_only_zero_or_one` |
| 2b, 2d ack-paced LOAD | `test_load_replies_after_each_piece_with_progress`, `test_load_reply_is_one_per_piece_and_in_order` |
| 2b PLAY, busy/mark/fault | `test_play_arms_then_reports_and_done`, `test_play_while_active_is_busy`, `test_play_refused_while_mark_set`, `test_play_refused_while_fault_set_and_set_is_not` |
| 2b SET echoes | `test_set_loop_echo_has_error_code`, `test_set_mode_and_value_echo` |
| 1, 2c stamps, useq | `test_every_report_is_stamped_and_useq_has_no_gaps`, `test_ticks_are_dash_routed_and_paced` (sim) |
| 2c ring, FETCH, reconnect | `test_fetch_resends_from_useq_and_reports_the_ring`, `test_ring_keeps_only_the_last_256_and_fetch_says_gone`, `test_captures_are_not_ringed`, `test_reconnect_hello_reports_ring_and_fetch_recovers_the_gap` |
| 5a make-safe sequence | `test_makesafe_is_ok_at_once_then_reports_madesafe_and_sets_mark`, `test_makesafe_runs_the_contract_sequence_once`, `test_repeat_abort_rearms_retract_but_skips_second_td_stop`, `test_com_route_stamps_com_and_runs_same_sequence` |
| 6 queue jump | `test_makesafe_jumps_the_queue_and_drops_the_rest` |
| 5a confirm, vendor withdraw | `test_confirmsafe_clears_mark_only_on_igors_own_low_reading`, `test_confirmsafe_edge_at_one_volt` (0.9 V yes, 1.5 V no), `test_vendorwithdraw_runs_once_per_makesafe_and_sets_fault` |
| 5a backup at 5 s (sim) | `test_backup_clears_mark_at_5s_when_height_is_low`, `test_backup_withdraws_and_faults_at_5s_when_height_is_high`, `test_backup_does_nothing_after_a_confirm`, `test_backup_edge_at_one_volt` |
| 12 CLEARFAULT | `test_clearfault_clears_both_flags_with_no_height_check_and_play_is_accepted` |
| 5b dead socket (sim) | `test_dead_socket_with_tip_down_parks_after_5s`, `test_dead_socket_with_tip_up_does_not_park`, `test_dead_socket_during_a_run_lets_it_finish_then_parks`, `test_dead_socket_park_edge_at_five_seconds` (4.8 s no, 5.2 s yes) |
| 3a, 9 igor-level equality | `test_golden_equivalence.py`: every golden scenario over v3 ends in the frozen Igor commands, same order (the make-safe per section 3a/5a); `test_cypher_igor_reads_that_survive_over_v3`; `test_cypher_igor_gv_reads_survive_over_v3` (F100, closed by W2's STARTUP after HELLO) |
| the suite's own pins | `test_contract_selfcheck.py` (codec, goldens hold no retired park lines, scenario coverage) |

## Rig checklist for the real .ipf (slice W1), Igor 6.37, tip in the air

The .ipf side cannot run here; `docs/migration/plan/w1-ipf-map.md` maps each verb to its Igor
function and to the fake. Head motored well clear, sample out or far below. Start the bridge as
usual (`scripts/start_bridge.py`), then drive the wire from a Python shell on the instrument PC
with the v3 client (`afm_litho.hardware.wire_v3`) or with raw lines on the accepted socket.
Each Igor command below is one command-line entry (no flow control, under 400 characters).

**0. Install and compile**
- `python scripts/install_afm_ipf.py`, recompile. The procedures must compile with no error
  (a compile error disables every user function, AFM_SafePark included).
- Igor: `print kIpfSha` -> 16 hex, equal to `wire_v3.expected_ipf_sha()` on the PC's checkout.
- Igor: `CtrlNamedBackground AFMv3guard, stop` on a fresh start -> no error (a task never
  started; the same check as AFMabortBackup).

**1. HELLO and reads** (send line -> expected reply)
- `HELLO 1 3 rig` -> `REPLY 1 ok proto=3;ipf=3;sha=<kIpfSha>;MicroscopeID=4;clock=<ticks>;ring=0-0;mark=0;fault=0;` then the STARTUP key=values. On the MFP-3D, `MicroscopeID` is its own id.
- From then on the 4 Hz telemetry arrives as `TICK <useq> <tick> 0 0 - <x> <y> <z> <defl> <sum> <amp>` (no more TELEM). Igor: `print root:AFM_v3HostSock == root:AFM_lrSock` -> 1.
- `HELLO 2 2 old` -> `REPLY 2 err mismatch proto=3;...`; then `SET 3 value Output.A 0` -> `REPLY 3 err host`. Send `HELLO 4 3 rig` again to get the rights back.
- `READ 5 value Height` -> `REPLY 5 ok <volts>`, the meter's Z Voltage.
- `READ 6 gv XLVDTSens` -> `REPLY 6 ok <number>`; `READ 7 gv NoSuchGlobal` -> `REPLY 7 ok nan`.
- `READ 8 flag abortFault` -> `REPLY 8 ok 0`; `READ 9 flag other` -> `REPLY 9 err bad flag other`.
- `READ 10 str PIDSLoop.2.OutputChannel` -> `REPLY 10 ok PIDSLoop.2.OutputChannel=<channel>;`; `READ 11 str NoSuch.Name` -> `REPLY 11 err td_ReadString(NoSuch.Name) failed`.
- `READ 12 xpt Crosspoint` -> `REPLY 12 ok ...InFast=<FilterOut|ACDefl|...>;...`.
- `STATUS 13` -> `REPLY 13 ok pass=none;run=-;step=0/0;mark=0;fault=0;tipdown=0;loop2=none;sock_dead_s=0;ring=0-0;tick=<ticks>;`.

**2. SET echoes** (tip in the air)
- `SET 20 value Output.A 0` -> `REPLY 20 ok value Output.A 0`.
- `SET 21 loop 2 Deflection -10 0 1000 Height 0 150` -> `REPLY 21 ok loop 2 Deflection -10 0 1000 Height 0 150 err=0`; `STATUS` -> `tipdown=0;loop2=-10,0,150`.
- `SET 22 mode Contact` -> `REPLY 22 ok mode Contact` (only with the tip withdrawn, as today).

**3. LOAD and a move** (small square, well inside the range)
- `LOAD 30 XW 4 <x0;x1;x2;x3 in sensor V>` -> `REPLY 30 ok XW 4/4`; the same for `YW`.
- `PLAY 32 move m1 XW YW -200 256 200` -> `REPLY 32 ok armed m1`, `REPORT .. sock m1 armed`, CAPs `CAP <useq> <tick> 0 m1 0 XSensor ...` and `... YSensor ...`, `REPORT .. m1 step 0 1 256`, `REPORT .. m1 done 256` (points per channel).
- A second PLAY while the first runs -> `REPLY <seq> err busy`.
- Igor history shows the same td_ lines as a v2 transit: `td_xSetOutWavePair(0,"0",...)`, `td_xSetInWavePair(0,"0","XSensor",...)`, `td_WriteString("Event.0","Once")`.

**4. Make-safe** (engage in air first: `SET 40 loop 2 Deflection 0.5 0 1000 Height` drives Height to 150 V)
- `MAKESAFE 41` -> `REPLY 41 ok`, then `REPORT .. 1 0 sock - madesafe stop`. Igor ran `td_Stop()`, `td_WriteValue("Output.A",0)`, the -10 V re-arm. Height falls under 1 V.
- `CONFIRMSAFE 42` -> `REPLY 42 ok height=<under 1>`, `REPORT .. - confirmed <h>`; `STATUS` -> `mark=0;tipdown=0`.
- Repeat abort while retracting: engage, `MAKESAFE`, then at once Igor `AFM_SafePark()` and `MAKESAFE` again -> one `td_Stop()` only; madesafe REPORTs stamped `sock`, `com`, `sock`.
- `PLAY` while `mark=1` -> `err mark`. `SET` while `mark=1` -> accepted.
- Backup with the service stopped: engage, `MAKESAFE`, stop the service. At 5 s Igor confirms on its own; after a reconnect `HELLO` + `FETCH <n> <first useq>` shows `REPORT .. - backup <h> cleared`.
- Vendor withdraw: `VENDORWITHDRAW 50` -> `REPLY 50 ok ran`, again -> `REPLY 51 ok already`; `READ flag abortFault` -> `ok 1`; `PLAY` -> `err fault`. `CLEARFAULT 52` -> `REPLY 52 ok fault=0`; `STATUS` -> `mark=0;fault=0`; PLAY is accepted again.

**5. Queue jump**
- Send `SET 60 value Output.A 0`, `PLAY 61 move q XW YW -200 256 200` and `MAKESAFE 62` in one write -> `REPLY 62 ok` first, then `REPLY 60 err dropped`, `REPLY 61 err dropped`. If the three lines arrive in two socket reads, the drop applies only to lines in the read that holds the MAKESAFE: note it.

**6. Dead socket** (contract 5b)
- Engage in air (`SET loop 2 ... 0.5 ...`), then kill the service. Expect within about 5.0-5.4 s: `td_Stop()`, bias 0, the -10 V re-arm in Igor's history; Height falls. Igor: `print root:AFM_v3DeadSince` -> the ticks of the first failed send (confirms that `SOCKITsendmsgF` reports the closed link; if it stays -1, the guard never starts: stop and report).
- With the tip lifted (`SET loop 2 ... -10 ...`) and the service killed: nothing runs.
- During a move: kill the service mid-run; the run finishes, then the park follows 5 s after the kill. After the reconnect (`start_bridge`, `HELLO`, `FETCH`), the ring holds the run's `done` and `madesafe socket`.

**6b. Restart during a run** (review M1; Phase 6 decision 6: a restarted host makes safe first)
- Load a chain longer than a few seconds, `PLAY <seq> lithochain c1 ...`, kill the service, then
  `start_bridge` and `HELLO` before the chain ends. Then `STATUS` until the chain has ended:
  `pass=none;run=-;` (the run ended on its real DONE although the reconnect blanked the uid
  globals). `FETCH` shows the run's `step` and `done` REPORTs. A PLAY is then not `err busy`.
  Igor: `print root:AFM_v3Run` -> empty. If the service sends MAKESAFE first (K3), the run ends
  as `aborted` instead and STATUS shows the same.

**6c. SOCKIT and background-task facts** (review S2, S4)
- Igor: `print AFM_SendStr(root:AFM_lrSock, "\n")` (one empty line, 1 byte) on a live link and after the service has been
  killed. Record both values: success must be 0 or the byte count, the dead link anything else
  (or a runtime error). If the dead link also returns 0, the dead-socket park never starts: stop
  and report.
- Every HELLO runs `CtrlNamedBackground AFMv3guard, stop`, usually on a task never started.
  On a fresh Igor: send `HELLO 1 3 rig` and check the history shows no error; repeat the HELLO.

**7. v2 unchanged**
- With `WIRE_VERSION=2`, run the usual litho and scan checks: Igor's history shows the same lines as before, and the link still carries TELEM (no HELLO, so no TICK).
