#pragma rtGlobals=1
#pragma IgorVersion=6.0
#pragma version=0.9

// Explicit dependencies (Igor loads these first). The bridge CALLS:
//   * AFM_SendStr / AFM_SockConnect / AFM_SockClose  -> afm_sockit.ipf
//   * AFM_MacroStartup                               -> afm_igor.ipf
// Both call Asylum functions (GV/td_) or the SOCKIT XOP, so they live in User
// Procedures and compile WITH this file once the MFP3D software is loaded.
// Declaring them here means a single `#include "afm_litho_run"` pulls the whole
// bridge in -- you never have to remember afm_sockit/afm_igor separately.
#include "afm_sockit"
#include "afm_igor"

// afm-litho: the THIN Igor lithography bridge.
//
// Per the whitepaper, Igor is only a thin "arm/fire/report" agent -- the
// armed waves run autonomously in the controller at 50 kHz, so the host never
// touches data fidelity. ALL of the smarts live in the Python DRIVER:
//   * the Python driver reads calibration out of Igor (afm_igor.ipf getters),
//   * does every conversion (microns->LVDT volts, tip volts->DAC volts) and the
//     interpolation / capture-length math,
//   * orchestrates the per-object cycle (lift, move, set down, arm, fire, wait
//     for the DONE report, lift, zero, next),
//   * and talks to this bridge over SOCKIT (transport is a swappable detail).
//
// This file therefore contains NO arithmetic and NO multi-step choreography.
// Every function is a thin, readable wrapper around one cryptic td_/ir_
// primitive, so a reader can see intent at a glance instead of decoding
// td_xSetOutWavePair.
//
// The DISPATCH LISTENER (AFM_DispatchTick, below) is the bridge's ear: a
// background task that reads the driver's protocol lines off the SOCKIT link,
// calls the matching wrapper, and writes the typed reply back. It is the exact
// mirror of the Python test stub's _dispatch (hardware/igor_stub.py), so the
// driver cannot tell real Igor from the stub. The wire protocol both ends share
// is documented in hardware/bridge_protocol.py. UID-CORRELATED (protocol v2): a
// request that expects a reply carries a <uid> as the FIRST token after the verb,
// and AFM_Dispatch echoes it in every reply so the Python single-reader router can
// deliver each reply to the right waiter with no stealing during concurrent traffic:
//   driver -> bridge : STARTUP <uid> <ver> | GETVALUE <uid> <name> | GETGV <uid> <name> |
//                      READXPT <uid> <group> | ARMFIRE <uid> ... | FIRE <uid> ... |
//                      SCANCHAIN <uid> ... | LITHOCHAIN <uid> ... |
//                      SETVALUE/SERVOLOOP/WAVE/OUTWAVE/INWAVE/RAMP/SHOWTIP/SETMODE ...
//                      (fire-and-forget, NO uid) | ABORT | ABORTCONFIRM | CLEARFAULT | ABORTFAULT <uid>
//   bridge -> driver : STARTUP <uid> proto=2;<k=v;..> | VALUE <uid> <n> |
//                      XPT <uid> <entry=value;..> | CAPTURE <uid> <name> <chan> v;.. |
//                      DONE <uid> <name> | SCANDONE <uid> <n> | ABORTED <uid> <name> |
//                      ERR <uid> <msg>
//                      | TELEM <x> <y> <zdrive> <defl> <sum>  (async, always-on, NO uid)
// GETVALUE reads a controller channel (td_ReadValue); GETGV reads an Asylum global
// (GV, e.g. XLVDTSens, which the Cypher rescales on every motor move) and replies
// with the same VALUE frame, "nan" when the read errors.
// Only STARTUP, GETVALUE, GETGV and READXPT reply synchronously; the write/arm verbs are
// fire-and-forget. A FIRE's completion returns later via AFM_Done (the general
// pass path) and ARMFIRE's via AFM_LithoDone (the fixed litho path); both stream
// any CAPTURE frames then the terminal DONE -- all tagged with the pass <uid> the
// FIRE/ARMFIRE carried, held in root:AFM_lrUID / root:AFM_scUID.
//
// Topology: Igor is the CLIENT (AFM_LithoRunStart connects out over SOCKIT, the
// proven mode); Python is the SERVER (LithographyRun.serve accepts the link).
//
// Requires the ARC (td_/ir_/GV) + SOCKIT (AFM_SendStr / SOCKITpeek) loaded.

// Protocol version (docs/IGOR_PYTHON_COMMS.md). Bumped to 2 for UID correlation +
// the versioned handshake: every request that expects a reply carries a UID as the
// first token after the verb, and AFM_Dispatch ECHOES it in the reply. The STARTUP
// reply carries proto=2 so a UID-aware Python driver and a mismatched .ipf fail
// LOUDLY at connect (the driver checks proto), never silently mid-run. RIG-PENDING:
// this half is validated on the instrument (it cannot compile off it).
StrConstant kProto = "2"

StrConstant kBiasInChannel = "UserIn2"  // MEASURED tip bias: Asylum alias for the front-panel
                                         // BNC In2, as the vendor litho arms it (Lithography.ipf
                                         // IR_XSetInWavePair(1,...,"UserIn2",...)). The operator
                                         // routes BNCIn2 to an ADC in the crosspoint by hand.
StrConstant kLateralInChannel = "Lateral" // lateral (torsional) photodetector signal, read by alias;
                                         // paired with kBiasInChannel on litho in-bank 2 (the ARC2
                                         // accepted this pair, rc=0, on the Cypher 2026-10-22).

StrConstant kEvent = "14"                // controller event our wave banks fire on. The
                                         // vendor reserves 0..6 for its own realtime code
                                         // (ARRealTimeProgramming help: "start backwards
                                         // from event.14 for your code"); using 0 contends
                                         // with it and fills the callback table sooner.

Constant kCapChunkPoints = 8192          // AFM_SendCapture splits a captured wave into
                                         // frames of at most this many points, so a large
                                         // end-of-pass capture never leaves as ONE giant
                                         // "CAPTURE ... v;v;..." line (the same failure class
                                         // we avoid sending waves INTO Igor -- see
                                         // WAVE_CHUNK_POINTS=8192 in hardware_arc.py, which
                                         // this mirrors). The host appends same-(name,chan)
                                         // frames in order into one array (setdefault/extend),
                                         // so multiple frames reassemble transparently. A
                                         // capture at/under this many points sends as ONE
                                         // frame -- the small-capture path is unchanged.

// ============================ run lifecycle =================================

// Open the SOCKIT link to the Python driver (Igor connects out as client) and
// start the dispatch listener. After this, Igor answers the driver's protocol.
Function AFM_LithoRunStart(host, port)
	String host
	Variable port

	// Connect into a LOCAL first: a failed dial (e.g. the service isn't listening
	// -- it accepts one link then stops, so a second start_bridge fails) must not
	// clobber root:AFM_lrSock and orphan an already-live link. Only commit the
	// global + (re)start the dispatch listener once the connect actually succeeds.
	Variable s = AFM_SockConnect(host, port)
	if(s < 0)
		return -1
	endif
	Variable/G root:AFM_lrSock = s
	AFM_V3LinkOpened()                                   // wire v3: a new link has no host until its HELLO
	NVAR sock = root:AFM_lrSock
	String/G root:AFM_lrName = ""
	String/G root:AFM_lrUID = ""                         // UID of the in-flight FIRE/ARMFIRE/LITHOCHAIN pass
	String/G root:AFM_scUID = ""                         // UID of the in-flight SCANCHAIN
	String/G root:AFM_lrBuf = ""                         // dispatch line buffer
	String/G root:AFM_capChans = ""                      // general-pass capture: channels
	String/G root:AFM_capWaves = ""                      //   and their capture waves
	Variable/G root:AFM_doneSent = 0                     // guard: AFM_Done runs once
	Variable/G root:AFM_lrDryRun = 0                     // 1 = print commands, run nothing
	Variable/G root:AFM_lrTrace = 0                      // 1 = echo every command to history
	CtrlNamedBackground AFMdispatch, period=3, proc=AFM_DispatchTick
	CtrlNamedBackground AFMdispatch, start
	// ALWAYS-ON telemetry+safety poll (AFMtip): started here at bridge startup, NOT
	// gated on SHOWTIP. It pushes one TELEM frame per tick (tip XY + Height +
	// Deflection + Sum) so the host has live tip position AND the safety channels
	// (rail-dwell / SUM / deflection) the whole time the link is up. SHOWTIP only
	// tells the HOST whether to DISPLAY the marker; it never stops this poll.
	Variable p0 = AFM_TipDesiredPeriod()
	Variable/G root:AFM_tipPeriod = p0
	CtrlNamedBackground AFMtip, period=(p0), proc=AFM_TipTick     // 60/period Hz
	CtrlNamedBackground AFMtip, start
	return sock
End

Function AFM_LithoRunEnd()
	CtrlNamedBackground AFMdispatch, stop
	CtrlNamedBackground AFMtip, stop                     // stop the live tip poll (link closes below)
	AFM_SetValue("Output.A", 0)                          // leave the bias safe
	NVAR/Z sock = root:AFM_lrSock
	if(NVAR_Exists(sock))
		AFM_SockClose(sock)
	endif
	return 0
End

// Diagnostics: watch and/or dry-run the command stream. Call from Igor's command
// line while the bridge is up.
//   AFM_LithoRunTrace(1)   -- echo every received command to the history (STILL runs).
//                             The last line printed before a hang localizes the hang.
//   AFM_LithoRunDryRun(1)  -- echo AND run nothing on the instrument: reads still
//                             answer, FIRE/ABORT report completion so the driver does
//                             not hang, but no setpoint, wave, servo or event is
//                             actually commanded. The way to SEE a whole pass without
//                             moving anything. Pass 0 to either to turn it off.
Function AFM_LithoRunTrace(on)
	Variable on
	Variable/G root:AFM_lrTrace = (on != 0)
	Printf "[bridge] trace = %d\r", (on != 0)
	return 0
End

Function AFM_LithoRunDryRun(on)
	Variable on
	Variable/G root:AFM_lrDryRun = (on != 0)
	if(on != 0)
		Variable/G root:AFM_lrTrace = 1                  // dry-run implies trace
	endif
	Printf "[bridge] dry-run = %d (commands print; nothing is commanded)\r", (on != 0)
	return 0
End

// ======================= dispatch listener (the ear) ========================

// Fires every ~50 ms (period=3 ticks): drain the SOCKIT buffer, then act on
// every WHOLE line the driver has sent. Because the armed litho runs in the
// CONTROLLER (not Igor's main thread), this task keeps servicing the socket
// DURING a write -- so an ABORT is picked up promptly.
Function AFM_DispatchTick(s)
	STRUCT WMBackgroundStruct &s

	NVAR sock = root:AFM_lrSock
	SVAR buf = root:AFM_lrBuf

	String chunk = SOCKITpeek(sock)
	if(strlen(chunk) > 0)
		buf += chunk
		buf = AFM_V3Intake(buf)                          // wire v3: a MAKESAFE runs before the lines queued with it
	endif
	Variable nl
	do
		nl = strsearch(buf, "\n", 0)
		if(nl < 0)
			break                                        // no complete line yet
		endif
		String line = buf[0, nl - 1]
		buf = buf[nl + 1, strlen(buf) - 1]               // keep the remainder
		AFM_DispatchFromSocket(line)
	while(1)
	return 0
End

// Act on ONE protocol line: split the verb, call the matching wrapper, write the
// reply. Mirror of igor_stub.py's _dispatch, verb for verb.
Function AFM_Dispatch(line)
	String line

	NVAR sock = root:AFM_lrSock
	String verb = StringFromList(0, line, " ")
	String reply

	// Visibility (trace/dry-run): echo the received command to the history. WAVE
	// carries the whole numeric wave, so summarize it as a point count; every other
	// verb is short and prints in full.
	NVAR/Z lrTrace = root:AFM_lrTrace
	NVAR/Z lrDry = root:AFM_lrDryRun
	if((NVAR_Exists(lrTrace) && lrTrace) || (NVAR_Exists(lrDry) && lrDry))
		if(cmpstr(verb, "WAVE") == 0 || cmpstr(verb, "WAVEAPP") == 0)
			Printf "[bridge<<] %s %s (%d pts)\r", verb, StringFromList(1, line, " "), ItemsInList(StringFromList(2, line, " "), ";")
		else
			Printf "[bridge<<] %s\r", line
		endif
	endif

	// Dry-run: command NOTHING. Reads still answer (harmless and useful); FIRE and
	// ABORT report completion so the driver does not block waiting for DONE; every
	// write/arm is printed above but not executed.
	if(NVAR_Exists(lrDry) && lrDry)
		String duid = StringFromList(1, line, " ")   // uid = first token after the verb
		strswitch(verb)
			case "STARTUP":
				AFM_SendStr(sock, "STARTUP " + duid + " proto=" + kProto + ";" + AFM_MacroStartup() + "\n")
				break
			case "GETVALUE":
				sprintf reply, "VALUE %s %.12g\n", duid, AFM_GetValue(StringFromList(2, line, " "))
				AFM_SendStr(sock, reply)
				break
			case "GETGV":
				AFM_SendStr(sock, "VALUE " + duid + " " + AFM_GetGVTok(StringFromList(2, line, " ")) + "\n")
				break
			case "READXPT":
				// Reads still answer in dry-run (read-only, harmless and useful).
				AFM_ReadXPT(duid, StringFromList(2, line, " "))
				break
			case "FIRE":
				AFM_SendStr(sock, "DONE " + duid + " " + StringFromList(2, line, " ") + "\n")
				break
			case "ARMFIRE":
				AFM_SendStr(sock, "DONE " + duid + " " + StringFromList(2, line, " ") + "\n")
				break
			case "ABORT":
				AFM_SendStr(sock, "ABORTED  \n")     // uid-less abort; no pass in dry-run
				break
			case "SCANCHAIN":
				AFM_SendStr(sock, "SCANDONE " + duid + " 0\n")   // dry-run: report done, command nothing
				break
			case "LITHOCHAIN":
				AFM_SendStr(sock, "DONE " + duid + " " + StringFromList(2, line, " ") + "\n")  // report done, command nothing
				break
		endswitch
		return 0
	endif

	strswitch(verb)
		case "STARTUP":
			// STARTUP <uid> <version>: echo the uid + our proto for the handshake.
			AFM_SendStr(sock, "STARTUP " + StringFromList(1, line, " ") + " proto=" + kProto + ";" + AFM_MacroStartup() + "\n")
			break
		case "GETVALUE":
			// GETVALUE <uid> <name>
			sprintf reply, "VALUE %s %.12g\n", StringFromList(1, line, " "), AFM_GetValue(StringFromList(2, line, " "))
			AFM_SendStr(sock, reply)
			break
		case "GETGV":
			// GETGV <uid> <name>: read-only; there is deliberately no SETGV.
			AFM_SendStr(sock, "VALUE " + StringFromList(1, line, " ") + " " + AFM_GetGVTok(StringFromList(2, line, " ")) + "\n")
			break
		case "SETVALUE":
			AFM_SetValue(StringFromList(1, line, " "), str2num(StringFromList(2, line, " ")))
			break
		case "SERVOLOOP":
			// SERVOLOOP <loop> <in> <sp> <p> <i> <out> [outMin outMax] -- the last two
			// are OPTIONAL (servoloop-limits fix, 2026-09-16): default -10/150 when
			// absent, so a caller that never learned about output limits (e.g. the
			// driver's internal lift/set-down arms) behaves exactly as before.
			// AFM_SetupServoLoop used to hardcode -10/150 itself, which parked a
			// Cypher withdraw's Height loop at -10 V instead of 0 (Joe, 2026-09-16) --
			// the driver decides the limits now. Parsed explicitly (no Igor optional-
			// param sugar) to stay Igor 6.38-safe.
			Variable slOutMin = -10, slOutMax = 150
			if(ItemsInList(line, " ") >= 9)
				slOutMin = str2num(StringFromList(7, line, " "))
				slOutMax = str2num(StringFromList(8, line, " "))
			endif
			AFM_SetupServoLoop(str2num(StringFromList(1, line, " ")), StringFromList(2, line, " "), str2num(StringFromList(3, line, " ")), str2num(StringFromList(4, line, " ")), str2num(StringFromList(5, line, " ")), StringFromList(6, line, " "), slOutMin, slOutMax)
			break
		case "STOPLOOP":
			AFM_StopLoop(str2num(StringFromList(1, line, " ")))
			break
		case "RAMP":
			AFM_RampOutput(StringFromList(1, line, " "), str2num(StringFromList(2, line, " ")), str2num(StringFromList(3, line, " ")), str2num(StringFromList(4, line, " ")))
			break
		case "WAVE":
			AFM_StoreWave(StringFromList(1, line, " "), StringFromList(2, line, " "))
			break
		case "WAVEAPP":
			AFM_AppendWave(StringFromList(1, line, " "), StringFromList(2, line, " "))
			break
		case "ARMFIRE":
			// ARMFIRE <uid> <name> <interp> <capLen> <decim>
			String/G root:AFM_lrUID = StringFromList(1, line, " ")
			AFM_ArmAndFireLitho(StringFromList(2, line, " "), str2num(StringFromList(3, line, " ")), str2num(StringFromList(4, line, " ")), str2num(StringFromList(5, line, " ")))
			break
		case "OUTWAVE":
			AFM_ArmOutWave(str2num(StringFromList(1, line, " ")), StringFromList(2, line, " "), StringFromList(3, line, " "), StringFromList(4, line, " "), str2num(StringFromList(5, line, " ")))
			break
		case "OUTPAIR":
			AFM_ArmOutPair(str2num(StringFromList(1, line, " ")), StringFromList(2, line, " "), StringFromList(3, line, " "), StringFromList(4, line, " "), StringFromList(5, line, " "), StringFromList(6, line, " "), str2num(StringFromList(7, line, " ")))
			break
		case "INWAVE":
			AFM_ArmInWave(str2num(StringFromList(1, line, " ")), StringFromList(2, line, " "), StringFromList(3, line, " "), str2num(StringFromList(4, line, " ")), str2num(StringFromList(5, line, " ")))
			break
		case "INPAIR":
			AFM_ArmInPair(str2num(StringFromList(1, line, " ")), StringFromList(2, line, " "), StringFromList(3, line, " "), StringFromList(4, line, " "), str2num(StringFromList(5, line, " ")), str2num(StringFromList(6, line, " ")))
			break
		case "FIRE":
			// FIRE <uid> <name> <event>
			String/G root:AFM_lrUID = StringFromList(1, line, " ")
			AFM_FireEvent(StringFromList(2, line, " "), StringFromList(3, line, " "))
			break
		case "SCANCHAIN":
			// SCANCHAIN <uid> <xname> <yname> <ppl> <lines> <interp> <decim> [<errChan>]
			// The optional trailing token is the mode-aware bank-0 ERROR channel
			// (Deflection contact | Amplitude AC). Absent -> Deflection (a pre-field host).
			String/G root:AFM_scUID = StringFromList(1, line, " ")
			String scErr = StringFromList(8, line, " ")
			if(strlen(scErr) == 0)
				scErr = "Deflection"
			endif
			AFM_ScanChainStart(StringFromList(2, line, " "), StringFromList(3, line, " "), str2num(StringFromList(4, line, " ")), str2num(StringFromList(5, line, " ")), str2num(StringFromList(6, line, " ")), str2num(StringFromList(7, line, " ")), errChan=scErr)
			break
		case "LITHOCHAIN":
			// LITHOCHAIN <uid> <name> <xname> <yname> <vname> <boundsName> <interp> <decim>
			String/G root:AFM_lrUID = StringFromList(1, line, " ")
			AFM_LithoChainStart(StringFromList(2, line, " "), StringFromList(3, line, " "), StringFromList(4, line, " "), StringFromList(5, line, " "), StringFromList(6, line, " "), str2num(StringFromList(7, line, " ")), str2num(StringFromList(8, line, " ")))
			break
		case "SHOWTIP":
			AFM_ShowTip(cmpstr(StringFromList(1, line, " "), "ON", 1) == 0)
			break
		case "SETMODE":
			// SETMODE <Contact|AC>: switch the instrument imaging mode via the
			// vendor's own mode-profile call (fire-and-forget, no reply).
			AFM_SetMode(StringFromList(1, line, " "))
			break
		case "READXPT":
			// READXPT <uid> <group>: read-only crosspoint read-back -> XPT <uid> InFast=<v>;
			// The authoritative litho-AC guard reads this. Read-only (td_ReadGroup), so safe
			// mid-scan; replies synchronously like GETVALUE.
			AFM_ReadXPT(StringFromList(1, line, " "), StringFromList(2, line, " "))
			break
		case "ABORT":
			AFM_SafePark()                               // the one abort entry point: td_Stop, bias 0, retract
			break
		case "ABORTCONFIRM":
			AFM_AbortConfirmed()                         // the host read Height < 1 V: the retract is confirmed
			break
		case "ABORTFAULT":
			AFM_SendStr(sock, "VALUE " + StringFromList(1, line, " ") + " " + num2str(AFM_ReadAbortFault()) + "\n")
			break
		case "CLEARFAULT":
			if(ItemsInList(line, " ") > 1)                   // CLEARFAULT <seq> is wire v3: host only, with a REPLY
				AFM_V3Dispatch(verb, line)
			else
				AFM_ClearFault()                             // the operator cleared the fault: both abort flags to 0
			endif
			break
		default:
			AFM_DispatchOther(sock, verb, line)
	endswitch
	return 0
End
// ======================= transport: store a wave ============================
Function AFM_StoreWave(name, body)                  // a WAVE line, values final (LVDT or DAC volts), into root:$name; returns its point count
	String name, body

	// Parse the ";"-delimited body in a SINGLE pass. The old form called
	// StringFromList(i, body, ";") inside the loop, and StringFromList rescans the
	// whole string from the start on every call, so filling an N-point wave was
	// O(N^2): a scan wave of a few thousand points froze Igor for many seconds at
	// arm time (this is what "Igor freezes at the start of a scan" was). Walking the
	// string once with strsearch is O(N), so a 16k-point wave parses in well under a
	// second.
	Variable n = ItemsInList(body, ";"), i = 0, start = 0, pos
	Make/O/N=(n) root:$name
	WAVE w = root:$name
	do
		pos = strsearch(body, ";", start)
		if(pos < 0)
			w[i] = str2num(body[start, strlen(body) - 1])
			break
		endif
		w[i] = str2num(body[start, pos - 1])
		start = pos + 1
		i += 1
	while(i < n)
	return n
End

// APPEND a ";"-delimited body to an existing wave (WAVEAPP). A big frame (e.g. a
// 256x256 raster, 131k points) is sent as WAVE (first chunk) + WAVEAPP (the rest),
// so one Igor wave is built up from bounded transfers -- no multi-MB line that
// chokes the socket. Parses the chunk once (reusing AFM_StoreWave's single-pass
// parse into a temp wave), then extends the target and copies the chunk in.
Function AFM_AppendWave(name, body)
	String name, body
	WAVE/Z w = root:$name
	if(!WaveExists(w))
		return AFM_StoreWave(name, body)      // no wave yet: this IS the first chunk
	endif
	Variable nAdd = AFM_StoreWave("AFM_appTmp", body)
	WAVE tmp = root:AFM_appTmp
	Variable n0 = numpnts(w)
	Redimension/N=(n0 + nAdd) w
	w[n0, n0 + nAdd - 1] = tmp[p - n0]
	return n0 + nAdd
End

// ================ general td_-mirroring surface (any channel set) ============
// The driver can drive ANY channel set (a scan pass, not just litho): load waves
// by name (WAVE), bind them to outputs (OUTWAVE/OUTPAIR), arm input capture
// (INWAVE/INPAIR), then FIRE. On completion AFM_Done streams one CAPTURE frame
// per armed input channel, then DONE. Mirror of igor_stub.py's general handlers.
// Requirement: arm all input banks of one pass with the SAME capture length, so
// they fill on the same event tick (AFM_Done then reads them all safely).

// BIND a loaded wave to an output channel.  wraps td_xSetOutWave.
Function AFM_ArmOutWave(bank, event, channel, name, interp)
	Variable bank, interp
	String event, channel, name
	WAVE w = root:$name
	return td_xSetOutWave(bank, event, channel, w, interp)
End

// BIND a loaded wave PAIR to a bank's output pair.  wraps td_xSetOutWavePair.
Function AFM_ArmOutPair(bank, event, chA, nameA, chB, nameB, interp)
	Variable bank, interp
	String event, chA, nameA, chB, nameB
	WAVE wA = root:$nameA
	WAVE wB = root:$nameB
	return td_xSetOutWavePair(bank, event, chA, wA, chB, wB, interp)
End

// A capture wave name for a channel: channel names have dots ("Output.Z") that
// are illegal in wave names, so sanitize to "cap_Output_Z".
Function/S AFM_CapWaveName(channel)
	String channel
	String w = "cap_" + channel
	w = ReplaceString(".", w, "_")
	w = ReplaceString("%", w, "_")
	return w
End

// Record one captured channel + its wave so AFM_Done can stream it back.
Function AFM_CapRecord(channel, waveName)
	String channel, waveName
	SVAR chans = root:AFM_capChans
	SVAR waves = root:AFM_capWaves
	chans += channel + ";"
	waves += waveName + ";"
	return 0
End

// ARM capture of one input channel into a fresh wave (callback AFM_Done fires
// when the bank fills).  wraps td_xSetInWave.
Function AFM_ArmInWave(bank, event, channel, capLen, decim)
	Variable bank, capLen, decim
	String event, channel
	String wn = AFM_CapWaveName(channel)
	Make/O/N=(capLen) root:$wn
	WAVE w = root:$wn
	AFM_CapRecord(channel, wn)
	return td_xSetInWave(bank, event, channel, w, "AFM_Done()", decim)
End

// ARM capture of an input PAIR.  wraps td_xSetInWavePair.
Function AFM_ArmInPair(bank, event, chA, chB, capLen, decim)
	Variable bank, capLen, decim
	String event, chA, chB
	String wnA = AFM_CapWaveName(chA), wnB = AFM_CapWaveName(chB)
	Make/O/N=(capLen) root:$wnA, root:$wnB
	WAVE wA = root:$wnA
	WAVE wB = root:$wnB
	AFM_CapRecord(chA, wnA)
	AFM_CapRecord(chB, wnB)
	return td_xSetInWavePair(bank, event, chA, wA, chB, wB, "AFM_Done()", decim)
End

// FIRE the event: set the report name, reset the done guard, trigger. Returns at
// once; completion normally arrives via AFM_Done (the in-bank capture callback).
Function AFM_FireEvent(name, event)
	String name, event
	String/G root:AFM_lrName = name
	Variable/G root:AFM_doneSent = 0
	td_WriteString("Event." + event, "Once")            // FIRE
	Variable/G root:AFM_passArmed = 1                   // a pass may play until the next td_Stop
	// No input bank armed means no capture callback, so AFM_Done would never run and the
	// driver would wait on DONE until it times out: send the terminal here (AFM_Done streams
	// zero CAPTURE frames for an empty record, just "DONE <name>"). Every real pass captures,
	// so this is a safety net for a bare FIRE, not the normal path.
	SVAR chans = root:AFM_capChans
	if(strlen(chans) == 0)
		AFM_Done()
	endif
	return 0
End

// GENERAL COMPLETION. Each armed in-bank's callback calls this when it fills; the
// guard makes only the FIRST do the work (all banks fill together on one event).
// Zeroes the bias (safety), streams a CAPTURE frame per armed channel, sends
// DONE, then clears the capture record for the next pass.
Function AFM_Done()
	NVAR sent = root:AFM_doneSent
	if(sent)
		return 0
	endif
	sent = 1
	SVAR name = root:AFM_lrName
	SVAR uid = root:AFM_lrUID
	NVAR sock = root:AFM_lrSock
	SVAR chans = root:AFM_capChans
	SVAR waves = root:AFM_capWaves
	AFM_SetValue("Output.A", 0)
	Variable n = ItemsInList(chans, ";"), i
	for(i = 0; i < n; i += 1)
		AFM_SendCapture(sock, uid, name, StringFromList(i, chans, ";"), "root:" + StringFromList(i, waves, ";"))
	endfor
	AFM_SendRunFrame(sock, "DONE " + uid + " " + name + "\n")
	chans = ""                                           // clear for the next pass
	waves = ""
	return 0
End

// ======================= primitives: set / get / loop =======================

// SET A CONTROLLER VALUE (a setpoint, a bias output, ...).  wraps td_WriteValue.
Function AFM_SetValue(param, value)
	String param
	Variable value
	return td_WriteValue(param, value)
End

// READ A CONTROLLER VALUE.  wraps td_ReadValue.
Function AFM_GetValue(param)
	String param
	return td_ReadValue(param)
End

// ONE ASYLUM GLOBAL as a reply token (AFM_TelemTok format): %.9g, or "nan" when GV
// raised a runtime error or returned NaN/Inf. GetRTError(1) clears the error so a
// bad name never aborts the dispatch loop.
Function/S AFM_GetGVTok(name)
	String name
	Variable v = GV(name)
	if(GetRTError(1) != 0)
		return "nan"
	endif
	return AFM_TelemTok(v)
End

// SET UP ONE SERVO LOOP (input -> output with a setpoint and gains). wraps
// ir_SetPISLoop with the litho-standard event; the driver passes the loop
// index, channels, setpoint, gains, and the OUTPUT LIMITS (outMin/outMax --
// servoloop-limits fix, 2026-09-16: this used to hardcode -10/150, which is
// wrong for a Cypher withdraw's re-arm, whose caller wants 0/150 so the Height
// servo cannot be driven negative; see docs/HARDWARE_BRINGUP.md / hw_approach.py).
Function AFM_SetupServoLoop(loop, inChan, setpoint, pgain, igain, outChan, outMin, outMax)
	Variable loop, setpoint, pgain, igain, outMin, outMax
	String inChan, outChan
	Variable err = ir_SetPISLoop(loop, "Always,Never", inChan, setpoint, pgain, igain, 0, outChan, outMin, outMax)
	AFM_V3NoteLoop(loop, setpoint, outMin, outMax)       // wire v3: tip-down tracking (contract 5b); commands nothing
	return err
End

// STOP ONE PIDS LOOP. Freezes the loop's output where it is (does not move it);
// a following RAMP is what parks the channel. Used by withdraw (stop the Z
// feedback loop, then ramp Output.Z to 0).
Function AFM_StopLoop(loop)
	Variable loop
	return ir_StopPISLoop(loop)
End

// RATE-LIMITED RAMP of one output channel to `dest`, at most `rate` per second, within `maxsec`
// seconds (wraps IR_SetRamp). Fire-and-forget; the controller runs the ramp on its own.
Function AFM_RampOutput(chan, dest, rate, maxsec)
	String chan
	Variable dest, rate, maxsec
	Variable/G root:AFM_passArmed = 1                   // a ramp moves an output (XY too) until td_Stop
	IR_SetRamp(maxsec, chan, rate, dest, "", 0, 0, "", 0, 0, "")
	return 0
End

// ======================= arm / fire the litho sequence ======================

// ARM AND FIRE THE LITHOGRAPHY SEQUENCE. The driver has already put the final
// waves in Igor (XLitho, YLitho = LVDT setpoint volts; VLitho = DAC volts) and
// computed `interp` (signed: negative = linear, |interp| ticks/point) and the
// capture length. This binds the XY drive and the bias drive to Event.0, arms a
// synchronized IN-wave capture through the input banks -- the whitepaper's "parallel
// capture" -- sets the capture-fill CALLBACK to our report, and fires. Returns
// immediately; the controller runs the whole trajectory and records on its own.
//
// UNIFORM PER-PASS BANK CONVENTION (litho single + litho chain + scan all share it):
//   bank 0  OUT = XY setpoints          IN = ZSensor + Deflection  (carries the callback)
//   bank 1                              IN = XSensor + YSensor      (position / tip trace)
//   bank 2  OUT = Output.A bias         IN = UserIn2 measured bias + Lateral (litho; fail-safe)
// In-wave and out-wave bank pools are INDEPENDENT (separate InWaves/OutWaves tables),
// so bank 0 and bank 2 each carry an out-wave AND an in-wave. The COMPLETION callback
// rides ONLY bank 0 -- a second fill callback would double-complete. All in-banks share
// one event + capture length, so they fill together and bank 0's callback can read them
// all. (Banks are 0-2 only; there is no bank 3.)
Function AFM_ArmAndFireLitho(name, interp, captureLen, decimation)
	String name
	Variable interp, captureLen, decimation

	Make/O/N=(captureLen) root:ZcapLitho, root:DcapLitho, root:XcapLitho, root:YcapLitho, root:VcapLitho, root:LcapLitho
	// Release any banks a prior pass left locked (banks 0-2 always exist, so these are
	// safe no-ops when unarmed) so a scan-then-litho sequence can re-arm the shared
	// in-banks. Waves auto-unlock after completion, but the vendor never relies on that.
	ir_StopOutWaveBank(0)
	ir_StopOutWaveBank(2)
	ir_StopInWaveBank(0)
	ir_StopInWaveBank(1)
	ir_StopInWaveBank(2)
	td_xSetOutWavePair(0, kEvent, "PIDSLoop.0.Setpoint", root:XLitho, "PIDSLoop.1.Setpoint", root:YLitho, interp)
	Printf "[arm] out0 XY ok\r"
	td_xSetOutWave(2, kEvent, "Output.A", root:VLitho, interp)
	Printf "[arm] out2 Output.A drive ok\r"
	td_xSetInWavePair(0, kEvent, "ZSensor", root:ZcapLitho, "Deflection", root:DcapLitho, "AFM_LithoDone()", decimation)
	Printf "[arm] in0 ZSensor/Deflection ok\r"
	td_xSetInWavePair(1, kEvent, "XSensor", root:XcapLitho, "YSensor", root:YcapLitho, "", decimation)
	Printf "[arm] in1 XSensor/YSensor ok\r"
	Variable/G root:AFM_passArmed = 1                   // out-waves armed: a pass may play until td_Stop
	// MEASURED tip-bias readback on bank 2 IN, the SAME bank that DRIVES the bias, same event
	// + capture length, NO fill callback (bank 0 alone drives AFM_LithoDone; a 2nd callback =
	// double completion). UserIn2 is the front-panel BNC In2 the operator tees the bias line into
	// and routes to an ADC by hand (before the x25 amp, so DAC volts); it does not replace the
	// COMMANDED VLitho (v_written), which is known. Lateral rides the same bank as a pair.
	// FAIL-SAFE ladder: pair refused -> clear the bank, log once, arm UserIn2 alone; that
	// refused too -> clear, log, fire ANYWAY. AFM_LithoDone streams only what armed.
	Variable lErr = td_xSetInWavePair(2, kEvent, kBiasInChannel, root:VcapLitho, kLateralInChannel, root:LcapLitho, "", decimation)
	Printf "[arm] in2 %s/%s measured bias + lateral (rc=%d)\r", kBiasInChannel, kLateralInChannel, lErr
	Variable vErr = lErr
	if(lErr != 0)
		ir_StopInWaveBank(2)                             // drop any partial binding
		printf "AFM: %s/%s pair not armed (td_xSetInWavePair err %d); arming %s alone\r", kBiasInChannel, kLateralInChannel, lErr, kBiasInChannel
		vErr = td_xSetInWave(2, kEvent, kBiasInChannel, root:VcapLitho, "", decimation)
		Printf "[arm] in2 %s measured bias (rc=%d)\r", kBiasInChannel, vErr
		if(vErr != 0)
			ir_StopInWaveBank(2)
			printf "AFM: %s tip-bias capture not armed (td_xSetInWave err %d); write proceeds without it\r", kBiasInChannel, vErr
		endif
	endif
	Variable/G root:AFM_VcapArmed = (vErr == 0)
	Variable/G root:AFM_LcapArmed = (lErr == 0)
	String/G root:AFM_lrName = name
	td_WriteString("Event." + kEvent, "Once")            // FIRE
	Printf "[arm] FIRED Event.%s\r", kEvent
	return 0
End

// SEND ONE CAPTURED CHANNEL back to the driver as one or more CAPTURE frames:
//   "CAPTURE <uid> <name> <channel> v;v;...;v\n"   (matches bridge_protocol.msg_capture)
// <uid> is the in-flight pass's UID (root:AFM_lrUID / root:AFM_scUID), so the Python
// router routes every chunk to that pass's waiter.
// CHUNKED: a large capture (a full-length litho write, up to the 87,000-pt cap, or a
// big scan frame) is split into frames of at most kCapChunkPoints points and each is
// sent as its OWN CAPTURE frame under the SAME <name> <channel>. The host appends
// same-(name,channel) frames in ARRIVAL ORDER into one array (setdefault(chan,[]).extend
// in lithography_run / hardware_arc; a length-gated accumulate in the chained-scan
// driver), so N frames reassemble into the one capture -- transport-safe, no single
// multi-MB line. A capture at/under kCapChunkPoints sends as exactly ONE frame, so the
// small-capture path (and every per-line scan capture, ppl << the chunk) is unchanged.
// TCP preserves order, and Igor background tasks are cooperative (no interleave), so the
// frames arrive contiguous and in order.
Function AFM_SendCapture(sock, uid, name, channel, wavePath)
	Variable sock
	String uid, name, channel, wavePath

	WAVE/Z w = $wavePath
	if(!WaveExists(w))
		return -1
	endif
	Variable n = numpnts(w), i, a = 0, b
	String s, tmp
	do
		b = a + kCapChunkPoints                          // this frame's exclusive end
		if(b > n)
			b = n
		endif
		s = "CAPTURE " + uid + " " + name + " " + channel + " "
		for(i = a; i < b; i += 1)
			sprintf tmp, "%.9g;", w[i]
			s += tmp
		endfor
		s += "\n"
		AFM_SendRunFrame(sock, s)                        // one bounded frame
		a = b
	while(a < n)                                         // n==0 sends exactly one empty frame
	return 0
End

// REPORT COMPLETION. The XOP fires this when the capture bank fills -- i.e. when
// the write finishes. It zeroes the bias immediately (safety, so the tip is not
// left live), streams the measured capture back (one CAPTURE frame per channel),
// then reports DONE; the driver collects the capture and orchestrates the lift.
Function AFM_LithoDone()
	SVAR name = root:AFM_lrName
	SVAR uid = root:AFM_lrUID
	NVAR sock = root:AFM_lrSock
	AFM_SetValue("Output.A", 0)
	AFM_SendCapture(sock, uid, name, "ZSensor", "root:ZcapLitho")        // bank 0: Z engagement
	AFM_SendCapture(sock, uid, name, "Deflection", "root:DcapLitho")     // bank 0: error
	AFM_SendCapture(sock, uid, name, "XSensor", "root:XcapLitho")        // bank 1: position
	AFM_SendCapture(sock, uid, name, "YSensor", "root:YcapLitho")
	NVAR/Z vArmed = root:AFM_VcapArmed
	if(NVAR_Exists(vArmed) && vArmed)                    // bank 2, only if the readback armed
		AFM_SendCapture(sock, uid, name, kBiasInChannel, "root:VcapLitho")    // MEASURED bias
	endif
	NVAR/Z lArmed = root:AFM_LcapArmed
	if(NVAR_Exists(lArmed) && lArmed)                    // bank 2 pair partner, only if the pair armed
		AFM_SendCapture(sock, uid, name, kLateralInChannel, "root:LcapLitho")
	endif
	AFM_SendRunFrame(sock, "DONE " + uid + " " + name + "\n")
	return 0
End

// ================================ abort =====================================

// ABORT, step 1 of AFM_SafePark (its only caller): bar the tick chains, td_Stop unless a repeat
// abort finds nothing armed since, zero the bias, send the in-flight pass its one terminal
// (ABORTED or SCANDONE -1) so the host's waiters unwind. AFMtip keeps polling (read-only).
Function AFM_Abort()
	Variable/G root:AFM_scAbort = 1                      // set FIRST: bar AFM_ScanTick from arming another line
	Variable/G root:AFM_scPending = 0                    //   before td_Stop halts the controller
	Variable/G root:AFM_ltAbort = 1                      // same for the litho chain: bar AFM_LithoTick
	Variable/G root:AFM_ltPending = 0                    //   from arming another chunk before td_Stop
	Variable rte                                         // GetRTError(1) after each send below
	// td_Stop undefines the loops: a repeat abort with nothing armed since must not kill the retract.
	NVAR/Z stopped = root:AFM_abortStopped
	if(!NVAR_Exists(stopped) || !stopped || AFM_PassMayPlay())
		td_Stop()
		AFM_StopRan()                                    // flag set, pass marker cleared
	endif
	AFM_SetValue("Output.A", 0)
	NVAR/Z sock = root:AFM_lrSock
	if(NVAR_Exists(sock))
		NVAR/Z scActive = root:AFM_scActive
		if(NVAR_Exists(scActive) && scActive)           // a chained scan is live: end it now, since
			scActive = 0                                //   td_Stop keeps its callback from firing
			String su = AFM_GlobalStr("root:AFM_scUID", "")     // the scan uid (SelectString would read a null SVAR)
			AFM_SendRunFrame(sock, "SCANDONE " + su + " -1\n")
			rte = GetRTError(1)                              // a dead link must not stop the retract
		else
			// No scan chain: a litho chain or single fire is the in-flight pass. Clear
			// the litho-chain active guard so AFM_LithoAdvance cannot ALSO send a DONE
			// -- the ABORTED here is its one terminal -- then send ABORTED tagged with
			// the pass uid (root:AFM_lrUID) so the router routes it to that waiter.
			NVAR/Z ltActive = root:AFM_ltActive
			if(NVAR_Exists(ltActive) && ltActive)
				ltActive = 0
			endif
			SVAR/Z name = root:AFM_lrName                    // the pass uid comes from AFM_GlobalStr ("" if missing)
			if(SVAR_Exists(name))
				String lu = AFM_GlobalStr("root:AFM_lrUID", "")
				AFM_SendRunFrame(sock, "ABORTED " + lu + " " + name + "\n")
				rte = GetRTError(1)                          // a dead link must not stop the retract
			endif
		endif
	endif
	return 0
End

// ABORT FLAGS (Joe's abort design, 2026-10-27; docs/migration/plan/abort-redesign.md)
//   root:AFM_abortStopped  1 from the abort's td_Stop until the retract is confirmed
//                          (Height < 1 V) or the operator clears the fault. While it is 1,
//                          a repeat abort with nothing armed since skips td_Stop, so it
//                          cannot undefine the retract loop an earlier call armed.
//   root:AFM_abortFault    1 when Igor's 5 s backup found Height still >= 1 V and ran the
//                          vendor Withdraw. The host reads it (ABORTFAULT) and latches
//                          its own fault; only the operator's clear (CLEARFAULT) resets it.
// The host confirms first: it polls Height for up to 2.5 s, then sends ABORTCONFIRM. The
// backup (AFM_AbortBackup, end of file) covers a host that is gone. Height is what the
// meter's "Z Voltage" shows (Meter.ipf:1451-1453).
Constant kAbortBackupTicks = 300                     // 5 s at 60 ticks/s
Constant kRetractedHeightV = 1                       // Height under 1 V counts as retracted

// ABORTCONFIRM: the host read Height < 1 V, so the next abort calls td_Stop again.
Function AFM_AbortConfirmed()
	Variable/G root:AFM_abortStopped = 0
	CtrlNamedBackground AFMabortBackup, stop
	Variable/G root:AFM_v3StopSeen = 0                   // wire v3: the next make-safe calls td_Stop
	return 0
End

// ABORTFAULT: Igor's fault flag, read only, so a lost reply loses nothing.
Function AFM_ReadAbortFault()
	NVAR/Z fault = root:AFM_abortFault
	return NVAR_Exists(fault) && fault
End

// CLEARFAULT: the operator cleared the fault, so both flags go back to 0 (Joe, 2026-10-27).
Function AFM_ClearFault()
	Variable/G root:AFM_abortStopped = 0
	Variable/G root:AFM_abortFault = 0
	CtrlNamedBackground AFMabortBackup, stop
	Variable/G root:AFM_v3StopSeen = 0                   // wire v3: the next make-safe calls td_Stop
	return 0
End

// SAFE PARK: the ONE abort entry point, called by the COM abort
// (bridge_router.send_safe_park -> Execute2("AFM_SafePark()")) and the socket ABORT, its
// fallback. Routine withdraws (transit, end of run, Withdraw) are the bare loop-2 re-arm.
//   1. AFM_Abort: bar the tick chains, td_Stop (not on a repeat with nothing armed), bias 0, terminal.
//   2. Re-arm loop 2 at -10 V Deflection, OutputMin 0, on every call: off the surface the
//      loop drives Height to 0 V and the tip retracts. "Always" runs it at once.
//   3. Arm Igor's 5 s backup (AFM_ArmAbortBackup, end of file).
// Rig facts (Cypher, 2026-10-27, INSTRUMENT_CONFIG.md section 6): only this re-arm
// retracts; a Height ramp does not move Z; td_Stop() undefines the loops. Mirrors the
// vendor's Cypher withdraw (Lithography.ipf:3873). RIG-PENDING: RIG_CHECKLIST_2026-10-28.
// On the MFP-3D Height is Output.Z; that loop 2 is its Z loop is still to confirm.
Function AFM_SafePark()
	AFM_Abort()                                          // td_Stop unless a bare repeat, bias 0, terminal
	// Re-arm loop 2 at the retract setpoint; OutputMin 0 parks Height at 0 V.
	ir_SetPISLoop(2, "Always,Never", "Deflection", -10, 0, 1000, 0, "Height", 0, 150)
	AFM_ArmAbortBackup()                                 // Igor confirms on its own in 5 s
	AFM_V3ParkEnds()                                     // wire v3: REPORT madesafe; commands nothing
	return 0
End

// ===================== callback-chained scan (continuous per-line) ===========
// Drive a whole raster as a chain of per-line out-wave fires, advanced by the in-wave
// completion CALLBACK instead of a host round-trip: when a line's capture bank fills,
// AFM_ScanAdvance streams that line back and arms + fires the NEXT line, all inside the
// controller/Igor. This closes most of the per-line gap the Python-driven per-FIRE path
// has (no socket round-trip between lines). The driver sends the FULL X and Y setpoint
// waves once (WAVE), then one:
//   SCANCHAIN <xname> <yname> <points_per_line> <lines> <interp> <decimation> [<errChan>]
// Per line the driver receives CAPTURE frames named "L<line>" (ZSensor + the mode-aware
// error channel: Deflection contact | Amplitude AC), then a terminal "SCANDONE
// <lines_done>". Abort via ABORT (sets AFM_scAbort).
Function AFM_ScanChainStart(xname, yname, ppl, nlines, interp, decim, [errChan])
	String xname, yname
	Variable ppl, nlines, interp, decim
	String errChan
	if(ParamIsDefault(errChan) || strlen(errChan) == 0)
		errChan = "Deflection"                           // contact default
	endif
	NVAR sock = root:AFM_lrSock
	WAVE/Z xw = root:$xname
	WAVE/Z yw = root:$yname
	if(!WaveExists(xw) || !WaveExists(yw))
		SVAR/Z scuid = root:AFM_scUID
		AFM_SendRunFrame(sock, "ERR " + AFM_GlobalStr("root:AFM_scUID", "-") + " SCANCHAIN missing wave\n")
		return -1
	endif
	// mode-aware ERROR capture channel: the SECOND channel of the bank-0 in-wave pair
	// (Deflection contact | Amplitude AC). ZSensor stays the first channel + callback.
	String/G root:AFM_scErrChan = errChan
	String/G root:AFM_scXName = xname
	String/G root:AFM_scYName = yname
	Variable/G root:AFM_scPPL = ppl
	Variable/G root:AFM_scLines = nlines
	Variable/G root:AFM_scInterp = interp
	Variable/G root:AFM_scDecim = decim
	Variable/G root:AFM_scLine = 0
	Variable/G root:AFM_scAbort = 0
	Variable/G root:AFM_scActive = 1                     // guards a single terminal SCANDONE
	Variable/G root:AFM_scPending = 0                    // 1 = a line finished; arm the next (deferred)
	// Advance the chain from a background task on the MAIN thread, NOT from inside the
	// capture-fill callback. Re-arming+re-firing an event from within its own controller
	// callback fills the ARC's bounded callback queue (~10 lines) and the chain silently
	// stalls (confirmed on hardware: 10 clean lines, then silence). The callback only sets
	// AFM_scPending; AFM_ScanTick arms the next line here, after the callback has unwound.
	// Deterministic ~1-tick seam, no stall.
	CtrlNamedBackground AFMscan, period=1, proc=AFM_ScanTick
	CtrlNamedBackground AFMscan, start
	AFM_ScanArmLine(0)
	return 0
End

// Arm and fire ONE line: slice the full waves, bind XY out + ZSensor/Deflection in with
// AFM_ScanAdvance as the capture-fill callback, fire.
Function AFM_ScanArmLine(ln)
	Variable ln
	NVAR ppl = root:AFM_scPPL
	NVAR interp = root:AFM_scInterp
	NVAR decim = root:AFM_scDecim
	SVAR xn = root:AFM_scXName
	SVAR yn = root:AFM_scYName
	WAVE xw = root:$xn
	WAVE yw = root:$yn
	Variable a = ln * ppl
	Make/O/N=(ppl) root:AFM_scLX, root:AFM_scLY, root:AFM_scCZ, root:AFM_scCD, root:AFM_scCV
	WAVE lx = root:AFM_scLX
	WAVE ly = root:AFM_scLY
	WAVE cz = root:AFM_scCZ
	WAVE cd = root:AFM_scCD
	WAVE cv = root:AFM_scCV
	lx = xw[a + p]
	ly = yw[a + p]
	// Release the prior line's bank bindings + in-wave callback registration BEFORE
	// re-arming. Without this the in-wave callback table fills after ~32 lines and
	// td_xSetInWavePair returns error 100 -- the chained-scan stall (confirmed on
	// hardware, then fixed, with the afm_cbtest bench). Vendor code clears banks this
	// way throughout (Lithography/HardWare/ForceMap: ir_StopInWaveBank).
	ir_StopOutWaveBank(0)
	ir_StopInWaveBank(0)
	ir_StopInWaveBank(1)                                 // release the tip-bias bank too
	td_xSetOutWavePair(0, kEvent, "PIDSLoop.0.Setpoint", lx, "PIDSLoop.1.Setpoint", ly, interp)
	// bank 0 IN pair: ZSensor (topography, carries the AFM_ScanAdvance callback) +
	// the mode-aware ERROR channel (Deflection contact | Amplitude AC). Only the second
	// channel NAME varies by mode -- still ONE pair on bank 0, no extra bank.
	SVAR/Z scErr = root:AFM_scErrChan
	String errChan = SelectString(SVAR_Exists(scErr), "Deflection", scErr)
	td_xSetInWavePair(0, kEvent, "ZSensor", cz, errChan, cd, "AFM_ScanAdvance()", decim)
	Variable/G root:AFM_passArmed = 1                   // out-wave armed: a pass may play until td_Stop
	// MEASURED tip bias on a SECOND in-wave bank (bank 1), same event + capture length, NO
	// advance callback. CRITICAL: only bank 0 drives the chain -- a 2nd advance callback per
	// line would double-advance and overflow the ARC's ~10-deep callback queue (the stall this
	// chain was built to avoid). Both banks fill on one tick, so AFM_ScanAdvance reads AFM_scCV.
	// FAIL-SAFE: if UserIn2 does not arm, we clear the bank, log once, and the
	// line still scans + advances without the bias capture.
	Variable vErr = td_xSetInWave(1, kEvent, kBiasInChannel, cv, "", decim)
	Variable/G root:AFM_scVArmed = (vErr == 0)
	if(vErr != 0)
		ir_StopInWaveBank(1)
		if(ln == 0)
			printf "AFM: %s tip-bias capture not armed on scan (td_xSetInWave err %d); scan proceeds without it\r", kBiasInChannel, vErr
		endif
	endif
	td_WriteString("Event." + kEvent, "Once")
	return 0
End

// Capture-fill callback: the line that just fired has filled its in-bank. Stream it,
// then either finish or REQUEST the next line. This runs inside the controller callback,
// so it must NOT arm/fire the next line here (see AFM_ScanChainStart): it only sets
// AFM_scPending and returns, letting AFM_ScanTick do the re-arm on the main thread.
Function AFM_ScanAdvance()
	NVAR sock = root:AFM_lrSock
	SVAR uid = root:AFM_scUID
	NVAR ln = root:AFM_scLine
	NVAR nlines = root:AFM_scLines
	NVAR abort = root:AFM_scAbort
	// Stream the MEASURED tip bias (if its bank armed) BEFORE the ZSensor/Deflection pair:
	// the Python demux completes a line on that pair, so the bias must already be buffered
	// when it does. Fail-safe: skipped entirely when the bank didn't arm.
	NVAR/Z vArmed = root:AFM_scVArmed
	if(NVAR_Exists(vArmed) && vArmed)
		AFM_SendCapture(sock, uid, "L" + num2str(ln), kBiasInChannel, "root:AFM_scCV")
	endif
	AFM_SendCapture(sock, uid, "L" + num2str(ln), "ZSensor", "root:AFM_scCZ")
	// error capture under the mode-aware channel name (Deflection contact | Amplitude
	// AC) -- the same name armed on the bank-0 pair, so the host demux maps it to error.
	SVAR/Z scErr = root:AFM_scErrChan
	String errChan = SelectString(SVAR_Exists(scErr), "Deflection", scErr)
	AFM_SendCapture(sock, uid, "L" + num2str(ln), errChan, "root:AFM_scCD")
	ln += 1
	NVAR/Z scActive = root:AFM_scActive
	if(abort || ln >= nlines)
		if(NVAR_Exists(scActive) && scActive)           // send the terminal only once
			scActive = 0
			AFM_SendRunFrame(sock, "SCANDONE " + uid + " " + num2str(ln) + "\n")
		endif
		Variable/G root:AFM_scPending = 0               // nothing left to arm
	else
		Variable/G root:AFM_scPending = 1               // defer the re-arm to AFM_ScanTick
	endif
	return 0
End

// Chain-advance background task (main thread): when a line has finished (AFM_scPending),
// arm the next one here -- after the callback that set the flag has fully unwound, so the
// controller's callback queue drains every line and never hits its ceiling. Bails on
// abort/finish (AFM_scActive cleared) and stops itself.
Function AFM_ScanTick(s)
	STRUCT WMBackgroundStruct &s
	NVAR/Z pending = root:AFM_scPending
	NVAR/Z active = root:AFM_scActive
	NVAR/Z abort = root:AFM_scAbort
	NVAR/Z ln = root:AFM_scLine
	if(!NVAR_Exists(pending) || !NVAR_Exists(active) || !NVAR_Exists(ln))
		CtrlNamedBackground AFMscan, stop
		return 0
	endif
	if(!active || (NVAR_Exists(abort) && abort))        // finished or aborted: stop advancing
		pending = 0
		CtrlNamedBackground AFMscan, stop
		return 0
	endif
	if(!pending)
		return 0                                        // a line is still in flight
	endif
	pending = 0
	AFM_ScanArmLine(ln)                                 // ln already points at the next line
	return 0
End

// ============ chunked litho write (seamless chained continuation) ============
// One OVERSIZE object (compiled toolpath > the 87,000-point out-wave cap -- the v19
// vendor cMaxOutWave; the v16 reference's 87,380 is too high for the rig) is fired
// as a chain of chunks -- the SAME callback-advanced mechanism as the scan chain,
// with ONE difference that must be exact: SEAM CONTINUITY. The bias (Output.A) is
// zeroed ONLY at the object's TRUE end, NEVER at a chunk seam; consecutive chunks
// slice XLitho/YLitho/VLitho at the SAME index (no lift, no gap), so position and
// voltage stay registered across the seam. The driver splits at pen-ups where it
// can (zero dose artifact); only a single continuous stroke > cap cuts mid-stroke.
//
// The driver sends the full XLitho/YLitho/VLitho once (WAVE), plus an integer bounds
// wave holding the chunk boundary indices [0, c1, ..., n], then one:
//   LITHOCHAIN <name> <xname> <yname> <vname> <boundsName> <interp> <decim>
// Per chunk the driver receives CAPTURE frames (UserIn2 measured bias, XSensor,
// YSensor) ALL under <name> -- it ACCUMULATES same-channel frames into one array per
// channel for the whole object -- then a single terminal DONE <name>. Abort via
// ABORT (AFM_Abort sets AFM_ltAbort, barring the tick from arming the next chunk).

// Capture buffers pad UP to a multiple of 32 points (mirror of compiler.capture_points
// / Lithography.ipf's inputPoints + (32 - mod(inputPoints,32))). Kept in Igor so the
// driver need only send the boundary indices; each chunk's banks share this length.
Function AFM_LithoCapPoints(n)
	Variable n
	if(n < 1)
		n = 1
	endif
	return n + mod(32 - mod(n, 32), 32)
End

// Seed the chain state, start the deferred-re-arm background task, arm chunk 0.
Function AFM_LithoChainStart(name, xname, yname, vname, boundsName, interp, decim)
	String name, xname, yname, vname, boundsName
	Variable interp, decim
	NVAR sock = root:AFM_lrSock
	WAVE/Z xw = root:$xname
	WAVE/Z yw = root:$yname
	WAVE/Z vw = root:$vname
	WAVE/Z bw = root:$boundsName
	if(!WaveExists(xw) || !WaveExists(yw) || !WaveExists(vw) || !WaveExists(bw))
		SVAR/Z lruid = root:AFM_lrUID
		AFM_SendRunFrame(sock, "ERR " + AFM_GlobalStr("root:AFM_lrUID", "-") + " LITHOCHAIN missing wave\n")
		return -1
	endif
	String/G root:AFM_ltName = name
	String/G root:AFM_lrName = name                      // so AFM_Abort's ABORTED carries the object name
	String/G root:AFM_ltXName = xname
	String/G root:AFM_ltYName = yname
	String/G root:AFM_ltVName = vname
	String/G root:AFM_ltBName = boundsName
	Variable/G root:AFM_ltInterp = interp
	Variable/G root:AFM_ltDecim = decim
	Variable/G root:AFM_ltNChunks = numpnts(bw) - 1      // boundaries [0,..,n] -> nChunks = pts-1
	Variable/G root:AFM_ltChunk = 0
	Variable/G root:AFM_ltAbort = 0
	Variable/G root:AFM_ltActive = 1                     // guards a single terminal DONE/ABORTED
	Variable/G root:AFM_ltPending = 0                    // 1 = a chunk finished; arm the next (deferred)
	// Advance the chain from a MAIN-THREAD background task, NOT from inside the
	// capture-fill callback -- re-arming an event from within its own controller
	// callback fills the ARC's ~10-deep callback queue and the chain stalls (the
	// exact reason the scan chain defers; confirmed on hardware). The callback only
	// sets AFM_ltPending; AFM_LithoTick arms the next chunk here after it unwinds.
	CtrlNamedBackground AFMlitho, period=1, proc=AFM_LithoTick
	CtrlNamedBackground AFMlitho, start
	AFM_LithoArmChunk(0)
	return 0
End

// Arm and fire ONE chunk: slice the full waves at [bounds[k]:bounds[k+1]] (SAME index
// for x/y/v), bind the banks, fire. IDENTICAL bank set to AFM_ArmAndFireLitho, just
// sliced -- the uniform convention: XY setpoints OUT bank 0; Output.A bias OUT bank 2;
// ZSensor/Deflection IN bank 0 with AFM_LithoAdvance as the fill callback; XSensor/YSensor
// IN bank 1 (no callback); the fail-safe UserIn2 measured bias IN bank 2 (no callback). Banks
// are RELEASED before re-arm, or td_xSetInWavePair returns err 100 after ~32 chunks (same
// as the scan chain). The advance callback rides ONLY bank 0.
Function AFM_LithoArmChunk(k)
	Variable k
	SVAR xn = root:AFM_ltXName
	SVAR yn = root:AFM_ltYName
	SVAR vn = root:AFM_ltVName
	SVAR bn = root:AFM_ltBName
	NVAR interp = root:AFM_ltInterp
	NVAR decim = root:AFM_ltDecim
	WAVE xw = root:$xn
	WAVE yw = root:$yn
	WAVE vw = root:$vn
	WAVE bw = root:$bn
	Variable a = bw[k]
	Variable b = bw[k + 1]
	Variable m = b - a                                   // this chunk's point count
	Variable capLen = AFM_LithoCapPoints(m)
	Make/O/N=(m) root:AFM_ltLX, root:AFM_ltLY, root:AFM_ltLV
	WAVE lx = root:AFM_ltLX
	WAVE ly = root:AFM_ltLY
	WAVE lv = root:AFM_ltLV
	lx = xw[a + p]                                       // consecutive indices, no lift, no gap
	ly = yw[a + p]
	lv = vw[a + p]                                       // XY and bias sliced at the SAME index
	Make/O/N=(capLen) root:AFM_ltZcap, root:AFM_ltDcap, root:AFM_ltXcap, root:AFM_ltYcap, root:AFM_ltVcap, root:AFM_ltLcap
	WAVE zc = root:AFM_ltZcap
	WAVE dc = root:AFM_ltDcap
	WAVE xc = root:AFM_ltXcap
	WAVE yc = root:AFM_ltYcap
	WAVE vc = root:AFM_ltVcap
	WAVE lc = root:AFM_ltLcap
	// Release the prior chunk's bank bindings + in-wave callback registrations BEFORE
	// re-arming (the err-100 fix; vendor code clears banks this way throughout).
	ir_StopOutWaveBank(0)
	ir_StopOutWaveBank(2)
	ir_StopInWaveBank(0)
	ir_StopInWaveBank(1)
	ir_StopInWaveBank(2)
	td_xSetOutWavePair(0, kEvent, "PIDSLoop.0.Setpoint", lx, "PIDSLoop.1.Setpoint", ly, interp)
	Printf "[arm/chunk] out0 XY ok\r"
	td_xSetOutWave(2, kEvent, "Output.A", lv, interp)
	Printf "[arm/chunk] out2 Output.A drive ok\r"
	td_xSetInWavePair(0, kEvent, "ZSensor", zc, "Deflection", dc, "AFM_LithoAdvance()", decim)
	Printf "[arm/chunk] in0 ZSensor/Deflection ok\r"
	td_xSetInWavePair(1, kEvent, "XSensor", xc, "YSensor", yc, "", decim)
	Printf "[arm/chunk] in1 XSensor/YSensor ok\r"
	Variable/G root:AFM_passArmed = 1                   // out-waves armed: a pass may play until td_Stop
	// MEASURED bias readback on bank 2 IN (the bank that drives the bias), same event + capture
	// length, NO fill callback: only bank 0 drives the advance (a 2nd advance callback would
	// double-advance and overflow the ARC callback queue). Lateral rides bank 2 as a pair.
	// FAIL-SAFE ladder: pair refused -> clear the bank, log once, arm UserIn2 alone; that
	// refused too -> clear, log once, and the chunk still writes + advances without it.
	Variable lErr = td_xSetInWavePair(2, kEvent, kBiasInChannel, vc, kLateralInChannel, lc, "", decim)
	Printf "[arm/chunk] in2 %s/%s measured bias + lateral (rc=%d)\r", kBiasInChannel, kLateralInChannel, lErr
	Variable vErr = lErr
	if(lErr != 0)
		ir_StopInWaveBank(2)
		if(k == 0)
			printf "AFM: %s/%s pair not armed on litho chain (td_xSetInWavePair err %d); arming %s alone\r", kBiasInChannel, kLateralInChannel, lErr, kBiasInChannel
		endif
		vErr = td_xSetInWave(2, kEvent, kBiasInChannel, vc, "", decim)
		Printf "[arm/chunk] in2 %s measured bias (rc=%d)\r", kBiasInChannel, vErr
		if(vErr != 0)
			ir_StopInWaveBank(2)
			if(k == 0)
				printf "AFM: %s tip-bias capture not armed on litho chain (td_xSetInWave err %d); write proceeds without it\r", kBiasInChannel, vErr
			endif
		endif
	endif
	Variable/G root:AFM_ltVArmed = (vErr == 0)
	Variable/G root:AFM_ltLArmed = (lErr == 0)
	td_WriteString("Event." + kEvent, "Once")            // FIRE this chunk
	Printf "[arm/chunk] FIRED Event.%s\r", kEvent
	return 0
End

// Capture-fill callback (runs INSIDE the controller callback): stream this chunk's
// captures, advance the chunk counter, then either finish or DEFER the next chunk.
// It must NOT arm the next chunk here (queue-stall, see AFM_LithoChainStart) and it
// must NOT zero the bias on the non-final branch (SEAM CONTINUITY): Output.A is zeroed
// only at the object's TRUE end, so the tip stays registered across a mid-stroke seam.
Function AFM_LithoAdvance()
	NVAR sock = root:AFM_lrSock
	SVAR name = root:AFM_ltName
	SVAR uid = root:AFM_lrUID
	NVAR chunk = root:AFM_ltChunk
	NVAR nchunks = root:AFM_ltNChunks
	NVAR abort = root:AFM_ltAbort
	// Stream the MEASURED bias and Lateral (each if armed), then ZSensor/Deflection + XSensor/YSensor
	// -- all under the object <name> + pass <uid>, so the driver accumulates same-channel
	// frames across chunks (the demux is channel-keyed, so the extra channels append per chunk).
	NVAR/Z vArmed = root:AFM_ltVArmed
	if(NVAR_Exists(vArmed) && vArmed)
		AFM_SendCapture(sock, uid, name, kBiasInChannel, "root:AFM_ltVcap")
	endif
	NVAR/Z lArmed = root:AFM_ltLArmed
	if(NVAR_Exists(lArmed) && lArmed)
		AFM_SendCapture(sock, uid, name, kLateralInChannel, "root:AFM_ltLcap")
	endif
	AFM_SendCapture(sock, uid, name, "ZSensor", "root:AFM_ltZcap")
	AFM_SendCapture(sock, uid, name, "Deflection", "root:AFM_ltDcap")
	AFM_SendCapture(sock, uid, name, "XSensor", "root:AFM_ltXcap")
	AFM_SendCapture(sock, uid, name, "YSensor", "root:AFM_ltYcap")
	chunk += 1
	NVAR/Z active = root:AFM_ltActive
	if(abort || chunk >= nchunks)
		AFM_SetValue("Output.A", 0)                     // TRUE end of the object: zero the bias
		if(NVAR_Exists(active) && active)               // send the terminal only once
			active = 0
			AFM_SendRunFrame(sock, "DONE " + uid + " " + name + "\n")
		endif
		Variable/G root:AFM_ltPending = 0               // nothing left to arm
	else
		Variable/G root:AFM_ltPending = 1               // defer the re-arm to AFM_LithoTick;
	endif                                               //   DO NOT zero Output.A here (seam)
	return 0
End

// Chain-advance background task (main thread): when a chunk has finished (AFM_ltPending),
// arm the next one here -- after the callback that set the flag has fully unwound, so the
// controller callback queue drains every chunk and never hits its ceiling. Bails on
// abort/finish (AFM_ltActive cleared) and stops itself. Same shape as AFM_ScanTick.
Function AFM_LithoTick(s)
	STRUCT WMBackgroundStruct &s
	NVAR/Z pending = root:AFM_ltPending
	NVAR/Z active = root:AFM_ltActive
	NVAR/Z abort = root:AFM_ltAbort
	NVAR/Z chunk = root:AFM_ltChunk
	if(!NVAR_Exists(pending) || !NVAR_Exists(active) || !NVAR_Exists(chunk))
		CtrlNamedBackground AFMlitho, stop
		return 0
	endif
	if(!active || (NVAR_Exists(abort) && abort))        // finished or aborted: stop advancing
		pending = 0
		CtrlNamedBackground AFMlitho, stop
		return 0
	endif
	if(!pending)
		return 0                                        // a chunk is still in flight
	endif
	pending = 0
	AFM_LithoArmChunk(chunk)                            // chunk already points at the next chunk
	return 0
End

// ================== always-on telemetry+safety poll (was Show Tip) ===========
// Mirror of Asylum's RedSpotBackground poll (docs/history/2026-08-25-ASYLUM_SHOW_TIP.md), widened into an
// ALWAYS-ON telemetry+safety poll: instead of drawing a marker on the image it PUSHES
// the live readout over SOCKIT so the HOST draws the tip AND runs the safety kernel. A
// CtrlNamedBackground ticks on the 60 Hz background clock; its rate = 60/period Hz, and
// the poll reads SIX channels with td_ReadValue -- tip X/Y (closed-loop LVDT), Output.Z
// (HV Z-drive), Deflection, Sum, Amplitude -- and sends one
//   TELEM <x> <y> <zdrive> <defl> <sum> <amp>\n
// per acting tick with the RAW sensor VOLTS. The host converts tip volts->um via hw_cal
// (the SAME conversion as the scan-capture path) and feeds the rail-dwell guard from the
// polled Output.Z; NO calibration is hardcoded here. This is a SCALAR poll -- it touches
// NO in-wave capture bank (unlike the reverted capture attempt, which armed a phantom
// bank 3 the hardware does not have).
//
// STARTED AT BRIDGE STARTUP (AFM_LithoRunStart), never gated on SHOWTIP -- so the safety
// channels are live the whole time the link is up, idle and mid-pass alike.
//
// ADAPTIVE RATE (60/period Hz) -- Joe's spec: 4 Hz idle, 10 Hz imaging + litho:
//   * a LITHO write is active (AFM_ltActive)  -> period=6  -> 10 Hz
//   * a SCAN is active (AFM_scActive)         -> period=6  -> 10 Hz
//   * IDLE (neither)                          -> period=15 -> 4 Hz  (cheap; enough for a
//     marker, and matches Asylum's own idle Show-Tip cadence).
// AFM_TipTick re-registers its own period ONLY when the desired rate changes (tracked
// in root:AFM_tipPeriod), so the rate follows the pass state no matter which order
// SHOWTIP and the pass start in, without ever re-registering every tick.
//
// Coexists with a running write/scan: td_ReadValue is READ-ONLY and this is a main-
// thread background task like AFM_ScanTick / AFM_LithoTick, so Igor's background
// tasks run cooperatively (no preemption) and AFM_SendStr lines never interleave.
// The armed out/in banks are untouched (this is a SCALAR poll -- NO in-wave capture
// bank; that is the whole point of the group-2 redo). The interleaved TELEM lines are
// recognized by the host router and routed to on_telem (a WS tip marker + the safety
// guard feed); the poll is ALWAYS on (started at AFM_LithoRunStart), so telemetry runs
// idle and during a pass alike, no host-side reader gymnastics needed.

// The CtrlNamedBackground period for the CURRENT pass state (rate = 60/period Hz):
// scan OR litho -> 6 (10 Hz), idle -> 15 (4 Hz). Joe's spec: 4 Hz idle, 10 Hz during
// imaging AND lithography. Read-only; safe if the state globals do not exist yet
// (treated as not-active -> the 4 Hz idle default).
Function AFM_TipDesiredPeriod()
	NVAR/Z lt = root:AFM_ltActive
	if(NVAR_Exists(lt) && lt)
		return 6                                        // litho write: 10 Hz
	endif
	NVAR/Z sc = root:AFM_scActive
	if(NVAR_Exists(sc) && sc)
		return 6                                        // scan: 10 Hz
	endif
	return 15                                           // idle default: 4 Hz
End

// SET THE INSTRUMENT IMAGING MODE (contact <-> AC) via the vendor's OWN mode-profile
// call -- the exact call Asylum's mode-profile files use, NOT a reimplementation. One
// call reconfigures the Z feedback InputChannel, gain sign, crosspoint (DC/AC), DDS
// drive (on for AC / off for contact) and the mode defaults:
//   Contact: MainPopupFunc("ImagingModePopup_0", 1, "Contact")
//   AC:      MainPopupFunc("ImagingModePopup_0", 2, "AC Mode")
// Source: MainPopupFunc is MainPanel.ipf:2562; the mode-name list ("Contact;AC Mode;..")
// is ScanMaster.ipf:8369; the vendor profiles Profiles\ContactModeTopography.txt and
// Profiles\ACAirTopography.txt invoke it verbatim. The wire token is the bare word
// ("Contact"/"AC") because "AC Mode" carries a space; we pass the full vendor name here.
//
// SAFETY-CRITICAL: this does NOT withdraw -- it stops the scan but leaves the tip on the
// surface, then changes the feedback channel + gain sign. The HOST (POST /api/mode)
// REFUSES the switch while the tip is engaged, so this is only ever reached with the tip
// withdrawn. RIG-PENDING: MainPopupFunc cannot be exercised off the instrument.
Function AFM_SetMode(mode)
	String mode
	if(cmpstr(mode, "AC", 1) == 0)
		MainPopupFunc("ImagingModePopup_0", 2, "AC Mode")
	else
		MainPopupFunc("ImagingModePopup_0", 1, "Contact")
	endif
	return 0
End

// READ THE LIVE CROSSPOINT (read-only mode read-back) and reply XPT <uid> InFast=<value>;
// The AUTHORITATIVE litho-AC guard reads this instead of trusting the fire-and-forget
// SETMODE state or a stale scan config. The mode discriminator is the InFast entry of the
// plain ARC "Crosspoint" group: FilterOut = contact (contact + current/c-AFM), ACDefl = AC.
// InFast is located by DIM-LABEL first, falling back to INDEX 2 of the ARC 16-output group
// order (InA, InB, InFast, ...). STRICTLY READ-ONLY (td_ReadGroup fills a text wave; nothing
// is routed or written), so it is safe to call mid-scan -- same class as the tip poll's
// td_ReadValue. Do NOT read/touch BNCOut0-2 (may be locked to Output.A on the rig).
// On failure (no data), reply ERR <uid> so the driver returns {} and the guard reads
// "unknown" (stays conservative -- a failed read never downgrades safety). Reply < 400 chars.
//   RIG-PENDING: td_ReadGroup + the reply cannot be exercised off the instrument (no XOP on
//   the dev box); the Python side is tested against the stub with mocked replies.
Function AFM_ReadXPT(uid, group)
	String uid, group
	NVAR sock = root:AFM_lrSock
	if(strlen(group) == 0)
		group = "Crosspoint"
	endif
	Make/T/O/N=0 root:AFM_xptTmp
	WAVE/T tmp = root:AFM_xptTmp
	Variable err = td_ReadGroup(group, tmp)
	Variable n = numpnts(tmp)
	if(n == 0)
		// No data -> treat as failure regardless of the return-code convention; the
		// driver reads an ERR reply as {} (guard "unknown", conservative).
		AFM_SendStr(sock, "ERR " + uid + " td_ReadGroup(" + group + ") returned no data (err " + num2istr(err) + ")\n")
		KillWaves/Z tmp
		return -1
	endif
	// Locate InFast by dim-label, else fall back to index 2 (ARC 16-output order).
	Variable i, idx = -1
	for(i = 0; i < n; i += 1)
		if(cmpstr(GetDimLabel(tmp, 0, i), "InFast") == 0)
			idx = i
			break
		endif
	endfor
	if(idx < 0 && n > 2)
		idx = 2
	endif
	String reply
	if(idx >= 0 && idx < n)
		reply = "XPT " + uid + " InFast=" + tmp[idx] + ";\n"
	else
		reply = "XPT " + uid + " \n"    // no InFast entry -> empty payload -> driver reads "unknown"
	endif
	AFM_SendStr(sock, reply)
	KillWaves/Z tmp
	return 0
End

// SHOWTIP ON/OFF is now a DISPLAY hint for the HOST only -- the telemetry+safety poll
// (AFMtip) is ALWAYS running (started at AFM_LithoRunStart) and must NEVER be stopped
// here, or the safety channels (rail-dwell / SUM / deflection) would go blind whenever
// the operator hid the tip marker. So this only ENSURES the poll is running (idempotent
// restart) on ON; it never stops it. The host owns whether to draw the marker.
Function AFM_ShowTip(on)
	Variable on
	if(on)
		NVAR/Z sock = root:AFM_lrSock
		if(NVAR_Exists(sock))
			Variable p0 = AFM_TipDesiredPeriod()
			Variable/G root:AFM_tipPeriod = p0
			CtrlNamedBackground AFMtip, period=(p0), proc=AFM_TipTick   // 60/period Hz
			CtrlNamedBackground AFMtip, start
		endif
	endif
	return 0
End

// One poll tick: keep the rate in step with the pass state, read the live position and
// push it. FAIL-SAFE + CHEAP -- if the link global is gone, stop the task; if either
// read errors (NaN/Inf), skip this tick so a bad read never sends a malformed frame or
// disturbs the write. Adjusting the period from within the task (main thread) is safe
// and only happens when the desired rate actually changes.
Function AFM_TipTick(s)
	STRUCT WMBackgroundStruct &s
	NVAR/Z sock = root:AFM_lrSock
	if(!NVAR_Exists(sock))
		CtrlNamedBackground AFMtip, stop
		return 0
	endif
	// Track the pass state: re-register the period ONLY when the desired rate changes
	// (4 Hz idle <-> 10 Hz scan/litho). Never stalls a pass -- read-only bookkeeping.
	Variable want = AFM_TipDesiredPeriod()
	NVAR/Z cur = root:AFM_tipPeriod
	if(!NVAR_Exists(cur) || cur != want)
		Variable/G root:AFM_tipPeriod = want
		CtrlNamedBackground AFMtip, period=(want)
	endif
	// Read the SIX poll channels. PER-CHANNEL NaN-safe: a channel that errors is
	// emitted as "nan" (AFM_TelemTok) rather than dropping the whole frame, so the tip
	// position never blinks out just because a safety channel hiccuped, and vice versa.
	//   RIG-PENDING: confirm the td_ReadValue names on the target controller --
	//   the Z channel (HV Z-drive; "Output.Z" MFP-3D, "Height" Cypher -- platform-gated
	//   below, matching the servo/withdraw fix), "Deflection" (feedback
	//   input), "DetectorSum" (photodiode sum; CONFIRMED on the Cypher and the MFP-3D,
	//   2026-09-15 -- lowercase "sum" reads 0 V on the Cypher and must not be used;
	//   matching the host's SUM interlock/clamp read in continuous.py::_read_sum, alias
	//   SUM_CHANNEL in service/hw_approach.py), and "amplitude" (LOWERCASE -- the
	//   confirmed rig scalar name; DISPLAY-only, the host shows it in AC mode).
	//   Amplitude is meaningful only in AC/tapping mode; it still rides every frame and
	//   the HOST decides whether to display it.
	Variable xv = td_ReadValue("XSensor")
	Variable yv = td_ReadValue("YSensor")
	// Z READ by alias "Height" on every platform (CLAUDE.md: channels by alias, never by
	// platform literal); "Output.Z" is dead on the Cypher (rig, 2026-09-25, spec 13.1 item 23).
	String zChan = "Height"
	Variable zv = td_ReadValue(zChan)
	Variable dv = td_ReadValue("Deflection")
	// "DetectorSum" (NOT "sum" -- lowercase reads 0 V on the Cypher; confirmed on the
	// Cypher and the MFP-3D, Joe, 2026-09-15). Mirrors SUM_CHANNEL in
	// service/hw_approach.py; keep the two in sync by hand (the .ipf cannot import it).
	Variable sv = td_ReadValue("DetectorSum")
	Variable av = td_ReadValue("amplitude")
	String msg
	msg = "TELEM " + AFM_TelemTok(xv) + " " + AFM_TelemTok(yv) + " " + AFM_TelemTok(zv)
	msg += " " + AFM_TelemTok(dv) + " " + AFM_TelemTok(sv) + " " + AFM_TelemTok(av) + "\n"
	AFM_SendTelem(sock, msg)                             // TELEM, or the same reads as TICK to a v3 host
	return 0
End

// One TELEM channel token: a %.9g-formatted number, or the literal "nan" when the
// read errored (numtype != 0 -> NaN/Inf). parse_telem on the host reads "nan" back to
// a real NaN, so a single bad channel is per-channel safe and never malforms the frame.
Function/S AFM_TelemTok(v)
	Variable v
	if(numtype(v) != 0)
		return "nan"
	endif
	String t
	sprintf t, "%.9g", v
	return t
End

// ======================= abort: Igor's own 5 s confirm =======================

// Re-armed by every abort, so the 5 s run from the latest one.
Function AFM_ArmAbortBackup()
	CtrlNamedBackground AFMabortBackup, stop
	CtrlNamedBackground AFMabortBackup, period=(kAbortBackupTicks), proc=AFM_AbortBackup
	CtrlNamedBackground AFMabortBackup, start=(ticks + kAbortBackupTicks)
	return 0
End

// One shot, 5 s after the latest abort; nothing to do if the host already confirmed.
// Height not under 1 V: the vendor Withdraw once, and Igor's fault flag for the host.
// The abort flag stays set: a later abort skips td_Stop until a confirm, unless a pass armed.
Function AFM_AbortBackup(s)
	STRUCT WMBackgroundStruct &s
	NVAR/Z stopped = root:AFM_abortStopped
	if(NVAR_Exists(stopped) && stopped)
		Variable h = td_ReadValue("Height")
		if(h < kRetractedHeightV)
			stopped = 0
			AFM_V3BackupRan(h, 0)                        // wire v3: REPORT backup <h> cleared
		else
			DoScanFunc("StopEngageButton")
			Variable/G root:AFM_abortFault = 1
			AFM_V3BackupRan(h, 1)                        // wire v3: REPORT backup <h> withdraw
		endif
	endif
	return 1                                             // nonzero stops the task: one shot
End

// Verbs outside the main switch: the wire v3 verbs (AFM_V3Dispatch, end of file) and GETSTR.
// GETSTR <uid> <name>: read one controller string parameter with td_ReadString and reply
// XPT <uid> <name>=<text>;  Used to ask which channel a PIDS loop drives
// (PIDSLoop.<n>.OutputChannel). td_ReadString returns "*ERROR*" instead of raising: a failed read replies ERR <uid> exception: ...
Function AFM_DispatchOther(sock, verb, line)
	Variable sock
	String verb, line

	if(AFM_V3IsVerb(verb) && ItemsInList(line, " ") > 1)     // wire v3: every v3 line carries a seq
		return AFM_V3Dispatch(verb, line)
	endif
	if(cmpstr(verb, "GETSTR") != 0)
		AFM_SendStr(sock, "ERR - unknown verb: " + verb + "\n")   // '-' = no uid to echo
		return -1
	endif
	String uid = StringFromList(1, line, " ")
	String name = StringFromList(2, line, " ")
	String text = td_ReadString(name)
	if(GetRTError(1) != 0 || cmpstr(text, "*ERROR*") == 0)
		AFM_SendStr(sock, "ERR " + uid + " exception: td_ReadString(" + name + ") failed\n")
		return -1
	endif
	AFM_SendStr(sock, "XPT " + uid + " " + name + "=" + text + ";\n")
	return 0
End

// One string global, or `dflt` when it does not exist. SelectString evaluates every argument
// (IgorMan V-624), so SelectString(SVAR_Exists(s), "", s) would read a null SVAR.
Function/S AFM_GlobalStr(path, dflt)
	String path, dflt
	SVAR/Z s = $path
	if(SVAR_Exists(s))
		return s
	endif
	return dflt
End

// ================================ wire v3 ====================================
// docs/migration/plan/phase-5-wire-v3-contract.md; the fake Igor (hardware/igor_stub_v3.py)
// is the executable spec, and docs/migration/plan/w1-ipf-map.md maps each verb to it.
// A v3 line runs the same arm, fire, set and make-safe functions as its v2 line, so Igor's
// td_/ir_ commands do not change. Only the replies and the up-messages are new:
//   REPLY <seq> ok|err <body>
//   REPORT <useq> <tick> <mark> <fault> <route> <run> <event>    (the last kRingSize are kept)
//   CAP <useq> <tick> <mark> <run> <step> <alias> <v;v;...>
//   TICK <useq> <tick> <mark> <fault> - <x> <y> <z> <defl> <sum> <amp>
// mark = root:AFM_abortStopped, fault = root:AFM_abortFault. The host is the link whose HELLO
// carried kWireV3; only the host may LOAD, PLAY, SET and CLEARFAULT.

Constant kWireV3 = 3                     // the proto a HELLO must carry
Constant kIpfVersion = 3                 // HELLO ipf=
StrConstant kIpfSha = "0000000000000000"
Constant kSockDeadParkTicks = 300        // kSockDeadParkS, 5 s (Joe, 2026-10-27)
Constant kRingSize = 256                 // REPORTs kept for FETCH

// The v3 globals, made once per experiment, so useq is never reset while the experiment lives.
Function AFM_V3Init()
	NVAR/Z ready = root:AFM_v3Ready
	if(NVAR_Exists(ready) && ready == kIpfVersion)
		return 0
	endif
	Variable/G root:AFM_v3Ready = kIpfVersion
	Variable/G root:AFM_v3Useq = 0                       // one series for REPORT, CAP and TICK
	Variable/G root:AFM_v3HostSock = -1                  // the link that said HELLO, -1 none
	Variable/G root:AFM_v3DeadSince = -1                 // ticks when a send to the host failed, -1 alive
	Variable/G root:AFM_v3TipDown = 0                    // contract 5b, from the loop-2 commands Igor ran
	String/G root:AFM_v3Loop2 = "none"                   // "<setpoint>,<min>,<max>" of loop 2, or none
	Variable/G root:AFM_v3VendorDone = 0                 // the vendor Withdraw ran for this make-safe
	Variable/G root:AFM_v3StopSeen = 0                   // root:AFM_abortStopped as v3 last saw it set
	String/G root:AFM_v3Route = "com"                    // where the running command came from
	String/G root:AFM_v3Why = "stop"                     // madesafe <why>: stop, or socket for the guard
	String/G root:AFM_v3Run = ""                         // the PLAY in flight, "" none
	String/G root:AFM_v3Kind = ""
	Variable/G root:AFM_v3Steps = 0
	Variable/G root:AFM_v3Step = 0
	Variable/G root:AFM_v3StepPts = -1                   // points of the open step, -1 before its first CAP
	Variable/G root:AFM_v3CapPts = 0
	String/G root:AFM_v3CapChan = ""
	Variable/G root:AFM_v3Total = 0
	Variable/G root:AFM_v3Closed = 0
	Variable/G root:AFM_v3Hold = 0                       // 1 while PLAY arms: frames wait for 'armed'
	String/G root:AFM_v3Totals = ""                      // LOAD's n_total per wave, name:n;
	Variable/G root:AFM_v3RingN = 0
	Variable/G root:AFM_v3RingHead = 0
	Variable/G root:AFM_v3Evicted = 0                    // the newest useq that left the ring
	Make/T/O/N=(kRingSize) root:AFM_v3RingTxt
	Make/D/O/N=(kRingSize) root:AFM_v3RingSeq
	Make/T/O/N=0 root:AFM_v3Held
	return 0
End

// A new link (AFM_LithoRunStart) has no host until it says HELLO.
Function AFM_V3LinkOpened()
	AFM_V3Init()
	Variable/G root:AFM_v3HostSock = -1
	return 0
End

Function AFM_V3IsVerb(verb)
	String verb
	return WhichListItem(verb, "HELLO;READ;STATUS;FETCH;LOAD;LOADAPP;PLAY;SET;MAKESAFE;CONFIRMSAFE;VENDORWITHDRAW;CLEARFAULT;") >= 0
End

// One v2 dispatch from the socket: a make-safe it causes is stamped route sock.
Function AFM_DispatchFromSocket(line)
	String line
	String/G root:AFM_v3Route = "sock"
	AFM_Dispatch(line)
	String/G root:AFM_v3Route = "com"                    // AFM_SafePark run any other way came over COM
	return 0
End

// A MAKESAFE runs before the other whole lines that arrived with it, and each PLAY, LOAD,
// LOADAPP or SET among them is dropped with REPLY <seq> err dropped (contract 6). The other
// lines, then any partial line, go back in order to AFM_DispatchTick's loop.
Function/S AFM_V3Intake(buf)
	String buf
	if(strsearch(buf, "MAKESAFE", 0) < 0)
		return buf
	endif
	String rest = buf, kept = "", oneLine, verb
	Variable nl, made = 0
	do                                                   // pass 1: every whole MAKESAFE line, in order
		nl = strsearch(rest, "\n", 0)
		if(nl < 0)
			break
		endif
		oneLine = rest[0, nl - 1]
		rest = rest[nl + 1, strlen(rest) - 1]
		if(cmpstr(StringFromList(0, oneLine, " "), "MAKESAFE") == 0)
			AFM_DispatchFromSocket(oneLine)
			made += 1
		endif
	while(1)
	if(made == 0)
		return buf                                       // "MAKESAFE" only in a partial line so far
	endif
	rest = buf
	do                                                   // pass 2: drop the queued state changes
		nl = strsearch(rest, "\n", 0)
		if(nl < 0)
			break
		endif
		oneLine = rest[0, nl - 1]
		rest = rest[nl + 1, strlen(rest) - 1]
		verb = StringFromList(0, oneLine, " ")
		if(WhichListItem(verb, "PLAY;LOAD;LOADAPP;SET;") >= 0)
			AFM_V3Reply(StringFromList(1, oneLine, " "), "err", "dropped")
		elseif(cmpstr(verb, "MAKESAFE") != 0)
			kept += oneLine + "\n"
		endif
	while(1)
	return kept + rest
End

// Every v3 verb (contract 2). The caller has checked that the line carries a seq.
Function AFM_V3Dispatch(verb, line)
	String verb, line
	AFM_V3Init()
	String seq = StringFromList(1, line, " ")
	strswitch(verb)
		case "HELLO":
			AFM_V3Hello(seq, line)
			break
		case "READ":
			AFM_V3Read(seq, line)
			break
		case "STATUS":
			AFM_V3Reply(seq, "ok", AFM_V3Status())
			break
		case "FETCH":
			AFM_V3Fetch(seq, str2num(StringFromList(2, line, " ")))
			break
		case "LOAD":
		case "LOADAPP":
			AFM_V3Load(seq, line)
			break
		case "PLAY":
			AFM_V3Play(seq, line)
			break
		case "SET":
			AFM_V3Set(seq, line)
			break
		case "MAKESAFE":
			AFM_V3Reply(seq, "ok", "")                   // ok at once; REPORT madesafe follows
			AFM_SafePark()                               // the one make-safe, contract 5a
			break
		case "CONFIRMSAFE":
			AFM_V3ConfirmSafe(seq)
			break
		case "VENDORWITHDRAW":
			AFM_V3VendorWithdraw(seq)
			break
		case "CLEARFAULT":
			AFM_V3ClearFault(seq)
			break
	endswitch
	return 0
End

// ---- v3 up-messages ----

Function AFM_V3Reply(seq, status, body)
	String seq, status, body
	NVAR/Z sock = root:AFM_lrSock
	if(!NVAR_Exists(sock))
		return -1
	endif
	String msg = "REPLY " + seq + " " + status
	if(strlen(body) > 0)
		msg += " " + body
	endif
	return AFM_SendStr(sock, msg + "\n")
End

// 1 when the link up now is the host and has not failed a send.
Function AFM_V3Host()
	NVAR/Z hostSock = root:AFM_v3HostSock
	NVAR/Z sock = root:AFM_lrSock
	NVAR/Z dead = root:AFM_v3DeadSince
	if(!NVAR_Exists(hostSock) || !NVAR_Exists(sock) || !NVAR_Exists(dead))
		return 0
	endif
	return hostSock >= 0 && hostSock == sock && dead < 0
End

// Send one up-message to the host. A failed send counts the link dead (contract 5b). Success
// is 0 or the byte count (which one SOCKIT returns is not confirmed); anything else, or
// a runtime error, is a failure. RIG-PENDING: SOCKITsendmsgF's success and closed-link returns.
Function AFM_V3ToHost(msg)
	String msg
	if(!AFM_V3Host())
		return -1
	endif
	NVAR sock = root:AFM_lrSock
	Variable stale = GetRTError(1)                       // clear an older error: only this send counts
	Variable rc = AFM_SendStr(sock, msg)
	Variable rtErr = GetRTError(1)
	if(rtErr != 0 || (rc != 0 && rc != strlen(msg)))
		AFM_V3SockDied()
	endif
	return rc
End

// The host link is dead from now; AFMv3guard parks a tip left down (contract 5b).
Function AFM_V3SockDied()
	NVAR dead = root:AFM_v3DeadSince
	if(dead < 0)
		dead = ticks
		CtrlNamedBackground AFMv3guard, period=3, proc=AFM_V3GuardTick
		CtrlNamedBackground AFMv3guard, start
	endif
	return 0
End

Function AFM_V3Mark()
	NVAR/Z stopped = root:AFM_abortStopped
	return NVAR_Exists(stopped) && stopped
End

Function/S AFM_V3Stamps()
	return num2istr(AFM_V3Mark()) + " " + num2istr(AFM_ReadAbortFault())
End

// A number as the fake writes it (%.12g; nan, inf, -inf).
Function/S AFM_V3Num(v)
	Variable v
	String t
	if(numtype(v) == 2)
		return "nan"
	endif
	if(numtype(v) == 1)
		if(v > 0)
			return "inf"
		endif
		return "-inf"
	endif
	sprintf t, "%.12g", v
	return t
End

// `s` without one trailing `tail` character.
Function/S AFM_V3Chop(s, tail)
	String s, tail
	Variable n = strlen(s)
	if(n > 0 && cmpstr(s[n - 1, n - 1], tail) == 0)
		return s[0, n - 2]
	endif
	return s
End

Function AFM_V3Report(run, event, route)
	String run, event, route
	AFM_V3Init()
	NVAR useq = root:AFM_v3Useq
	useq += 1
	String msg = "REPORT " + num2istr(useq) + " " + num2istr(ticks) + " " + AFM_V3Stamps()
	msg += " " + route + " " + run + " " + event
	AFM_V3Ring(useq, msg)
	return AFM_V3ToHost(msg + "\n")
End

Function AFM_V3Ring(useq, msg)
	Variable useq
	String msg
	WAVE/T txt = root:AFM_v3RingTxt
	WAVE seqs = root:AFM_v3RingSeq
	NVAR n = root:AFM_v3RingN
	NVAR head = root:AFM_v3RingHead
	NVAR evicted = root:AFM_v3Evicted
	Variable i
	if(n < kRingSize)
		i = mod(head + n, kRingSize)
		n += 1
	else
		evicted = seqs[head]                             // the oldest REPORT leaves the ring
		i = head
		head = mod(head + 1, kRingSize)
	endif
	seqs[i] = useq
	txt[i] = msg
	return 0
End

Function/S AFM_V3RingSpan()
	WAVE seqs = root:AFM_v3RingSeq
	NVAR n = root:AFM_v3RingN
	NVAR head = root:AFM_v3RingHead
	if(n == 0)
		return "0-0"
	endif
	return num2istr(seqs[head]) + "-" + num2istr(seqs[mod(head + n - 1, kRingSize)])
End

// TELEM to a v2 link; to the v3 host the same six reads as TICK (contract 2c).
Function AFM_SendTelem(sock, msg)
	Variable sock
	String msg
	if(!AFM_V3Host())
		return AFM_SendStr(sock, msg)
	endif
	String reads = msg[5, strlen(msg) - 1]                // the six reads after "TELEM"
	NVAR useq = root:AFM_v3Useq
	useq += 1
	return AFM_V3ToHost("TICK " + num2istr(useq) + " " + num2istr(ticks) + " " + AFM_V3Stamps() + " -" + reads)
End

// ---- the run: v2 completion frames as v3 (the fake's _RunLink) ----

// While a PLAY is in flight its pass functions are the only senders of pass frames, so every
// frame is that run's and goes out as v3, whatever uid it carries: AFM_LithoRunStart blanks
// the uid globals on a reconnect, and the run must still end. With no PLAY, frames go as today.
Function AFM_SendRunFrame(sock, msg)
	Variable sock
	String msg
	SVAR/Z run = root:AFM_v3Run
	if(SVAR_Exists(run) && strlen(run) > 0)
		return AFM_V3RunFrame(msg)
	endif
	return AFM_SendStr(sock, msg)
End

// CAPTURE becomes CAP; DONE, SCANDONE and ERR become the run's one terminal REPORT. AFM_Abort's
// ABORTED and SCANDONE -1 are left to AFM_V3ParkEnds, so nothing but the send runs before the
// re-arm. Frames sent while PLAY is still arming wait until its 'armed' REPORT has gone.
Function AFM_V3RunFrame(msg)
	String msg
	NVAR hold = root:AFM_v3Hold
	SVAR run = root:AFM_v3Run
	if(hold)
		WAVE/T held = root:AFM_v3Held
		Variable n = numpnts(held)
		Redimension/N=(n + 1) held
		held[n] = msg
		return 0
	endif
	if(strlen(run) == 0)
		return 0                                         // the run has already ended
	endif
	msg = AFM_V3Chop(msg, "\n")
	String verb = StringFromList(0, msg, " "), text
	Variable sp1, sp2
	strswitch(verb)
		case "CAPTURE":
			return AFM_V3Cap(msg)
		case "DONE":
			return AFM_V3RunEnds("done")
		case "SCANDONE":
			if(str2num(StringFromList(2, msg, " ")) < 0)
				return 0                                 // AFM_Abort's: AFM_V3ParkEnds reports it
			endif
			return AFM_V3RunEnds("done")
		case "ABORTED":
			return 0                                     // only AFM_Abort sends it: AFM_V3ParkEnds reports it
		case "ERR":
			sp1 = strsearch(msg, " ", 0)
			sp2 = strsearch(msg, " ", sp1 + 1)
			text = msg[sp2 + 1, strlen(msg) - 1]         // ERR <uid> <text>
			return AFM_V3RunEnds("error -1 " + text)
	endswitch
	return 0
End

// CAPTURE <run> <name> <alias> <v;...;> as CAP. Step k is scan line L<k>, the litho chunk being
// streamed (AFM_LithoAdvance counts it after), or 0 for a move or a single write.
Function AFM_V3Cap(msg)
	String msg
	NVAR useq = root:AFM_v3Useq
	NVAR step = root:AFM_v3Step
	NVAR stepPts = root:AFM_v3StepPts
	NVAR capPts = root:AFM_v3CapPts
	SVAR capChan = root:AFM_v3CapChan
	SVAR run = root:AFM_v3Run
	SVAR kind = root:AFM_v3Kind
	String uid = StringFromList(1, msg, " ")             // "" after a reconnect, not the run token
	String name = StringFromList(2, msg, " ")
	String chan = StringFromList(3, msg, " ")
	Variable at = strlen("CAPTURE " + uid + " " + name + " " + chan + " ")
	String vals = msg[at, strlen(msg) - 1]
	vals = AFM_V3Chop(vals, ";")                         // v2 frames end in "v;"
	Variable k = 0
	if(cmpstr(kind, "scanchain") == 0)
		k = str2num(name[1, strlen(name) - 1])
	elseif(cmpstr(kind, "lithochain") == 0)
		NVAR/Z chunk = root:AFM_ltChunk
		if(NVAR_Exists(chunk))
			k = chunk
		endif
	endif
	if(k != step)
		AFM_V3CloseStep()
		step = k
		capChan = ""
	endif
	if(cmpstr(chan, capChan, 1) != 0)                    // a capture longer than kCapChunkPoints comes in pieces
		capChan = chan
		capPts = 0
	endif
	capPts += ItemsInList(vals, ";")
	stepPts = capPts
	useq += 1
	return AFM_V3ToHost("CAP " + num2istr(useq) + " " + num2istr(ticks) + " " + num2istr(AFM_V3Mark()) + " " + run + " " + num2istr(k) + " " + chan + " " + vals + "\n")
End

// REPORT step <k> <n> <points> for the step whose captures have all gone.
Function AFM_V3CloseStep()
	NVAR step = root:AFM_v3Step
	NVAR steps = root:AFM_v3Steps
	NVAR stepPts = root:AFM_v3StepPts
	NVAR total = root:AFM_v3Total
	NVAR closed = root:AFM_v3Closed
	SVAR run = root:AFM_v3Run
	if(stepPts < 0)
		return 0
	endif
	AFM_V3Report(run, "step " + num2istr(step) + " " + num2istr(steps) + " " + num2istr(stepPts), "sock")
	total += stepPts
	closed += 1
	stepPts = -1
	return 0
End

// The run's one terminal: done <points> (done <lines> for a scan), aborted, or error.
Function AFM_V3RunEnds(event)
	String event
	SVAR run = root:AFM_v3Run
	SVAR kind = root:AFM_v3Kind
	NVAR total = root:AFM_v3Total
	NVAR closed = root:AFM_v3Closed
	AFM_V3CloseStep()
	if(cmpstr(event, "done") == 0)
		if(cmpstr(kind, "scanchain") == 0)
			event += " " + num2istr(closed)
		else
			event += " " + num2istr(total)
		endif
	endif
	String ended = run
	run = ""                                             // contract 4 rule 9: the run is over first
	return AFM_V3Report(ended, event, "sock")
End

// ---- v3 verbs ----

Function/S AFM_V3HelloBody()
	String s = "proto=" + num2istr(kWireV3) + ";ipf=" + num2istr(kIpfVersion) + ";sha=" + kIpfSha + ";"
	s += "MicroscopeID=" + AFM_GetGVTok("MicroscopeID") + ";clock=" + num2istr(ticks) + ";"
	s += "ring=" + AFM_V3RingSpan() + ";mark=" + num2istr(AFM_V3Mark()) + ";fault=" + num2istr(AFM_ReadAbortFault()) + ";"
	return s + AFM_MacroStartup()
End

// HELLO <seq> <proto> <svc_version>: a matching proto makes this link the host.
Function AFM_V3Hello(seq, line)
	String seq, line
	NVAR/Z sock = root:AFM_lrSock
	NVAR hostSock = root:AFM_v3HostSock
	String body = AFM_V3HelloBody()
	if(str2num(StringFromList(2, line, " ")) != kWireV3)
		if(NVAR_Exists(sock) && hostSock == sock)
			hostSock = -1                                // a mismatched HELLO takes the rights away
		endif
		return AFM_V3Reply(seq, "err", "mismatch " + body)
	endif
	if(NVAR_Exists(sock))
		hostSock = sock
	endif
	Variable/G root:AFM_v3DeadSince = -1
	CtrlNamedBackground AFMv3guard, stop                 // the host is back: nothing to park
	return AFM_V3Reply(seq, "ok", body)
End

// READ <seq> value|gv|flag|str|xpt <name>: read-only, from anyone.
Function AFM_V3Read(seq, line)
	String seq, line
	String what = StringFromList(2, line, " ")
	String name = StringFromList(3, line, " ")
	strswitch(what)
		case "value":
			return AFM_V3Reply(seq, "ok", AFM_V3Num(AFM_GetValue(name)))
		case "gv":
			return AFM_V3Reply(seq, "ok", AFM_GetGVTok(name))
		case "flag":
			if(cmpstr(name, "abortFault", 1) != 0)
				return AFM_V3Reply(seq, "err", "bad flag " + name)
			endif
			return AFM_V3Reply(seq, "ok", num2istr(AFM_ReadAbortFault()))
		case "str":
			return AFM_V3ReadStr(seq, name)
		case "xpt":
			return AFM_V3ReadXpt(seq, name)
	endswitch
	return AFM_V3Reply(seq, "err", "bad read " + what)
End

// td_ReadString answers "*ERROR*" instead of raising, so that text is a failed read.
Function AFM_V3ReadStr(seq, name)
	String seq, name
	String text = td_ReadString(name)
	if(GetRTError(1) != 0 || cmpstr(text, "*ERROR*") == 0)
		return AFM_V3Reply(seq, "err", "td_ReadString(" + name + ") failed")
	endif
	return AFM_V3Reply(seq, "ok", name + "=" + text + ";")
End

// The crosspoint group as k=v; (read-only td_ReadGroup). InFast falls back to index 2 of the
// ARC 16-output order when it has no dimension label, as in AFM_ReadXPT.
Function AFM_V3ReadXpt(seq, group)
	String seq, group
	if(strlen(group) == 0)
		group = "Crosspoint"
	endif
	Make/T/O/N=0 root:AFM_xptTmp
	WAVE/T tmp = root:AFM_xptTmp
	Variable err = td_ReadGroup(group, tmp)
	Variable n = numpnts(tmp), i, fast = -1
	String body = "", lbl
	for(i = 0; i < n; i += 1)
		lbl = GetDimLabel(tmp, 0, i)
		if(strlen(lbl) > 0)
			body += lbl + "=" + tmp[i] + ";"
			if(cmpstr(lbl, "InFast") == 0)
				fast = i
			endif
		endif
	endfor
	if(fast < 0 && n > 2)
		body += "InFast=" + tmp[2] + ";"
	endif
	KillWaves/Z tmp
	if(n == 0)
		return AFM_V3Reply(seq, "err", "td_ReadGroup(" + group + ") returned no data (err " + num2istr(err) + ")")
	endif
	return AFM_V3Reply(seq, "ok", body)
End

Function AFM_V3DeadSeconds()
	NVAR dead = root:AFM_v3DeadSince
	if(dead < 0)
		return 0
	endif
	return (ticks - dead) / 60
End

Function/S AFM_V3Status()
	SVAR run = root:AFM_v3Run
	SVAR kind = root:AFM_v3Kind
	SVAR loop2 = root:AFM_v3Loop2
	NVAR step = root:AFM_v3Step
	NVAR steps = root:AFM_v3Steps
	NVAR tipDown = root:AFM_v3TipDown
	String s = "pass=none;run=-;step=0/0;"
	if(strlen(run) > 0)
		s = "pass=" + kind + ";run=" + run + ";step=" + num2istr(step) + "/" + num2istr(steps) + ";"
	endif
	s += "mark=" + num2istr(AFM_V3Mark()) + ";fault=" + num2istr(AFM_ReadAbortFault()) + ";tipdown=" + num2istr(tipDown) + ";"
	s += "loop2=" + loop2 + ";sock_dead_s=" + AFM_V3Num(AFM_V3DeadSeconds()) + ";"
	return s + "ring=" + AFM_V3RingSpan() + ";tick=" + num2istr(ticks) + ";"
End

// FETCH <seq> <from>: every ringed REPORT from useq `from` on, unchanged, then the span.
Function AFM_V3Fetch(seq, from)
	String seq
	Variable from
	WAVE/T txt = root:AFM_v3RingTxt
	WAVE seqs = root:AFM_v3RingSeq
	NVAR n = root:AFM_v3RingN
	NVAR head = root:AFM_v3RingHead
	NVAR evicted = root:AFM_v3Evicted
	NVAR useq = root:AFM_v3Useq
	NVAR sock = root:AFM_lrSock
	Variable i, k
	if(from <= evicted)
		k = useq + 1
		if(n > 0)
			k = seqs[head]
		endif
		return AFM_V3Reply(seq, "err", "gone " + num2istr(k))
	endif
	for(i = 0; i < n; i += 1)
		k = mod(head + i, kRingSize)
		if(seqs[k] >= from)
			AFM_SendStr(sock, txt[k] + "\n")
		endif
	endfor
	return AFM_V3Reply(seq, "ok", AFM_V3RingSpan())
End

Function AFM_V3NotHost(seq)
	String seq
	if(AFM_V3Host())
		return 0
	endif
	AFM_V3Reply(seq, "err", "host")
	return 1
End

// LOAD <seq> <name> <n_total> <v;...> | LOADAPP <seq> <name> <v;...>: the reply goes only once
// the piece is stored in root:<name>, so the host paces itself on it (contract 2d).
Function AFM_V3Load(seq, line)
	String seq, line
	if(AFM_V3NotHost(seq))
		return 0
	endif
	Variable isApp = cmpstr(StringFromList(0, line, " "), "LOADAPP") == 0
	String name = StringFromList(2, line, " ")
	String body = StringFromList(4 - isApp, line, " ")
	Variable nPiece = ItemsInList(body, ";")
	SVAR totals = root:AFM_v3Totals
	if(nPiece > kCapChunkPoints)
		return AFM_V3Reply(seq, "err", "bad piece " + num2istr(nPiece))
	endif
	WAVE/Z w = root:$name
	if(isApp && !WaveExists(w))
		return AFM_V3Reply(seq, "err", "bad noname " + name)
	endif
	if(isApp)
		if(nPiece > 0)
			AFM_AppendWave(name, body)
		endif
	else
		if(nPiece > 0)
			AFM_StoreWave(name, body)
		else
			Make/O/N=0 root:$name
		endif
		totals = ReplaceNumberByKey(name, totals, str2num(StringFromList(3, line, " ")))
	endif
	WAVE stored = root:$name
	Variable have = numpnts(stored)
	Variable total = NumberByKey(name, totals)
	if(numtype(total) != 0)
		total = have
	endif
	return AFM_V3Reply(seq, "ok", name + " " + num2istr(have) + "/" + num2istr(total))
End

// PLAY <seq> <kind> <run> ...: refused while a make-safe is unconfirmed, a fault is set or a
// run is in flight (contract 2b, section 12); otherwise armed through today's arm functions.
Function AFM_V3Play(seq, line)
	String seq, line
	if(AFM_V3NotHost(seq))
		return 0
	endif
	SVAR run = root:AFM_v3Run
	if(AFM_V3Mark())
		return AFM_V3Reply(seq, "err", "mark")
	elseif(AFM_ReadAbortFault())
		return AFM_V3Reply(seq, "err", "fault")
	elseif(strlen(run) > 0)
		return AFM_V3Reply(seq, "err", "busy")
	endif
	String kind = StringFromList(2, line, " ")
	String token = StringFromList(3, line, " ")
	Variable steps = AFM_V3PlaySteps(kind, line)
	if(steps == -1)
		return AFM_V3Reply(seq, "err", "bad kind " + kind)
	elseif(steps < 0)
		return AFM_V3Reply(seq, "err", "bad args")
	endif
	String missing = AFM_V3MissingWave(kind, line)
	if(strlen(missing) > 0)
		return AFM_V3Reply(seq, "err", "bad noname " + missing)
	endif
	AFM_V3RunBegins(kind, token, steps)
	AFM_V3Arm(kind, line)
	AFM_V3Reply(seq, "ok", "armed " + token)
	AFM_V3Report(token, "armed", "sock")
	return AFM_V3Release()
End

// The steps a PLAY reports: 1 for a move or a single write, the chunks of a chain, the lines
// of a scan. -1 for an unknown kind, -2 for a wrong token count.
Function AFM_V3PlaySteps(kind, line)
	String kind, line
	Variable nTok = ItemsInList(line, " ")
	strswitch(kind)
		case "move":
			if(nTok != 9)
				return -2
			endif
			return 1
		case "litho1":
			if(nTok != 11)
				return -2
			endif
			return 1
		case "lithochain":
			if(nTok != 11)
				return -2
			endif
			String bwName = StringFromList(8, line, " ")
			WAVE/Z bw = root:$bwName
			if(!WaveExists(bw))
				return 1
			endif
			return max(1, numpnts(bw) - 1)
		case "scanchain":
			if(nTok != 11)
				return -2
			endif
			return str2num(StringFromList(7, line, " "))
	endswitch
	return -1
End

Function AFM_V3RunBegins(kind, token, steps)
	String kind, token
	Variable steps
	String/G root:AFM_v3Run = token
	String/G root:AFM_v3Kind = kind
	Variable/G root:AFM_v3Steps = steps
	Variable/G root:AFM_v3Step = 0
	Variable/G root:AFM_v3StepPts = -1
	Variable/G root:AFM_v3CapPts = 0
	String/G root:AFM_v3CapChan = ""
	Variable/G root:AFM_v3Total = 0
	Variable/G root:AFM_v3Closed = 0
	Variable/G root:AFM_v3Hold = 1
	Make/T/O/N=0 root:AFM_v3Held
	return 0
End

// Each kind runs the v2 lines it replaces (contract 3): move = OUTPAIR + INPAIR + FIRE on event
// 0, litho1 = ARMFIRE, lithochain = LITHOCHAIN, scanchain = SCANCHAIN; the run token is the uid.
Function AFM_V3Arm(kind, line)
	String kind, line
	String t3 = StringFromList(3, line, " "), t4 = StringFromList(4, line, " ")
	String t5 = StringFromList(5, line, " "), t6 = StringFromList(6, line, " ")
	String t7 = StringFromList(7, line, " "), t8 = StringFromList(8, line, " ")
	String t9 = StringFromList(9, line, " "), t10 = StringFromList(10, line, " ")
	strswitch(kind)
		case "move":
			AFM_ArmOutPair(0, "0", "PIDSLoop.0.Setpoint", t4, "PIDSLoop.1.Setpoint", t5, str2num(t6))
			AFM_ArmInPair(0, "0", "XSensor", "YSensor", str2num(t7), str2num(t8))
			String/G root:AFM_lrUID = t3
			AFM_FireEvent("move", "0")
			break
		case "litho1":
			AFM_V3LithoWave(t5, "XLitho")                // ARMFIRE drives the fixed waves
			AFM_V3LithoWave(t6, "YLitho")
			AFM_V3LithoWave(t7, "VLitho")
			String/G root:AFM_lrUID = t3
			AFM_ArmAndFireLitho(t4, str2num(t8), str2num(t9), str2num(t10))
			break
		case "lithochain":
			String/G root:AFM_lrUID = t3
			AFM_LithoChainStart(t4, t5, t6, t7, t8, str2num(t9), str2num(t10))
			break
		case "scanchain":
			String/G root:AFM_scUID = t3
			AFM_ScanChainStart(t4, t5, str2num(t6), str2num(t7), str2num(t8), str2num(t9), errChan=t10)
			break
	endswitch
	return 0
End

// litho1: the first of its three named waves that is not loaded, or "".
Function/S AFM_V3MissingWave(kind, line)
	String kind, line
	String wn
	Variable i
	if(cmpstr(kind, "litho1") != 0)
		return ""
	endif
	for(i = 5; i <= 7; i += 1)
		wn = StringFromList(i, line, " ")
		WAVE/Z w = root:$wn
		if(!WaveExists(w))
			return wn
		endif
	endfor
	return ""
End

// Copy a loaded wave onto the name ARMFIRE reads, when the host named its own.
Function AFM_V3LithoWave(have, want)
	String have, want
	WAVE/Z src = root:$have
	if(cmpstr(have, want) != 0 && WaveExists(src))
		Duplicate/O src, root:$want
	endif
	return 0
End

// 'armed' has gone: send the frames that arrived while PLAY armed, in order.
Function AFM_V3Release()
	NVAR hold = root:AFM_v3Hold
	WAVE/T held = root:AFM_v3Held
	Variable i, n = numpnts(held)
	hold = 0
	for(i = 0; i < n; i += 1)
		AFM_V3RunFrame(held[i])
	endfor
	Redimension/N=0 held
	return 0
End

// SET <seq> value|loop|ramp|loopstop|mode|showtip ...: never refused for mark or fault (the
// operator keeps the loops). The echo is what Igor applied, with the td_ code where there is one.
Function AFM_V3Set(seq, line)
	String seq, line
	if(AFM_V3NotHost(seq))
		return 0
	endif
	String what = StringFromList(2, line, " ")
	String a = StringFromList(3, line, " "), b = StringFromList(4, line, " ")
	String c = StringFromList(5, line, " "), d = StringFromList(6, line, " ")
	Variable err
	strswitch(what)
		case "value":
			AFM_SetValue(a, str2num(b))
			return AFM_V3Reply(seq, "ok", "value " + a + " " + AFM_V3Num(AFM_GetValue(a)))
		case "loop":
			return AFM_V3SetLoop(seq, line)
		case "ramp":
			err = AFM_RampOutput(a, str2num(b), str2num(c), str2num(d))
			return AFM_V3Reply(seq, "ok", "ramp " + a + " " + b + " " + c + " " + d + " err=" + num2istr(err))
		case "loopstop":
			err = AFM_StopLoop(str2num(a))
			if(str2num(a) == 2)
				String/G root:AFM_v3Loop2 = "none"
			endif
			return AFM_V3Reply(seq, "ok", "loopstop " + a + " err=" + num2istr(err))
		case "mode":
			AFM_SetMode(a)
			return AFM_V3Reply(seq, "ok", "mode " + a)
		case "showtip":
			AFM_ShowTip(cmpstr(a, "ON", 1) == 0)
			return AFM_V3Reply(seq, "ok", "")
	endswitch
	return AFM_V3Reply(seq, "err", "bad set " + what)
End

// SET <seq> loop <n> <in> <sp> <p> <i> <out> [<min> <max>]: SERVOLOOP, -10/150 when absent.
Function AFM_V3SetLoop(seq, line)
	String seq, line
	String n = StringFromList(3, line, " "), inCh = StringFromList(4, line, " ")
	String sp = StringFromList(5, line, " "), pg = StringFromList(6, line, " ")
	String ig = StringFromList(7, line, " "), outCh = StringFromList(8, line, " ")
	String lo = "-10", hi = "150"
	if(ItemsInList(line, " ") >= 11)
		lo = StringFromList(9, line, " ")
		hi = StringFromList(10, line, " ")
	endif
	Variable err = AFM_SetupServoLoop(str2num(n), inCh, str2num(sp), str2num(pg), str2num(ig), outCh, str2num(lo), str2num(hi))
	return AFM_V3Reply(seq, "ok", "loop " + n + " " + inCh + " " + sp + " " + pg + " " + ig + " " + outCh + " " + lo + " " + hi + " err=" + num2istr(err))
End

// What Igor tracks from the loop commands it ran (contract 5b): loop 2 at any setpoint but -10
// puts the tip down (engage, set-down); -10 lifts it (lift, withdraw). td_Stop alone changes nothing.
Function AFM_V3NoteLoop(loop, setpoint, outMin, outMax)
	Variable loop, setpoint, outMin, outMax
	if(loop != 2)
		return 0
	endif
	AFM_V3Init()
	String/G root:AFM_v3Loop2 = AFM_V3Num(setpoint) + "," + AFM_V3Num(outMin) + "," + AFM_V3Num(outMax)
	Variable/G root:AFM_v3TipDown = (setpoint != -10)
	return 0
End

// CONFIRMSAFE: Igor reads Height itself; below kRetractedHeightV the make-safe is confirmed.
Function AFM_V3ConfirmSafe(seq)
	String seq
	Variable h = td_ReadValue("Height")
	if(h < kRetractedHeightV)
		AFM_AbortConfirmed()
		Variable/G root:AFM_v3TipDown = 0
		AFM_V3Reply(seq, "ok", "height=" + AFM_V3Num(h))
		return AFM_V3Report("-", "confirmed " + AFM_V3Num(h), "sock")
	endif
	return AFM_V3Reply(seq, "err", "height=" + AFM_V3Num(h))
End

// VENDORWITHDRAW: the vendor Withdraw once per make-safe, then Igor's fault (contract 5a).
Function AFM_V3VendorWithdraw(seq)
	String seq
	NVAR vwDone = root:AFM_v3VendorDone
	if(vwDone)
		return AFM_V3Reply(seq, "ok", "already")
	endif
	DoScanFunc("StopEngageButton")                       // the vendor Withdraw; it also kills the loops
	vwDone = 1
	Variable/G root:AFM_abortFault = 1
	String/G root:AFM_v3Loop2 = "none"
	return AFM_V3Reply(seq, "ok", "ran")
End

// CLEARFAULT <seq>: host only; both flags to 0 with no Height check, so PLAY is accepted again
// (Joe, 2026-10-27: the operator takes responsibility; contract section 12).
Function AFM_V3ClearFault(seq)
	String seq
	if(AFM_V3NotHost(seq))
		return 0
	endif
	AFM_ClearFault()
	Variable/G root:AFM_v3VendorDone = 0
	return AFM_V3Reply(seq, "ok", "fault=0")
End

// ---- make-safe hooks: AFM_SafePark and AFM_AbortBackup call these after the withdraw; they
// command nothing ----

// After the re-arm: the run in flight ends as aborted (contract 4 rule 9), then REPORT
// madesafe <why>, stamped with the route the make-safe came by.
Function AFM_V3ParkEnds()
	AFM_V3Init()
	SVAR run = root:AFM_v3Run
	SVAR route = root:AFM_v3Route
	SVAR why = root:AFM_v3Why
	NVAR/Z stopSeen = root:AFM_v3StopSeen
	String parked = "-"
	if(!NVAR_Exists(stopSeen) || !stopSeen)              // the flag was clear, so td_Stop ran now:
		Variable/G root:AFM_v3VendorDone = 0             //   a new make-safe (contract 5a step 5)
	endif
	Variable/G root:AFM_v3StopSeen = 1
	String/G root:AFM_v3Loop2 = "-10,0,150"              // the re-arm AFM_SafePark just ran
	if(strlen(run) > 0)
		parked = run
		Variable/G root:AFM_v3Hold = 0
		AFM_V3RunEnds("aborted")
	endif
	return AFM_V3Report(parked, "madesafe " + why, route)
End

// AFM_AbortBackup ran: REPORT backup <h> cleared|withdraw.
Function AFM_V3BackupRan(h, withdrew)
	Variable h, withdrew
	AFM_V3Init()
	if(withdrew)
		Variable/G root:AFM_v3VendorDone = 1
		String/G root:AFM_v3Loop2 = "none"
		return AFM_V3Report("-", "backup " + AFM_V3Num(h) + " withdraw", "-")
	endif
	Variable/G root:AFM_v3TipDown = 0
	Variable/G root:AFM_v3StopSeen = 0                   // the backup cleared root:AFM_abortStopped
	return AFM_V3Report("-", "backup " + AFM_V3Num(h) + " cleared", "-")
End

// AFMv3guard, 20 Hz while the host link is dead (contract 5b). Once kSockDeadParkTicks have
// passed and no run is playing (a run finishes on its own), it parks a tip left down, then
// stops: one check per dead link.
Function AFM_V3GuardTick(s)
	STRUCT WMBackgroundStruct &s
	NVAR/Z dead = root:AFM_v3DeadSince
	SVAR/Z run = root:AFM_v3Run
	NVAR/Z tipDown = root:AFM_v3TipDown
	if(!NVAR_Exists(dead) || !SVAR_Exists(run) || !NVAR_Exists(tipDown))
		return 1
	endif
	if(dead < 0)
		return 1                                         // a HELLO brought the host back
	endif
	if(ticks - dead < kSockDeadParkTicks || strlen(run) > 0)
		return 0
	endif
	if(tipDown)
		String/G root:AFM_v3Route = "-"
		String/G root:AFM_v3Why = "socket"
		AFM_SafePark()
		String/G root:AFM_v3Route = "com"
		String/G root:AFM_v3Why = "stop"
	endif
	return 1
End

// ---- abort panel: the operator's escape when the afm-litho service is dead ----
// Igor's own Withdraw cannot be relied on here: once afm-litho has engaged the tip, Igor's
// engage state does not know it. The button runs the one abort, AFM_SafePark, and nothing else.
Menu "Macros"
	"AFM Abort panel", AFM_ShowAbortPanel()
End

Constant kAbortPanelTicks = 360                      // the status check, 6 s at 60 ticks/s

// A small floating panel (NewPanel /FLT, IgorMan V-500) with one ABORT button and a status line.
// Opening it again brings the existing panel to the front.
Function AFM_ShowAbortPanel()
	DoWindow/F AFMAbort
	if(V_flag)
		return 0
	endif
	String/G root:AFM_abortPanelMsg = "Ready. ABORT runs AFM_SafePark()."
	NewPanel/K=1/FLT/N=AFMAbort/W=(100,100,380,200) as "AFM Abort"
	// appearance=os9 so Windows draws the red fill (native buttons may ignore fColor).
	Button abortBtn, win=AFMAbort, pos={10,10}, size={260,50}, appearance={os9}, fColor=(65535,0,0), fSize=16, fStyle=1, title="ABORT (SafePark)", proc=AFM_AbortButtonProc
	TitleBox abortMsg, win=AFMAbort, pos={10,68}, size={260,20}, frame=0, fixedSize=1, variable=root:AFM_abortPanelMsg
	SetActiveSubwindow _endfloat_                        // required after NewPanel/FLT (V-500)
	return 0
End

// Mouse up (eventCode 2, V-45) runs the abort. Height read straight after shows the retract
// starting; AFMabortPanel rewrites the line after 6 s, once Igor's 5 s backup has run.
Function AFM_AbortButtonProc(ba) : ButtonControl
	STRUCT WMButtonAction &ba
	if(ba.eventCode != 2)
		return 0
	endif
	AFM_SafePark()
	Variable h = td_ReadValue("Height")
	String/G root:AFM_abortPanelMsg = "SafePark sent; Height " + num2str(h) + " V (retracting)"
	CtrlNamedBackground AFMabortPanel, stop
	CtrlNamedBackground AFMabortPanel, period=(kAbortPanelTicks), proc=AFM_AbortPanelTick
	CtrlNamedBackground AFMabortPanel, start=(ticks + kAbortPanelTicks)
	return 0
End

// One shot, 6 s after the press: Height and the two abort flags. Reads only; returns 1 (stop).
Function AFM_AbortPanelTick(s)
	STRUCT WMBackgroundStruct &s
	Variable h = td_ReadValue("Height")
	NVAR/Z stopped = root:AFM_abortStopped
	NVAR/Z fault = root:AFM_abortFault
	String st = "?", ft = "?"
	if(NVAR_Exists(stopped))
		st = num2str(stopped)
	endif
	if(NVAR_Exists(fault))
		ft = num2str(fault)
	endif
	String head = "CHECK Z"
	if(h < kRetractedHeightV)
		head = "Retracted"
	endif
	String/G root:AFM_abortPanelMsg = head + ": Height " + num2str(h) + " V; stopped " + st + ", fault " + ft
	return 1
End

// ---- abort: when td_Stop must run ----

// root:AFM_passArmed is 1 from the moment a pass arms an out-wave or fires an event until the
// next td_Stop. A stale root:AFM_abortStopped (a park never confirmed, a fault clear that never
// reached Igor) must not stop an abort from stopping that pass: its waves would keep playing,
// and the bias wave overrides Output.A = 0 (Cypher rig, 2026-10-28).
Function AFM_PassMayPlay()
	NVAR/Z armed = root:AFM_passArmed
	return NVAR_Exists(armed) && armed
End

// td_Stop just ran: the once-per-abort flag is set, nothing plays any more, and wire v3 counts
// a new make-safe (AFM_V3ParkEnds then resets its vendor-withdraw latch).
Function AFM_StopRan()
	Variable/G root:AFM_abortStopped = 1
	Variable/G root:AFM_passArmed = 0
	Variable/G root:AFM_v3StopSeen = 0
	return 0
End
