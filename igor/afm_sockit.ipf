#pragma rtGlobals=1
#pragma IgorVersion=6.0
#pragma version=0.8

// afm-litho SOCKIT push wrappers -- OPTIONAL. Requires the SOCKIT XOP loaded
// AND the Asylum MFP3D software (AFM_StreamTick calls GV("ARSysScanBusy"), an
// Asylum global) -- so it will NOT compile at Igor startup; it must live in User
// Procedures and load AFTER the software, like afm_igor/afm_litho_run. Kept
// separate from the generic afm_litho.ipf (which has no XOP/Asylum dependency);
// SOCKIT is the opt-in higher-bandwidth PUSH transport (Igor -> Python).
//
// Model: Python runs a TCP server; Igor connects as a client and PUSHES scan
// lines to it. Control (start scan, params) stays on COM. Igor never needs to
// RECEIVE over the socket, so we use the send-only path -- no receive timeout.
//
// Confirmed SOCKIT calls (from the XOP headers, on Igor 6.38):
//   SOCKITopenconnectionF(ipStr, port, timeout) -> connection id   (function)
//   SOCKITsendmsgF(sockID, message)             -> bytes sent       (function, send-only)
//   SOCKITcloseConnection(sockID)               -> status           (function -- NOT an operation)

// Open a TCP connection to host:port. Returns the connection id (>= 0), or a
// negative value on failure.
Function AFM_SockConnect(host, port)
	String host
	Variable port

	return SOCKITopenconnectionF(host, port, 5)
End

// Push one row (scan line) of a 2D wave to the socket as a newline-framed text
// message: "L <row> <npix> v;v;...;v\n". Python frames on '\n' and feeds the
// browser's existing per-line renderer. Sent in ONE socket message -- no COM
// results-buffer limit, no per-chunk round-trips. Returns bytes sent, or -1 if
// the wave is missing.
Function AFM_SendRow(sock, wavePath, row)
	Variable sock, row
	String wavePath

	WAVE/Z w = $wavePath
	if(!WaveExists(w))
		return -1
	endif

	Variable npix = DimSize(w, 0), i
	String s, tmp
	sprintf s, "L %d %d ", row, npix
	for(i = 0; i < npix; i += 1)
		sprintf tmp, "%.9g;", w[i][row]
		s += tmp
	endfor
	s += "\n"

	return SOCKITsendmsgF(sock, s)
End

// Close a connection opened by AFM_SockConnect. SOCKITcloseConnection is a
// FUNCTION here (registered in the XOP's function table), so it takes parens.
Function AFM_SockClose(sock)
	Variable sock

	return SOCKITcloseConnection(sock)
End

// --- Autonomous line streaming (the observer) --------------------------------
// A background task pushes each newly-completed line of an image wave to Python
// as a scan fills it -- no Python polling. Point it at Asylum's own scan output:
//   AFM_StreamStart("127.0.0.1", 9193,
//                   "root:Packages:MFP3D:Main:HeightImage0",
//                   "root:Packages:MFP3D:Main:LineCount")
// then start an Asylum scan (DoScanFunc over COM). Config lives in root: globals
// so the background proc can reach it.
//
// The tick reads ONLY the signals Asylum exposes -- LineCount, ScanLines,
// ScanDown, ARSysScanBusy -- and emits three frame types to Python (see
// docs/SCAN_STREAMING.md and the Python mirror hardware/igor_afm_emulator.py):
//   S <rows> <cols>   frame start (on a LineCount reset)
//   L <row> <npix> .. one completed line, in acquisition order
//   D                 frame end (finished OR aborted)
// It is DIRECTION-AWARE (up-scan counts LineCount 0->ScanLines and sends rows
// 0..; down-scan counts ScanLines-1->-1 and sends rows ScanLines-1..) and treats
// a LineCount reset as a new-frame boundary, so continuous mode streams each
// frame with its own S/.../D.

Function AFM_StreamStart(host, port, wavePath, lineCountPath)
	String host, wavePath, lineCountPath
	Variable port

	Variable/G root:AFM_streamSock = SOCKITopenconnectionF(host, port, 5)
	NVAR sock = root:AFM_streamSock
	if(sock < 0)
		return -1
	endif
	String/G root:AFM_streamWave = wavePath
	String/G root:AFM_streamLC = lineCountPath
	Variable/G root:AFM_streamSent = 0        // completed-count streamed this frame
	Variable/G root:AFM_streamActive = 0      // are we inside a frame (S sent)?

	CtrlNamedBackground AFMstream, period=3, proc=AFM_StreamTick
	CtrlNamedBackground AFMstream, start
	return sock
End

// Fires every ~50 ms: emit a frame start on a LineCount reset, every line
// completed since the last tick, and a frame end on finish or abort. Mirror of
// AFMStreamTick.poll() in hardware/igor_afm_emulator.py.
Function AFM_StreamTick(s)
	STRUCT WMBackgroundStruct &s

	NVAR sock = root:AFM_streamSock
	SVAR wavePath = root:AFM_streamWave
	SVAR lcPath = root:AFM_streamLC
	NVAR sent = root:AFM_streamSent
	NVAR active = root:AFM_streamActive

	NVAR/Z lc = $lcPath
	if(!NVAR_Exists(lc))
		return 0				// scan hasn't created the counter yet
	endif
	WAVE/Z img = $wavePath
	if(!WaveExists(img))
		return 0
	endif

	// the only inputs real Igor has (ScanMaster.ipf uses these same globals)
	WAVE MVW = root:Packages:MFP3D:Main:Variables:MasterVariablesWave
	Variable scanLines = MVW[%ScanLines][0]
	Variable scanDown = MVW[%ScanDown][0]
	Variable pixels = DimSize(img, 0)
	Variable busy = GV("ARSysScanBusy")

	// LineCount -> direction-independent completed-count
	Variable completed
	if(scanDown)
		completed = scanLines - 1 - lc
	else
		completed = lc
	endif

	// new frame: busy, and completed reset below what we've already streamed
	String hdr
	if(busy && (!active || completed < sent))
		sprintf hdr, "S %d %d\n", scanLines, pixels
		AFM_SendStr(sock, hdr)
		active = 1
		sent = 0
	endif

	// stream every newly-completed line, in acquisition order
	Variable r
	do
		if(!active || sent >= completed)
			break
		endif
		if(scanDown)
			r = scanLines - 1 - sent
		else
			r = sent
		endif
		AFM_SendRow(sock, wavePath, r)
		sent += 1
	while(1)

	// frame ended: finished (all lines) or aborted (busy dropped early)
	if(active && (completed >= scanLines || !busy))
		AFM_SendStr(sock, "D\n")
		active = 0
	endif
	return 0
End

Function AFM_StreamStop()
	CtrlNamedBackground AFMstream, stop
	NVAR/Z sock = root:AFM_streamSock
	if(NVAR_Exists(sock))
		AFM_SockClose(sock)
	endif
	return 0
End

// --- Stream a REAL scan (the data-out path) ----------------------------------
// The RT scan images Asylum fills live at root:Packages:MFP3D:Main:<Channel>Image0
// (ScanMaster.ipf:3199), and scan progress is root:Packages:MFP3D:Main:LineCount
// (ScanMaster.ipf:1779). So streaming the Height channel of a real scan is just
// AFM_StreamStart pointed at those. These wrap that so the operator/service does
// ONE call -- no long wave paths -- and any scan that runs streams its Height
// lines out to the Python receiver (hardware/sockit_receiver.py) as they fill.

StrConstant kHeightImage = "root:Packages:MFP3D:Main:HeightImage0"
StrConstant kLineCount   = "root:Packages:MFP3D:Main:LineCount"

// Start live-streaming the Height channel of the real scan. Call this (once) any
// time before/at scan start; AFM_StreamTick pushes each line as LineCount
// advances. Returns the socket id (negative if the receiver isn't listening).
Function AFM_StreamScanStart(host, port)
	String host
	Variable port

	return AFM_StreamStart(host, port, kHeightImage, kLineCount)
End

// Stream a COMPLETED capture out in one shot: connect, push EVERY row of a 2D
// image wave (a finished scan, or one LoadWave'd from a real .ibw), then close.
// Use when the frame is already whole -- e.g. to verify the data-out + render
// path with real data before a live scan. `wavePath` is a 2D single-channel
// image (fast dim = pixels, slow dim = lines). Returns the row count, or -1.
Function AFM_StreamImageOut(host, port, wavePath)
	String host, wavePath
	Variable port

	WAVE/Z w = $wavePath
	if(!WaveExists(w))
		return -1
	endif
	Variable sock = AFM_SockConnect(host, port)
	if(sock < 0)
		return -1
	endif
	Variable nrows = DimSize(w, 1), r
	for(r = 0; r < nrows; r += 1)
		AFM_SendRow(sock, wavePath, r)
	endfor
	AFM_SockClose(sock)
	return nrows
End

// --- Throughput stress-test helpers ------------------------------------------
// The row string is built pixel-by-pixel with sprintf + string-append, and
// Igor's `s += tmp` can reallocate, so the per-row build cost may grow FASTER
// than linearly in the pixel count. These time the work IN Igor (stopMSTimer,
// microseconds) so the measurement isn't polluted by per-call COM overhead;
// scripts/stress_sockit.py drives them across a sweep of row widths.

// Build row 0's message `reps` times WITHOUT sending -- isolates the
// sprintf + string-append cost. Returns total microseconds for `reps` builds.
Function AFM_BenchBuild(wavePath, reps)
	String wavePath
	Variable reps

	WAVE/Z w = $wavePath
	if(!WaveExists(w))
		return -1
	endif
	Variable npix = DimSize(w, 0), i, r
	String s, tmp
	Variable t0 = stopMSTimer(-2)
	for(r = 0; r < reps; r += 1)
		sprintf s, "L %d %d ", 0, npix
		for(i = 0; i < npix; i += 1)
			sprintf tmp, "%.9g;", w[i][0]
			s += tmp
		endfor
		s += "\n"
	endfor
	return stopMSTimer(-2) - t0
End

// Build AND send row 0 `reps` times over an open socket -- build + socket +
// TCP flow control (the real sustained push cost). Returns total microseconds.
Function AFM_BenchSend(sock, wavePath, reps)
	Variable sock, reps
	String wavePath

	Variable r
	Variable t0 = stopMSTimer(-2)
	for(r = 0; r < reps; r += 1)
		AFM_SendRow(sock, wavePath, 0)
	endfor
	return stopMSTimer(-2) - t0
End

// --- Two-way: RECEIVING from Python on the same socket -----------------------
// The F-opened connection (AFM_SockConnect -> SOCKITopenconnectionF) is
// registered with SOCKIT's background reader thread (addWorker/FD_SET in the
// XOP), so bytes Python sends are buffered inside the XOP as they arrive.
// SOCKITpeek(sock) returns everything buffered since the last call and clears
// it, so Igor can RECEIVE with NO change to how the socket was opened. This is
// the piece the earlier operation-form attempt got wrong: it opened with the
// F-function but tried to receive with the OPERATION-form send/recv (which
// wants an operation-opened wavebufferinfo). The F-form receive is SOCKITpeek.
// Control can now flow Python -> Igor over the same link, not only Igor -> Py.

// Send an arbitrary text message (not a wave row). Returns bytes sent.
Function AFM_SendStr(sock, msg)
	Variable sock
	String msg

	return SOCKITsendmsgF(sock, msg)
End

// Drain whatever Python has sent so far -- may be "", a partial message, or
// several messages concatenated. Returns the raw bytes as a string.
Function/S AFM_Peek(sock)
	Variable sock

	return SOCKITpeek(sock)
End

// Wait up to timeoutS seconds for at least one newline-framed message,
// accumulating partial reads; returns everything received ("" on timeout).
// SOCKIT's reader runs on its own thread, so the spin still lets bytes arrive.
Function/S AFM_RecvWait(sock, timeoutS)
	Variable sock, timeoutS

	String acc = "", chunk
	Variable t0 = stopMSTimer(-2)
	do
		chunk = SOCKITpeek(sock)
		if(strlen(chunk) > 0)
			acc += chunk
			if(strsearch(acc, "\n", 0) >= 0)
				return acc
			endif
		endif
	while((stopMSTimer(-2) - t0) < timeoutS * 1e6)
	return acc
End

// --- Two-way round-trip stress test ------------------------------------------
// Measures the request/reply LATENCY of the link: send an nbyte message, spin
// on SOCKITpeek until the echo's newline returns, `reps` times. Timed in Igor
// (stopMSTimer), so no per-call COM overhead. Python runs a byte-echo server
// (scripts/stress_sockit_twoway.py). The message is built here (not passed over
// COM) so large sizes don't hit the ~400-char command limit. Returns total
// microseconds for `reps` round-trips, or -1 if a reply stalled (>5 s).
Function AFM_BenchRoundtrip(sock, nbytes, reps)
	Variable sock, nbytes, reps

	Variable n = max(1, nbytes)
	String msg = PadString("", n - 1, 0x58) + "\n"   // n bytes: 'X'... + newline
	String acc, chunk
	Variable r, dummy, tstart
	Variable t0 = stopMSTimer(-2)
	for(r = 0; r < reps; r += 1)
		dummy = SOCKITsendmsgF(sock, msg)
		acc = ""
		tstart = stopMSTimer(-2)
		do
			chunk = SOCKITpeek(sock)
			if(strlen(chunk) > 0)
				acc += chunk
				if(strsearch(acc, "\n", 0) >= 0)
					break
				endif
			endif
			if((stopMSTimer(-2) - tstart) > 5e6)     // 5 s stall guard
				return -1
			endif
		while(1)
	endfor
	return stopMSTimer(-2) - t0
End
