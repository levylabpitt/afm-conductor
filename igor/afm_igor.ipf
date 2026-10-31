#pragma rtGlobals=1
#pragma IgorVersion=6.0
#pragma version=0.5

// afm-litho Asylum-specific telemetry + GUI startup values. Requires the
// MFP3D/ARC XOP + Asylum procedures (td_ReadValue, GV) -- it will NOT compile
// without them, so it is kept OUT of the generic afm_litho.ipf. This is the
// "info out" the browser needs: live meters (AFM_State) and, on connect, every
// value a control initialises from (AFM_MacroStartup).
//
// All getters return a ";"-delimited "key=value" string (SI: metres/volts/Hz),
// so one COM readback -- fprintf 0,"%s",AFM_MacroStartup() -- fills the panel.
// The GV/td_ names come from the validated map in twin/igor_commands.py
// (SCAN_PARAM_GLOBALS) and physics.Instrument.from_note. Reads happen Igor-side;
// off-instrument they return 0/NaN.

// ---- live meters (polled continuously) -------------------------------------
//   x,y      tip position (XSensor/YSensor)   z    height
//   defl     deflection (contact error)        amp  amplitude (AC)
//   sum      detector sum (laser health)       bias tip-drive DAC (Output.A)
//   scanning ARSysScanBusy (nonzero while scanning)
Function/S AFM_State()
	String s = ""
	s += "x=" + num2str(td_ReadValue("XSensor")) + ";"
	s += "y=" + num2str(td_ReadValue("YSensor")) + ";"
	s += "z=" + num2str(td_ReadValue("Height")) + ";"
	s += "defl=" + num2str(td_ReadValue("Deflection")) + ";"
	s += "amp=" + num2str(td_ReadValue("Amplitude")) + ";"
	s += "sum=" + num2str(td_ReadValue("DetectorSum")) + ";"
	s += "bias=" + num2str(td_ReadValue("Output.A")) + ";"     // DAC volts; x ampGain = tip
	s += "scanning=" + num2str(GV("ARSysScanBusy")) + ";"
	return s
End

// ---- scan panel: size / rate / pixels / lines / angle / mode ---------------
// Feeds the #size, #rate, #pixels, #lines, #angle controls and the mode popup.
// size is metres here; the GUI shows microns. mode is the index (0=contact,
// 1=ac), which the service maps with MODE_FROM_INDEX.
Function/S AFM_GetScanParams()
	String s = ""
	s += "size_m=" + num2str(GV("ScanSize")) + ";"
	s += "line_rate_hz=" + num2str(GV("ScanRate")) + ";"
	s += "pixels=" + num2str(GV("ScanPoints")) + ";"
	s += "lines=" + num2str(GV("ScanLines")) + ";"
	s += "angle_deg=" + num2str(GV("ScanAngle")) + ";"
	s += "mode=" + num2str(GV("ImagingMode")) + ";"
	return s
End

// ---- setpoint: a different global per feedback mode ------------------------
// Feeds the #setpoint control. Reads the live mode, then the matching global
// (DeflectionSetpointVolts in contact, AmplitudeSetpointVolts in AC).
Function/S AFM_GetSetpoint()
	String s = ""
	Variable mode = GV("ImagingMode")
	if(mode == 1)
		s += "setpoint=" + num2str(GV("AmplitudeSetpointVolts")) + ";"
		s += "setpoint_mode=ac;"
	else
		s += "setpoint=" + num2str(GV("DeflectionSetpointVolts")) + ";"
		s += "setpoint_mode=contact;"
	endif
	return s
End

// ---- scanner limits (slider maxima, surface extent) ------------------------
Function/S AFM_GetScanner()
	String s = ""
	s += "max_scan_m=" + num2str(GV("MaxScanSize")) + ";"       // full XY range, metres
	s += "input_range=" + num2str(GV("InputRange")) + ";"
	return s
End

// ---- feedback gains (the gain controls) ------------------------------------
Function/S AFM_GetGains()
	String s = ""
	s += "igain=" + num2str(GV("IntegralGain")) + ";"
	s += "pgain=" + num2str(GV("ProportionalGain")) + ";"
	return s
End

// ---- cantilever + drive (tune panel / displayed lever params) --------------
Function/S AFM_GetCantilever()
	String s = ""
	s += "f0=" + num2str(GV("ThermalFrequency")) + ";"
	s += "q=" + num2str(GV("ThermalQ")) + ";"
	s += "k=" + num2str(GV("SpringConstant")) + ";"
	s += "invols=" + num2str(GV("InvOLS")) + ";"
	s += "amp_invols=" + num2str(GV("AmpInvOLS")) + ";"
	s += "drive_hz=" + num2str(GV("DriveFrequency")) + ";"
	s += "drive_v=" + num2str(GV("DriveAmplitude")) + ";"
	return s
End

// ---- LVDT calibration (the litho um<->volts conversion uses these live) -----
Function/S AFM_GetCal()
	// %.15g, NOT num2str (~5-6 sig figs): the LVDT sensitivity/offset scale tip
	// position directly, so the driver's micron->volt conversion needs the full
	// calibration precision, not a truncated readback.
	String s = "", t
	sprintf t, "x_lvdt_sens=%.15g;", GV("XLVDTSens")
	s += t
	sprintf t, "y_lvdt_sens=%.15g;", GV("YLVDTSens")
	s += t
	sprintf t, "z_lvdt_sens=%.15g;", GV("ZLVDTSens")
	s += t
	sprintf t, "x_lvdt_offset=%.15g;", GV("XLVDTOffset")
	s += t
	sprintf t, "y_lvdt_offset=%.15g;", GV("YLVDTOffset")
	s += t
	sprintf t, "z_lvdt_offset=%.15g;", GV("ZLVDTOffset")
	s += t
	return s
End

// ---- XY closed-loop gains, computed the vendor's way from calibration ----------
// ScanMaster / HardWare derive the XY loop gains from stored calibration, NOT from a
// prior scan. The SetVar the operator edits holds the LOG10 of the gain, and the
// applied value is 10^value * sign(PiezoSens):
//   HardWare.ipf GetFeedbackParms:  FB.Igain = 10^MVW[%XIGain]*sign(MVW[%XPiezoSens])
//   ScanMaster SpecialGainFunc/XYGainsFunc: result = 10^varNum; result *= sign(...)
// XIGain/YIGain/XPiezoSens/YPiezoSens are instrument parameters loaded at startup
// (Initialization.ipf cInstParmList), so these are valid on first contact with NO
// scan -- which is the whole point of computing rather than reading the live loop.
// P is normally unused (XPGain defaults to -inf -> 10^-inf = 0, matching a P=0
// readback); S (the "secret" double-integral) is NOT carried over the bridge (P/I
// only). numtype guards a missing/degenerate global to 0 rather than NaN.
Function/S AFM_GetXYLoopGains()
	String s = "", t
	Variable sx = sign(GV("XPiezoSens")), sy = sign(GV("YPiezoSens"))
	Variable xp = 10^GV("XPGain") * sx, xi = 10^GV("XIGain") * sx
	Variable yp = 10^GV("YPGain") * sy, yi = 10^GV("YIGain") * sy
	if(numtype(xp) != 0)
		xp = 0
	endif
	if(numtype(yp) != 0)
		yp = 0
	endif
	if(numtype(xi) != 0)
		xi = 0
	endif
	if(numtype(yi) != 0)
		yi = 0
	endif
	sprintf t, "x_pgain=%.15g;", xp
	s += t
	sprintf t, "x_igain=%.15g;", xi
	s += t
	sprintf t, "y_pgain=%.15g;", yp
	s += t
	sprintf t, "y_igain=%.15g;", yi
	s += t
	return s
End

// ---- everything the GUI needs on connect, in one call ----------------------
// The service calls this once on startup (COM readback, or over SOCKIT if it
// ever outgrows the ~1000-char history buffer) and populates every control +
// meter from it. Composed of the getters above so a single control can also be
// refreshed on its own.
Function/S AFM_MacroStartup()
	String s = ""
	s += AFM_GetScanParams()
	s += AFM_GetSetpoint()
	s += AFM_GetScanner()
	s += AFM_GetGains()
	s += AFM_GetCantilever()
	s += AFM_GetCal()
	s += AFM_GetXYLoopGains()
	s += AFM_State()
	// Platform id so the HOST can gate the Cypher-vs-MFP-3D differences (the Z loop output
	// is "Height" on both platforms). cMicroscopeCypher == 4 (HardWare.ipf:10); the host
	// compares against that. See hardware_arc.is_cypher.
	s += "microscope_id=" + num2str(GV("MicroscopeID")) + ";"
	return s
End
