"""Documented MFP-3D / ARC controller error codes (ae_*), for traceable errors.

Every td_/ir_ function returns one of these; 0 (ae_None) is success. This is a
different channel from Igor's own command-execution error: Execute2's errorCode
only catches a bad command STRING, so an ARC function that runs cleanly but
rejects its arguments returns a nonzero ae_ code that Igor-level success hides.
Decoding it turns "the write failed with 10" into "ae_BadAlias: Requested alias
is bad or does not exist" -- which is what makes a failure on the instrument
traceable to a documented cause.

Source: docs/reference/asylum-td-command-reference.txt, section 3, "MFP-3D
CONTROLLER ERROR CODE REFERENCE" (the constants live in xARXOPDefs.ipf). Kept
verbatim so a code seen on the bench maps to exactly the vendor's wording.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArcCode:
    code: int
    name: str
    category: str
    description: str

    def __str__(self) -> str:
        return f"{self.name} ({self.code}, {self.category}): {self.description}"


_TABLE: dict[int, ArcCode] = {}


def _add(category: str, rows: list[tuple[int, str, str]]) -> None:
    for code, name, desc in rows:
        _TABLE[code] = ArcCode(code, name, category, desc)


_add("Basic", [
    (0, "ae_None", "No error."),
    (1, "ae_NotImplementedYet", "Feature not yet implemented."),
    (2, "ae_File", "Problem reading or writing a file."),
])
_add("ParmLib", [
    (3, "ae_BadParm", "Requested parameter is bad or does not exist."),
    (4, "ae_BadParmGroup", "Requested parameter group is bad or does not exist."),
    (5, "ae_BadValue", "Value sent to parameter is bad."),
    (6, "ae_BadParmXML", "Error in file used to recreate parameter lists."),
    (7, "ae_DWARFInfoNotFound",
     "Variable missing or misnamed in Smart Start or DSP Director module."),
    (8, "ae_DWARFInfoBad",
     "Bad variable information in Smart Start or DSP Director module."),
    (9, "ae_libelfDLL", "Could not load libelf.dll."),
    (10, "ae_BadAlias", "Requested alias is bad or does not exist."),
    (11, "ae_BadProperty", "Property used in wrong context, or does not exist."),
    (12, "ae_BadFunction", "Function used in wrong context, or does not exist."),
])
_add("More Basic", [
    (20, "ae_OutOfRange", "One of the parameters exceeds the range for the function."),
])
_add("Igor-Specific", [
    (100, "ae_WaveInUse", "Wave is in use elsewhere."),
    (101, "ae_BadWaveType", "Wave type was not expected."),
    (102, "ae_BadWaveLength", "Wave length was not expected."),
    (103, "ae_CannotUseSameWave", "Same wave cannot do two different things."),
    (104, "ae_MismatchedWaveLength", "One wave's length does not match the other."),
])
_add("MFP3D XOP", [
    (1101, "ae_NotConnectedYet", "Not connected to a controller yet."),
    (1102, "ae_USB", "USB problem; DSP may have hung."),
    (1103, "ae_SSB", "SmartStart Bus problem; device may have hung."),
    (1104, "ae_OneWire", "OneWire Bus problem; device may have hung."),
    (1105, "ae_BadDevice", "Requested device is bad or does not exist."),
    (1106, "ae_BlankDevice", "Device is blank or needs reprogramming."),
    (1107, "ae_FirmwareFailure", "Problem with device firmware or firmware upload."),
    (1108, "ae_VersionMismatch", "Device code does not match code found by XOP."),
    (1109, "ae_JumperSet", "Device has programming jumper set."),
    (1110, "ae_Verify", "Verification during device programming failed."),
    (1111, "ae_FlashNotValid", "Device program or infoblock has problems."),
    (1113, "ae_BadBankNumber", "Bank number was not expected."),
    (1114, "ae_BadInterpDecimFactor",
     "Interpolation or decimation factor was not expected."),
    (1115, "ae_BadEventName", "Event name was not expected."),
    (1116, "ae_BadEventTriggerName", "Event trigger name was not expected."),
    (1117, "ae_PIDSLoopNotConfigured", "Appropriate PIDSLoop must be configured first."),
    (1118, "ae_ThermalMode", "Cannot do that in thermal mode."),
    (1119, "ae_ChannelNotConfigured",
     "Appropriate CTFC channel must be configured first."),
    (1120, "ae_ScanChannelNotConfigured",
     "Appropriate scan channel must be configured first."),
    (1121, "ae_Cypher", "Problem talking to Cypher."),
    (1122, "ae_ParmNotEditable",
     "Parameter cannot be changed using the current command/operation."),
    (1123, "ae_FastScanNotConfigured", "FastScan must be configured first."),
])
_add("ARStatusBar", [
    (2001, "ae_BadPosition", "No button or text at this position."),
    (2002, "ae_NotStatusText", "Operation only works on status text area."),
])
_add("General", [
    (40000, "ae_Generic", "Error with no specific code."),
    (40001, "ae_OS", "OS error with no alternate code."),
    (40002, "ae_NeedAdminPrivs", "Need administrator privileges."),
    (50000, "ae_AdditionalErrorCodes",
     "Application-defined error codes start at this value."),
])


def decode(code: int) -> ArcCode:
    """Map an ARC return code to its documented meaning.

    Codes >= 50000 are application-defined (ae_AdditionalErrorCodes); anything
    else not in the table is reported as undocumented rather than silently
    treated as fine, so a surprising code still reads as an error.
    """
    code = int(code)
    if code in _TABLE:
        return _TABLE[code]
    if code >= 50000:
        return ArcCode(code, "ae_ApplicationDefined", "General",
                       "Application-defined error code (>= 50000).")
    return ArcCode(code, "ae_Undocumented", "?",
                   "Nonzero code not present in the vendor error reference.")
