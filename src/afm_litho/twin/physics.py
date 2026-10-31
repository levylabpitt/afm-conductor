"""Cantilever + sample physics behind the software ARC.

Cartoon fidelity, honest interfaces: the ARC sees only voltages and returns
only voltages, exactly like the real controller. Positions are meters
internally; LVDT calibration (sensitivity, offset) converts to volts at the
boundary, matching Position = (V - Offset) * Sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..io.ibw import read_ibw


@dataclass
class Instrument:
    """One microscope. Two twins, one code path: MFP-3D and Cypher differ in
    scanner range, LVDT calibration, and controller channel names, and every
    one of those is declared in the .ibw metadata of any scan they wrote.
    """

    model: str = "MFP3D"
    microscope_id: int = 0
    xy_range_um: float = 90.0        # MFP-3D 90 um, Cypher 30 um
    x_lvdt_sens: float = 7.4643e-6   # m/V
    y_lvdt_sens: float = 7.0428e-6
    z_lvdt_sens: float = 1.3036e-6
    x_lvdt_offset: float = -0.63     # V
    y_lvdt_offset: float = 0.07
    z_lvdt_offset: float = 0.0
    x_sensor: str = "Input.X"        # controller channel names
    y_sensor: str = "Input.Y"
    z_sensor: str = "Input.Z"
    dual_lockin: bool = False        # Cypher: LockinA / LockinB
    fast_capture: bool = False       # Cypher: td_ReadCapture
    # piezo calibration
    x_piezo_sens: float = 4.5113e-7  # m/V
    y_piezo_sens: float = 4.5867e-7
    z_piezo_sens: float = 6.8805e-8
    # lever calibration (as the instrument last measured it)
    invols: float = 1.0e-7           # m/V, deflection
    amp_invols: float = 1.09e-7      # m/V, amplitude
    spring_constant: float = 1.0     # N/m
    thermal_f0: float = 65713.0      # Hz
    thermal_q: float = 20.0
    drive_frequency: float = 65713.0
    drive_amplitude: float = 0.689   # V
    # z feedback
    integral_gain: float = 12.0
    proportional_gain: float = 0.0
    input_range: float = 1.78        # V
    # Tip holder (ARTipHolderParms). The bias in the panel is the bias AT
    # THE TIP; the DAC is commanded with bias/tip_gain, so a high-voltage
    # holder reaches +-150 V from a +-10 V output. TipGain == 15 is the
    # backpack HV path in HardWare.ipf.
    tip_gain: float = 1.0
    # ORCA transimpedance amplifier: the c-AFM current path. Two gain
    # settings on a dual-gain holder, and the channel is signed
    # (cOrcaSign = -1).
    orca_gain: float = 1.0           # A/V
    orca_gain2: float = 1.0
    orca_sign: float = -1.0
    is_orca: bool = False

    @classmethod
    def from_note(cls, note: dict) -> "Instrument":
        """Build the instrument from an Asylum wave note.

        Every scan the instruments write carries their own calibration: the
        scanner range (MaxScanSize), LVDT and piezo sensitivities with
        offsets, lever InvOLS / spring constant / thermal f0 and Q, drive
        settings, feedback gains, and the controller channel names. The twin
        configures itself from that, so loading a scan from any machine
        reproduces that machine.
        """
        def f(key, default):
            try:
                return float(note.get(key, default))
            except (TypeError, ValueError):
                return default

        model = str(note.get("MicroscopeModel", "MFP3D")).strip() or "MFP3D"
        cypher = model.lower().startswith("cypher")
        return cls(
            model=model,
            microscope_id=int(f("MicroscopeID", 0)),
            xy_range_um=f("MaxScanSize", 90e-6) * 1e6,
            x_lvdt_sens=f("XLVDTSens", 7.4643e-6),
            y_lvdt_sens=f("YLVDTSens", 7.0428e-6),
            z_lvdt_sens=f("ZLVDTSens", 1.3036e-6),
            x_lvdt_offset=f("XLVDTOffset", 0.0),
            y_lvdt_offset=f("YLVDTOffset", 0.0),
            z_lvdt_offset=f("ZLVDTOffset", 0.0),
            x_sensor=str(note.get("XSensor", "Input.X")),
            y_sensor=str(note.get("YSensor", "Input.Y")),
            z_sensor=str(note.get("ZSensor", "Input.Z")),
            dual_lockin=cypher,
            fast_capture=cypher,
            x_piezo_sens=f("XPiezoSens", 4.5113e-7),
            y_piezo_sens=f("YPiezoSens", 4.5867e-7),
            z_piezo_sens=f("ZPiezoSens", 6.8805e-8),
            invols=f("InvOLS", 1.0e-7),
            amp_invols=f("AmpInvOLS", 1.09e-7),
            spring_constant=f("SpringConstant", 1.0),
            thermal_f0=f("ThermalFrequency", 65713.0),
            thermal_q=f("ThermalQ", 20.0),
            drive_frequency=f("DriveFrequency", f("ThermalFrequency", 65713.0)),
            drive_amplitude=f("DriveAmplitude", 0.689),
            integral_gain=f("IntegralGain", 12.0),
            proportional_gain=f("ProportionalGain", 0.0),
            input_range=f("InputRange", 1.78),
            # tip holder and ORCA, when the scan recorded them
            tip_gain=f("TipGain", 1.0) or 1.0,
            orca_gain=f("OrcaGain", 1.0) or 1.0,
            orca_gain2=f("OrcaGain2", 1.0) or 1.0,
            is_orca=bool(note.get("OrcaGain")),
        )

    def bias_to_dac(self, bias_v: float) -> float:
        """Panel bias -> commanded DAC volts (Lithography.ipf: bias/TipGain)."""
        return bias_v / (self.tip_gain or 1.0)

    def dac_to_bias(self, dac_v: float) -> float:
        return dac_v * (self.tip_gain or 1.0)

    def current_from_volts(self, volts: float, gain2: bool = False) -> float:
        """ORCA output volts -> amps, signed (cOrcaSign = -1)."""
        g = self.orca_gain2 if gain2 else self.orca_gain
        return volts * (g or 1.0) * self.orca_sign

    def describe(self) -> str:
        return (f"{self.model} (#{self.microscope_id}) · "
                f"{self.xy_range_um:g} um scanner · "
                f"lever f0 {self.thermal_f0 / 1e3:.1f} kHz, Q {self.thermal_q:g}, "
                f"k {self.spring_constant:g} N/m")


# Dose written into the metal mask by seed_contacts, in the same units as a
# written wire's dose. Saturating, so a lead reaching a pad edge is
# galvanically connected.
METAL_SEED_DOSE = 25.0


def _enclosed_by_a_raised_rim(ndimage, raised: np.ndarray,
                              min_frac: float = 2e-4, max_frac: float = 0.05
                              ) -> np.ndarray | None:
    """Regions ringed by raised material whose insides are not raised.

    Returns None when the scan has no such pockets, so a sample that really
    is a field of mesas keeps working the way it did.

    IT RETURNS None ON EVERY SAMPLE IN data/samples, so nothing in this repo
    currently detects a pill and the contacts are still the gold. Measured:
    the raised mask has ZERO enclosed holes at the global threshold, and zero
    again with a local relief filter (h minus a Gaussian of sigma 4, 8 and 14
    px) at 1.5 and 2.5 robust sigma. Whatever makes a pill's boundary read as
    raised to a person is not a closed loop in these height maps at any
    threshold tried. Do not assume this function is doing anything until a
    scan is found where it fires.

    The closing seals a rim broken by a scan line or a bit of noise before
    the fill is attempted; without it one bad row drains the pill and it
    disappears. `max_frac` throws out the whole-frame "pocket" you get when
    the outermost features happen to ring the image.
    """
    r = max(2, int(round(0.004 * max(raised.shape))))
    g = np.mgrid[-r:r + 1, -r:r + 1]
    disc = np.hypot(g[0], g[1]) <= r
    sealed = ndimage.binary_closing(raised, disc)
    pockets = ndimage.binary_fill_holes(sealed) & ~sealed
    if not pockets.any():
        return None

    lab, n = ndimage.label(pockets)
    if not n:
        return None
    sizes = ndimage.sum(pockets, lab, range(1, n + 1))
    keep = np.zeros_like(pockets)
    total = pockets.size
    for k in range(n):
        if min_frac * total <= sizes[k] <= max_frac * total:
            keep |= lab == k + 1
    return keep if keep.any() else None


def _amplitude_layer(w) -> int | None:
    """Data-array index of the AmplitudeRetrace channel in an Asylum wave.

    Asylum records each stacked channel's kind in the note as
    "Channel{N}DataType" (1-indexed: Channel1 -> layer 0). We resolve amplitude
    by that name rather than by a guessed position, then fall back to the
    conventional stack order (Height, Amplitude, Phase, ZSensor) when the note
    is silent. Returns None for a wave that isn't a channel stack.
    """
    if w.data.ndim != 3 or w.data.shape[2] < 2:
        return None
    for i in range(1, w.data.shape[2] + 1):
        if w.note_dict.get(f"Channel{i}DataType", "").strip().lower() == "amplitude":
            return i - 1
    return 1  # Asylum's default HeightRetrace, AmplitudeRetrace, ... order


# Asylum saves Height = (Z drive V - 70) * (-ZPiezoSens) (ScanMaster.ipf GetRTScaling).
ASYLUM_HEIGHT_ZERO_V = 70.0


def _layer_by_type(w, kind: str) -> int | None:
    """Layer whose note says Channel{N}DataType == kind, or None."""
    if w.data.ndim != 3:
        return None
    for i in range(1, w.data.shape[2] + 1):
        if w.note_dict.get(f"Channel{i}DataType", "").strip().lower() == kind:
            return i - 1
    return None


def _note_scale(w, key: str) -> float | None:
    """A nonzero finite float from the wave note, else None."""
    try:
        value = float(w.note_dict.get(key))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value != 0.0 else None


def telemetry_maps_from_ibw(w) -> dict:
    """Raw Z-drive volts and amplitude volts per pixel, from the .ibw's own metadata.
    A map whose layer or scale is missing is left out."""
    maps = {}
    layer = _layer_by_type(w, "height")
    layer = 0 if layer is None and w.data.ndim == 3 else layer
    sens = _note_scale(w, "ZPiezoSens")
    if layer is not None and sens is not None:
        height_m = np.asarray(w.data[..., layer], dtype=np.float64)
        maps["zdrive_v"] = ASYLUM_HEIGHT_ZERO_V - height_m / sens
    layer = _amplitude_layer(w)
    inv_ols = _note_scale(w, "AmpInvOLS")
    if layer is not None and inv_ols is not None:
        maps["amp_v"] = np.asarray(w.data[..., layer], dtype=np.float64) / inv_ols
    return maps


def _pills_from_amplitude(ndimage, amp: np.ndarray,
                          sigma_edge: float = 1.5,
                          min_frac: float = 1.2e-3,
                          max_frac: float = 0.03) -> np.ndarray | None:
    """Interface contacts (PILLS) from the AC-mode amplitude channel.

    A pill's interior sits at the surrounding height and only its rim stands
    proud, so it is invisible to a height threshold (see the note in
    _enclosed_by_a_raised_rim). In amplitude it is not: the feedback lags on
    the rim's slope, so the rim reads as a bright closed ring while the flat
    interior -- and the flat background -- read low. A LEAD is thin and all
    edge, so it never encloses a pocket; only a pill does. So: threshold the
    amplitude gradient into a rim mask, bridge the gaps a scan line leaves,
    fill the enclosed pockets, and keep the ones whose SHAPE reads like a
    stadium. Returns the pill mask (height-map pixel grid) or None if the scan
    has no such pockets, so a caller can fall back to the height path.

    Validated on data/samples: SA40656B0002 -> all eight radial pills;
    the noisier SA40656B0000 -> four of eight, every one on a real pill.
    """
    a = ndimage.gaussian_filter(np.asarray(amp, dtype=float), 1.0)
    gmag = np.hypot(*np.gradient(a))
    med = np.median(gmag)
    mad = np.median(np.abs(gmag - med)) or (gmag.std() or 1.0)
    edge = (gmag - med) / (1.4826 * mad) > sigma_edge

    scale = max(amp.shape)
    rc = max(1, int(round(0.008 * scale)))
    g = np.mgrid[-rc:rc + 1, -rc:rc + 1]
    disc = np.hypot(g[0], g[1]) <= rc
    sealed = ndimage.binary_closing(edge, disc)
    pockets = ndimage.binary_fill_holes(sealed) & ~sealed
    pockets = ndimage.binary_dilation(pockets, disc)   # grow out to the rim centre

    lab, n = ndimage.label(pockets)
    if not n:
        return None
    keep = np.zeros_like(pockets)
    total = pockets.size
    for k in range(1, n + 1):
        comp = lab == k
        area = int(comp.sum())
        if not (min_frac * total <= area <= max_frac * total):
            continue
        ys, xs = np.nonzero(comp)
        cy, cx = ys.mean(), xs.mean()
        y, x = ys - cy, xs - cx
        m = len(xs)
        uxx, uyy, uxy = (x * x).sum() / m, (y * y).sum() / m, (x * y).sum() / m
        theta = 0.5 * np.arctan2(2 * uxy, uxx - uyy)
        c, s = np.cos(theta), np.sin(theta)
        length = float(np.ptp(x * c + y * s)) + 1.0     # along the major axis
        width = float(np.ptp(-x * s + y * c)) + 1.0      # along the minor axis
        # A stadium fills its oriented box and is not a hairline.
        if area / max(1.0, length * width) < 0.6 or length / max(1.0, width) > 4.0:
            continue
        keep |= comp
    return keep if keep.any() else None


def _deepest_point(ndimage, comp: np.ndarray) -> tuple[float, float]:
    """Index of the point furthest from any edge of `comp`."""
    # Padded, so the edge of the scan counts as an edge. Without it, a
    # feature running off the side of the frame has its furthest-from-an-edge
    # point ON the frame boundary -- which put contacts back on the mirror
    # seam by a different route.
    d = ndimage.distance_transform_edt(np.pad(comp, 1))
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    return float(i) - 1.0, float(j) - 1.0


def _mirror_images(v0: float, period_um: float, half_range_um: float
                   ) -> list[float]:
    """Where a feature at `v0` in the loaded scan repeats across the scanner.

    The exact inverse of the mirror tiling in Sample._map_coords, which folds
    a coordinate with `u = mod(v + p/2, 2p)` and reflects it when `u > p`. A
    point therefore has two families of images, `v0 + 2kp` and `p - v0 + 2kp`,
    and writing them down is the difference between contacts that sit on
    features and contacts that sit on the seams between tiles.
    """
    out: list[float] = []
    k = int(half_range_um // (2 * period_um)) + 2
    for i in range(-k, k + 1):
        for v in (v0 + 2 * i * period_um, period_um - v0 + 2 * i * period_um):
            if abs(v) <= half_range_um and not any(abs(v - w) < 1e-6 for w in out):
                out.append(v)
    return sorted(out)


@dataclass
class Sample:
    """A surface: height(x, y) in meters over a scanner-frame coordinate system.

    Seeded from a real Asylum scan (channel 0 = height) placed at the center
    of the scanner range; outside the map, a gently tilted plane with terrace
    steps continues the surface so scans off the map edge still look real.
    """

    height_map: np.ndarray          # (nx, ny) meters
    extent: float                   # physical size of the map (m), square
    # residual sample tilt after coarse alignment: ~0.2 um across the 90 um
    # scanner range, the scale a real stage leaves behind
    tilt: tuple[float, float] = (0.20e-6 / 90e-6, -0.15e-6 / 90e-6)
    step_height: float = 0.4e-9     # LAO/STO unit-cell terrace step
    name: str = "synthetic"
    # written-state (c-AFM lithography): dose grid over the full scanner range.
    # Positive tip bias deposits dose (protonates the surface, writes wires);
    # negative bias removes it. Local conductance is a sigmoid of dose.
    dose_n: int = 1024
    dose_range: float = 90e-6
    dose: np.ndarray | None = None
    write_radius: float = 60e-9     # effective written half-width
    # AC-mode amplitude channel, same pixel grid as height_map, for pill
    # detection (see _pills_from_amplitude). None for a synthetic/height-only
    # sample, which keeps the height-mesa detection path.
    detect_map: np.ndarray | None = None
    # The capture's own retrace, aligned with height_map, so the twin scan can show it.
    retrace_map: np.ndarray | None = None
    # Raw "zdrive_v" / "amp_v" volts the instrument recorded, same [x, y] grid as height_map.
    telemetry_maps: dict | None = None

    def __post_init__(self):
        if self.dose is None:
            self.dose = np.zeros((self.dose_n, self.dose_n), dtype=np.float32)
        self.seed_contacts()

    def seed_contacts(self, rel_height: float | None = None):
        """Interface contacts: mesas in the topography are conductive pads.

        Threshold adapts to the map: halfway between the median and the 95th
        percentile (plane-removed maps have feature amplitudes of tens of
        nm). Pad cells are seeded with saturating dose, so leads written to
        a pad edge are galvanically connected in the written-state map.
        """
        from scipy import ndimage

        h = self.height_map
        if rel_height is None:
            rel_height = max(20e-9, 0.5 * (np.percentile(h, 95) - np.median(h)))
        mask = h > (np.median(h) + rel_height)
        self.metal_mask = np.zeros((self.dose_n, self.dose_n), dtype=bool)
        self.contact_centroids_um = []
        if not mask.any():
            return

        # Evaluate the mirror-tiled pad mask over the whole scanner range, so
        # the repeating contact array of the real chip is present everywhere
        # the tip can reach, not only under the loaded scan.
        nx, ny = h.shape
        n = self.dose_n
        ax = (np.arange(n) / (n - 1) - 0.5) * self.dose_range
        gx, gy = np.meshgrid(ax, ax, indexing="ij")
        fx, fy = self._map_coords(gx, gy)
        tiled = mask[np.clip(fx.astype(int), 0, nx - 1),
                     np.clip(fy.astype(int), 0, ny - 1)]
        self.metal_mask = tiled
        self.dose[tiled] = np.maximum(self.dose[tiled], METAL_SEED_DOSE)

        # The interface contacts are PILLS, and a pill is not a mesa.
        #
        # User request, 2026-08-08, marking eight of them on a scan: "you are off on
        # the electrodes. They really look like pills, and the boundary is
        # raised."
        #
        # That last clause is the whole detection. The pill's INTERIOR sits at
        # the surrounding height and only its rim stands proud, so `h >
        # threshold` -- which is all this used to do -- finds the top gold
        # running out to the bond pads and can never find a pill.
        #
        # The rim IS visible in the AC-mode amplitude channel, where the
        # feedback lags on its slope, so when a scan carries amplitude
        # (detect_map, from from_ibw) we find the pills there -- validated on
        # data/samples: all eight radial pills on SA40656B0002. The height-only
        # route (_enclosed_by_a_raised_rim, which finds no pocket on any real
        # scan) remains the fallback for a synthetic/height-only sample. Solid
        # gold has no enclosed pocket either way, so it cannot masquerade as a
        # pill.
        contacts = None
        if self.detect_map is not None:
            contacts = _pills_from_amplitude(ndimage, self.detect_map)
        if contacts is None:
            contacts = _enclosed_by_a_raised_rim(ndimage, mask)
        if contacts is not None:
            mask = contacts
            # An interface contact touches the 2DEG, so it is an electrode:
            # seed it conductive as well, or a lead reaching a pill lands on
            # nothing.
            tiled_pills = contacts[np.clip(fx.astype(int), 0, nx - 1),
                                   np.clip(fy.astype(int), 0, ny - 1)]
            self.metal_mask = self.metal_mask | tiled_pills
            self.dose[tiled_pills] = np.maximum(self.dose[tiled_pills],
                                                METAL_SEED_DOSE)

        # Contact centroids, found in the LOADED SCAN and then mirrored out.
        #
        # They used to be found by labelling the tiled mask, and that is
        # wrong in a way that is invisible until you try to use them: a
        # feature touching the edge of the scan merges with its own
        # reflection across the seam, and the merged component's centroid
        # lands ON the seam by symmetry. Measured on SA40656B0000, 21 of the
        # 40 "contacts" sat exactly on the +-15.06 um mirror line and NOT ONE
        # was inside the 30 um scan. User request: "i dont think you found the
        # interface contacts. I wrote between two of them and no jump."
        # There was nothing in the frame to write between.
        #
        # Labelling the loaded map cannot merge anything across a seam,
        # because there is no seam in it. The images are then generated from
        # the tiling rule instead of being discovered by accident.
        lab, ncomp = ndimage.label(mask)
        if not ncomp:
            return
        p_um = self.extent * 1e6
        r_um = self.dose_range * 1e6 / 2.0
        px_um = self.extent * 1e6 / max(nx, 1)
        sizes = ndimage.sum(mask, lab, range(1, ncomp + 1))
        cents = []
        for k in np.argsort(sizes)[::-1]:
            if sizes[k] < 0.0005 * mask.size:
                continue
            # The most interior point of the component, not its centre of
            # mass. A contact is where an electrode is attached, and the
            # centre of mass of a C-shaped or ringed feature is off the
            # material entirely: three of the eight found in this scan sat on
            # bare surface. The peak of the distance transform is the
            # furthest point from any edge, so it is inside by construction
            # and it is also the most room a 2.5 um pad can be given.
            ci, cj = _deepest_point(ndimage, lab == k + 1)
            x0 = (ci / (nx - 1) - 0.5) * p_um
            y0 = (cj / (ny - 1) - 0.5) * p_um
            area = float(sizes[k] * px_um ** 2)
            for x in _mirror_images(x0, p_um, r_um):
                for y in _mirror_images(y0, p_um, r_um):
                    cents.append((x, y, area))
        # nearest first, so the ones in the frame someone is looking at are
        # the ones that survive the cap
        cents.sort(key=lambda c: c[0] ** 2 + c[1] ** 2)
        self.contact_centroids_um = cents[:40]
        # Anything the user has defined replaces the guess entirely.
        self._apply_user_contacts()

    def set_contacts(self, pills) -> None:
        """Define the interface contacts by hand, as PILLS.

        User report, after the detection failed to find them: "I need to define
        the contacts. maybe you can give me a tool that puts the contacts in
        various places, that I can define (for now). later I will give the
        klayout that created these and it will be automatic."

        So the unit is the shape, not a point: centre, size and angle of a
        stadium. That is what a person marks, and it is what a KLayout cell
        will hand over, so the automatic path can fill exactly this structure
        and nothing downstream has to learn a second one.

        A contact is conductive: it touches the 2DEG, and a lead reaching one
        has to land on something.
        """
        self.user_contacts = [dict(p) for p in (pills or [])]
        self.seed_contacts()
        # seed_contacts returns early on a scan with no mesas at all, so apply
        # again here: the user's marks must not depend on the detector having
        # got far enough to hand over.
        self._apply_user_contacts()

    def _apply_user_contacts(self) -> bool:
        # None means "nobody has said", and the detector's guess stands.
        # An EMPTY LIST means "there are none", which is a different claim and
        # has to survive: "i also want to erase what's there."
        pills = getattr(self, "user_contacts", None)
        if pills is None:
            return False
        if not pills:
            self.contact_centroids_um = []
            return True
        n = self.dose_n
        ax = (np.arange(n) / (n - 1) - 0.5) * self.dose_range * 1e6
        gx, gy = np.meshgrid(ax, ax, indexing="ij")

        mask = np.zeros((n, n), dtype=bool)
        cents = []
        for p in pills:
            a = np.deg2rad(float(p.get("angle_deg", 0.0)))
            dx, dy = gx - float(p["x_um"]), gy - float(p["y_um"])
            u = dx * np.cos(a) + dy * np.sin(a)
            v = -dx * np.sin(a) + dy * np.cos(a)
            hw = max(float(p.get("w_um", 2.0)), 1e-6) / 2
            hh = max(float(p.get("h_um", 2.0)), 1e-6) / 2
            r = min(hw, hh)                       # a stadium: fully round ends
            qx = np.maximum(np.abs(u) - (hw - r), 0.0)
            qy = np.maximum(np.abs(v) - (hh - r), 0.0)
            mask |= np.hypot(qx, qy) <= r
            # a stadium: a disc plus the straight part along the LONG axis
            area = np.pi * r * r + max(0.0, 2 * max(hw, hh) - 2 * r) * 2 * r
            cents.append((float(p["x_um"]), float(p["y_um"]), float(area)))

        self.metal_mask = self.metal_mask | mask
        self.dose[mask] = np.maximum(self.dose[mask], METAL_SEED_DOSE)
        self.contact_centroids_um = cents
        return True

    def set_range(self, range_m: float):
        """Point the written-state grid at this instrument's scanner range."""
        if abs(range_m - self.dose_range) < 1e-12:
            return
        self.dose_range = float(range_m)
        self.reset_written()

    def reset_written(self):
        self.dose[:] = 0.0
        self.seed_contacts()

    def is_metal_um(self, x_um: float, y_um: float) -> bool:
        mm = getattr(self, "metal_mask", None)
        if mm is None:
            return False
        i, j = self._dose_idx(x_um * 1e-6, y_um * 1e-6)
        return bool(mm[i, j])

    # dose-grid helpers -----------------------------------------------------

    def _dose_idx(self, x_m: float, y_m: float) -> tuple[int, int]:
        n = self.dose_n
        i = int((x_m / self.dose_range + 0.5) * (n - 1) + 0.5)
        j = int((y_m / self.dose_range + 0.5) * (n - 1) + 0.5)
        return min(max(i, 0), n - 1), min(max(j, 0), n - 1)

    def deposit(self, x_m: float, y_m: float, bias_v: float, dt: float):
        """Accumulate write/erase dose at the tip position (called per tick)."""
        if abs(bias_v) < 1.0:
            return
        i, j = self._dose_idx(x_m, y_m)
        r = 1  # 3x3 stamp ~ write_radius at the default grid pitch (88 nm)
        lo_i, hi_i = max(i - r, 0), min(i + r + 1, self.dose_n)
        lo_j, hi_j = max(j - r, 0), min(j + r + 1, self.dose_n)
        if bias_v > 0:
            self.dose[lo_i:hi_i, lo_j:hi_j] += bias_v * dt * 8.0
        else:
            self.dose[lo_i:hi_i, lo_j:hi_j] *= max(0.0, 1.0 + bias_v * dt * 40.0)
            # gold cannot be erased
            mm = getattr(self, "metal_mask", None)
            if mm is not None:
                sub = mm[lo_i:hi_i, lo_j:hi_j]
                self.dose[lo_i:hi_i, lo_j:hi_j][sub] = np.maximum(
                    self.dose[lo_i:hi_i, lo_j:hi_j][sub], 25.0)

    def conductance(self, x_m: float, y_m: float) -> float:
        """Local written-state conductance, 0..1 (sigmoid of accumulated dose)."""
        i, j = self._dose_idx(x_m, y_m)
        d = float(self.dose[i, j])
        return d * d / (1.0 + d * d)

    def height_scalar(self, x_m: float, y_m: float) -> float:
        """Fast single-point height for the per-tick executor."""
        half = self.extent / 2.0
        nx, ny = self.height_map.shape
        fx = (x_m + half) / self.extent * (nx - 1)
        fy = (y_m + half) / self.extent * (ny - 1)
        base = self.tilt[0] * x_m + self.tilt[1] * y_m
        if 0 <= fx <= nx - 1 and 0 <= fy <= ny - 1:
            x0, y0 = int(fx), int(fy)
            tx, ty = fx - x0, fy - y0
            x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)
            h = (self.height_map[x0, y0] * (1 - tx) * (1 - ty)
                 + self.height_map[x1, y0] * tx * (1 - ty)
                 + self.height_map[x0, y1] * (1 - tx) * ty
                 + self.height_map[x1, y1] * tx * ty)
            return float(h + base)
        import math as _m
        return base + self.step_height * _m.floor((x_m + 2 * y_m) / 1.5e-6)

    @classmethod
    def from_ibw(cls, path: str | Path, channel: int = 0) -> "Sample":
        w = read_ibw(path)
        data = w.data[..., channel] if w.data.ndim == 3 else w.data
        data = np.asarray(data, dtype=np.float64)
        data = data - np.nanmean(data)
        # remove plane fit so the map sits flat; the world adds its own tilt
        nx, ny = data.shape
        gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        a = np.column_stack([gx.ravel(), gy.ravel(), np.ones(nx * ny)])
        coef, *_ = np.linalg.lstsq(a, data.ravel(), rcond=None)
        data = data - (coef[0] * gx + coef[1] * gy + coef[2])
        extent = w.sfA[0] * data.shape[0]
        # The amplitude channel, RAW (not plane-removed -- it isn't topography):
        # the pill detector reads its rim edges. Same pixel grid as height.
        detect = None
        li = _amplitude_layer(w)
        if li is not None:
            detect = np.asarray(w.data[..., li], dtype=np.float64)
        return cls(height_map=data, extent=extent, name=w.name, detect_map=detect,
                   telemetry_maps=telemetry_maps_from_ibw(w) or None)

    def _map_coords(self, x: np.ndarray, y: np.ndarray):
        """Mirror-tiled fractional indices into the height map.

        A loaded scan covers less than the 90 um scanner range (30 um is
        typical), but the real chip extends across the whole range with its
        pad array repeating. Reflecting the map at its edges continues the
        surface seamlessly (no discontinuity at the boundary) and repeats the
        contact structures the way a real sample does, instead of inventing
        terrain with different statistics.
        """
        nx, ny = self.height_map.shape
        p = self.extent

        def tri(v):
            u = np.mod(np.asarray(v, dtype=np.float64) + p / 2.0, 2 * p)
            return np.where(u > p, 2 * p - u, u) / p          # 0..1, mirrored

        return tri(x) * (nx - 1), tri(y) * (ny - 1)

    def height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear-interpolated height at absolute scanner coordinates (m)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        nx, ny = self.height_map.shape
        fx, fy = self._map_coords(x, y)
        x0 = np.clip(fx.astype(int), 0, nx - 1)
        y0 = np.clip(fy.astype(int), 0, ny - 1)
        x1 = np.minimum(x0 + 1, nx - 1)
        y1 = np.minimum(y0 + 1, ny - 1)
        tx = fx - x0
        ty = fy - y0
        h = (self.height_map[x0, y0] * (1 - tx) * (1 - ty)
             + self.height_map[x1, y0] * tx * (1 - ty)
             + self.height_map[x0, y1] * (1 - tx) * ty
             + self.height_map[x1, y1] * tx * ty)
        return h + self.tilt[0] * x + self.tilt[1] * y


@dataclass
class Cantilever:
    f0: float = 285e3            # resonance (Hz), AC-mode lever
    Q: float = 420.0
    k: float = 26.0              # N/m
    invols: float = 60e-9        # m of deflection per V at the detector
    free_amplitude_v: float = 1.0
    contact_stiffness: float = 5e3  # effective sample-contact spring seen in defl

    def deflection_v(self, indentation_m: np.ndarray) -> np.ndarray:
        """Contact-mode deflection (V) for tip pressed `indentation` into surface."""
        d = np.maximum(indentation_m, 0.0)
        return d / self.invols

    def amplitude_v(self, gap_m: np.ndarray) -> np.ndarray:
        """Tapping amplitude (V) vs mean tip-surface gap: amplitude limited by wall."""
        free_m = self.free_amplitude_v * self.invols
        a = np.clip(gap_m / free_m, 0.0, 1.0)
        # soft knee near contact
        return self.free_amplitude_v * (a * (2 - a))


@dataclass
class World:
    """Everything outside the controller: scanner, lever, surface, noise."""

    sample: Sample
    lever: Cantilever = field(default_factory=Cantilever)
    instrument: Instrument = field(default_factory=Instrument)
    # LVDT calibrations (m/V) and offsets (V): GV values the driver reads.
    # Overwritten from the instrument in __post_init__ unless pinned.
    x_lvdt_sens: float = 6.0e-6
    y_lvdt_sens: float = 6.0e-6
    z_lvdt_sens: float = 2.0e-6
    x_lvdt_offset: float = 0.0
    y_lvdt_offset: float = 0.0
    z_lvdt_offset: float = 0.0
    use_instrument_cal: bool = True
    scanner_tau: float = 1.2e-3      # first-order XY scanner response (s)
    z_tau: float = 0.25e-3
    noise_z: float = 0.05e-9         # rms surface-tracking noise (m)
    noise_defl_v: float = 0.0015
    drift_v: tuple[float, float] = (0.0, 0.0)  # slow XY drift (V/s), settable
    # Free-air deflection baseline (V) and its thermal drift (V/s). This is the
    # photodiode deflection reading with the tip OFF the surface -- what the
    # vendor records as FreeAirDeflection (td_ReadValue("DeflectionValue"),
    # ScanMaster.ipf:4665). It walks thermally (laser/lever/detector), which is
    # exactly what makes an ABSOLUTE contact-mode setpoint drift into a Z rail;
    # the relative-setpoint loop re-references to it each frame. Default 0 (no
    # drift), so it is inert unless a test/scenario sets free_air_defl_drift_v.
    free_air_defl_v: float = 0.0
    free_air_defl_drift_v: float = 0.0  # V/s baseline walk, settable
    # Simulated HV Z-DRIVE voltage (V, in the -10..+150 V bipolar-controller
    # domain) -- the twin analogue of the readback the rail-dwell safety guard
    # watches (td_ReadValue("Output.Z"); docs/SAFETY_KERNEL_RAIL_DWELL.md). It is
    # a value the twin can DRIVE to a rail for tests (a fixed level, or a thermal
    # walk via z_drive_drift_v), exactly like free_air_defl_v above. Default 75 V
    # (mid-range, not railed) so the guard is inert unless a scenario walks/pins
    # it. This is NOT wired to the twin's own z_v servo state -- it is an
    # independent, deterministic knob so the guard is exactly assertable.
    z_drive_v: float = 75.0
    z_drive_drift_v: float = 0.0        # V/s Z-drive walk toward a rail, settable
    # Simulated photodiode SUM voltage (V) -- the twin analogue of the readback the
    # host-side fail-safe monitors poll each re-reference (td_ReadValue("DetectorSum"),
    # CONFIRMED on the Cypher and the MFP-3D, 2026-09-15 -- lowercase "sum" reads 0 V on
    # the Cypher; docs/SAFETY_KERNEL_PLAN.md Sec 0). The 0.9*SUM
    # clamp asserts |deflection| <= SUM physically, and the SUM-loss interlock trips
    # when the laser/lever is lost (two consecutive reads < 0.1 V). A deterministic
    # knob (no noise) so both are exactly assertable; can walk toward 0 via
    # detector_sum_drift_v. Default 5 V (a healthy sum) so both monitors are inert
    # unless a scenario drives it.
    detector_sum_v: float = 5.0
    detector_sum_drift_v: float = 0.0   # V/s SUM walk (toward laser/lever loss), settable

    # state (volts at the piezo/LVDT boundary)
    x_v: float = 0.0
    y_v: float = 0.0
    z_v: float = 0.0
    engaged: bool = False
    imaging_mode: str = "Contact"    # or "AC"
    t: float = 0.0

    def __post_init__(self):
        """Adopt the instrument's own calibration (from its .ibw metadata)."""
        if not self.use_instrument_cal:
            return
        i = self.instrument
        self.x_lvdt_sens = i.x_lvdt_sens
        self.y_lvdt_sens = i.y_lvdt_sens
        self.z_lvdt_sens = i.z_lvdt_sens
        self.x_lvdt_offset = i.x_lvdt_offset
        self.y_lvdt_offset = i.y_lvdt_offset
        self.z_lvdt_offset = i.z_lvdt_offset
        self.lever.f0 = i.thermal_f0
        self.lever.Q = i.thermal_q
        self.lever.k = i.spring_constant
        self.lever.invols = i.invols
        # the sample's written-state grid spans this instrument's scanner
        self.sample.set_range(i.xy_range_um * 1e-6)

    def positions_m(self) -> tuple[float, float]:
        return (
            (self.x_v - self.x_lvdt_offset) * self.x_lvdt_sens,
            (self.y_v - self.y_lvdt_offset) * self.y_lvdt_sens,
        )

    def free_air_deflection(self) -> float:
        """The off-contact deflection baseline (V), walked by thermal drift over
        the world clock. The twin analogue of td_ReadValue("DeflectionValue")
        taken after a withdraw+settle. Deterministic (no noise) so the relative-
        setpoint path is exactly assertable off-instrument."""
        return self.free_air_defl_v + self.free_air_defl_drift_v * self.t

    def z_drive_voltage(self) -> float:
        """The simulated HV Z-drive voltage (V), walked by thermal drift over the
        world clock and clamped to the -10..+150 V rail (a physical bipolar
        controller cannot exceed it). The twin analogue of td_ReadValue("Output.Z")
        the rail-dwell safety guard polls. Deterministic (no noise), so the guard
        trips are exactly assertable off-instrument."""
        v = self.z_drive_v + self.z_drive_drift_v * self.t
        return float(min(150.0, max(-10.0, v)))

    def sum_voltage(self) -> float:
        """The simulated photodiode SUM voltage (V), walked by drift over the world
        clock and floored at 0 (a physical sum cannot go negative). The twin analogue
        of td_ReadValue("DetectorSum") the host-side fail-safe reads each re-reference.
        Deterministic (no noise), so the 0.9*SUM clamp and the SUM-loss interlock are
        exactly assertable off-instrument."""
        v = self.detector_sum_v + self.detector_sum_drift_v * self.t
        return float(max(0.0, v))

    def step_block(self, x_cmd_v: np.ndarray, y_cmd_v: np.ndarray, z_cmd_v: np.ndarray,
                   dt: float, rng: np.random.Generator):
        """Advance the world through a block of samples.

        Inputs are commanded piezo voltages per sample (after the ARC's servo
        loops); returns per-sample sensor readings:
        (x_sensor_v, y_sensor_v, z_sensor_v, deflection_v, amplitude_v)
        """
        n = len(x_cmd_v)
        ax = np.exp(-dt / self.scanner_tau)
        az = np.exp(-dt / self.z_tau)

        xs = np.empty(n)
        ys = np.empty(n)
        zs = np.empty(n)
        x, y, z = self.x_v, self.y_v, self.z_v
        for i in range(n):
            x = ax * x + (1 - ax) * x_cmd_v[i]
            y = ax * y + (1 - ax) * y_cmd_v[i]
            z = az * z + (1 - az) * z_cmd_v[i]
            xs[i] = x
            ys[i] = y
            zs[i] = z
        self.x_v, self.y_v, self.z_v = float(x), float(y), float(z)
        self.t += n * dt

        x_m = (xs - self.x_lvdt_offset) * self.x_lvdt_sens
        y_m = (ys - self.y_lvdt_offset) * self.y_lvdt_sens
        # Sign convention (matches the capture: -10 V = fully retracted, tip
        # high above the surface; increasing voltage lowers the tip):
        tip_height = -(zs - self.z_lvdt_offset) * self.z_lvdt_sens

        surface = self.sample.height(x_m, y_m)
        surface = surface + rng.normal(0.0, self.noise_z, n)

        indentation = surface - tip_height          # > 0 when pressed in
        gap = np.maximum(tip_height - surface, 0.0)  # > 0 when free
        defl = self.lever.deflection_v(indentation)
        amp = self.lever.amplitude_v(gap)
        self.engaged = bool(gap[-1] < 5 * self.lever.free_amplitude_v * self.lever.invols)

        defl = defl + rng.normal(0.0, self.noise_defl_v, n)
        return xs, ys, zs, defl, amp
