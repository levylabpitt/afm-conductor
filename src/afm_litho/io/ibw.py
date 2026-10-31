"""Minimal Igor Binary Wave (IBW) v5 reader.

The .ibw boundary codec, import direction. Enough of the v5 format to load
Asylum Research scan files: the data array, per-dimension scaling, and the
text note (which Asylum fills with scan metadata as "Key: value" lines).

Format references: Igor Technical Note TN003 and the WaveHeader5 struct,
which the legacy LabVIEW reader embeds verbatim.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# wave type codes -> numpy dtypes (complex flag is bit 0)
_TYPE_TABLE = {
    2: np.float32,
    4: np.float64,
    8: np.int8,
    0x10: np.int16,
    0x20: np.int32,
    0x48: np.uint8,
    0x50: np.uint16,
    0x60: np.uint32,
}

MAXDIMS = 4


@dataclass
class IBWave:
    name: str
    data: np.ndarray
    sfA: list[float]        # per-dimension step
    sfB: list[float]        # per-dimension origin
    data_units: str
    dim_units: list[str]
    note: str
    note_dict: dict[str, str] = field(default_factory=dict)

    @property
    def channel_names(self) -> list[str]:
        """Asylum stacks channels in the 3rd dimension; labels live in the note."""
        n = self.data.shape[2] if self.data.ndim == 3 else 1
        listed = self.note_dict.get("ChannelList", "")
        names = [s for s in listed.replace(";", ",").split(",") if s]
        return names if len(names) == n else [f"ch{i}" for i in range(n)]


def _parse_note(note: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in note.replace("\r", "\n").split("\n"):
        if ":" in line:
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if k and k not in out:
                out[k] = v
    return out


def read_ibw(path: str | Path) -> IBWave:
    raw = Path(path).read_bytes()
    version = struct.unpack_from("<h", raw, 0)[0]
    if version not in (5,):
        raise ValueError(f"unsupported IBW version {version} in {path}")

    # BinHeader5 (64 bytes): short version; short checksum; int wfmSize; int formulaSize;
    # int noteSize; int dataEUnitsSize; int dimEUnitsSize[4]; int dimLabelsSize[4];
    # int sIndicesSize; int optionsSize1; int optionsSize2
    bh = struct.unpack_from("<hhiiii4i4iiii", raw, 0)
    wfm_size = bh[2]
    formula_size = bh[3]
    note_size = bh[4]
    data_e_units_size = bh[5]
    dim_e_units_sizes = bh[6:10]
    dim_labels_sizes = bh[10:14]
    bin_header_size = 64
    wh_off = bin_header_size

    # WaveHeader5 field offsets (verified against Asylum files):
    #   npnts +12, type +16, bname +28 (32 bytes), nDim +68, sfA +84, sfB +116
    wave_type = struct.unpack_from("<h", raw, wh_off + 16)[0]
    npnts = struct.unpack_from("<i", raw, wh_off + 12)[0]
    n_dim = list(struct.unpack_from("<4i", raw, wh_off + 68))
    sfA = list(struct.unpack_from("<4d", raw, wh_off + 84))
    sfB = list(struct.unpack_from("<4d", raw, wh_off + 116))
    name = _printable(struct.unpack_from("<32s", raw, wh_off + 28)[0])

    dtype = _TYPE_TABLE.get(wave_type)
    if dtype is None:
        raise ValueError(f"unsupported wave type 0x{wave_type:x}")

    dims = [d for d in n_dim if d > 0] or [npnts]
    count = int(np.prod(dims))
    data_off = bin_header_size + 320  # WaveHeader5 is 320 bytes; wData begins right after
    data = np.frombuffer(raw, dtype=dtype, count=count, offset=data_off)
    if len(dims) > 1:
        # Igor stores fastest-varying dimension first: reshape as (dimN,...,dim0).T
        data = data.reshape(dims[::-1]).T
    else:
        data = data.copy()

    pos = bin_header_size + wfm_size + formula_size
    note = raw[pos: pos + note_size].decode("latin-1", errors="replace")
    pos += note_size
    data_units = raw[pos: pos + data_e_units_size].decode("latin-1", errors="replace")
    pos += data_e_units_size
    dim_units = []
    for sz in dim_e_units_sizes:
        dim_units.append(raw[pos: pos + sz].decode("latin-1", errors="replace"))
        pos += sz
    for sz in dim_labels_sizes:
        pos += sz

    return IBWave(
        name=name,
        data=data,
        sfA=sfA,
        sfB=sfB,
        data_units=data_units,
        dim_units=dim_units,
        note=note,
        note_dict=_parse_note(note),
    )


def _printable(b: bytes) -> str:
    s = b.split(b"\x00", 1)[0]
    return "".join(chr(c) for c in s if 32 <= c < 127)


# --- writer: the exact inverse of read_ibw ----------------------------------

# numpy dtype -> wave type code (inverse of _TYPE_TABLE). Anything else is
# promoted to float64 (Igor v5's general numeric type).
_DTYPE_TO_TYPE = {
    np.dtype(np.float32): 2,
    np.dtype(np.float64): 4,
    np.dtype(np.int8): 8,
    np.dtype(np.int16): 0x10,
    np.dtype(np.int32): 0x20,
    np.dtype(np.uint8): 0x48,
    np.dtype(np.uint16): 0x50,
    np.dtype(np.uint32): 0x60,
}


def _wave_name(s: str, default: str = "wave0") -> str:
    """A safe Igor v5 wave name: alnum/underscore only, starts with a letter or
    underscore, <=31 chars. Igor rejects hyphens and leading digits in names."""
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(s))
    if not out:
        out = default
    if not (out[0].isalpha() or out[0] == "_"):
        out = "w" + out
    return out[:31]


def write_ibw(path, data, *, name: str = "wave0", sfA=None, sfB=None,
              data_units: str = "", dim_units=None, note: str = "") -> Path:
    """Write `data` as an Igor Binary Wave v5 file -- the byte-exact inverse of
    `read_ibw` (`write_ibw -> read_ibw` round-trips data, per-dim `sfA`/`sfB`
    scaling, `data_units`, `dim_units`, and the text `note`).

    Handles 1-D (e.g. IV: a current array on a scaled bias axis) and 2-D/3-D
    (Asylum stacks scan channels in the 3rd dimension -- put a `ChannelList` note
    key so the reader's `channel_names` resolves). Igor stores the fastest-varying
    dimension first (Fortran order); the reader's `reshape(dims[::-1]).T` is
    mirrored here by `tobytes(order="F")`.

    `sfA`/`sfB` are per-dimension step/origin (a value is `sfB[d] + i*sfA[d]`);
    give one entry per used dimension (unused dims default to step 1, origin 0).
    No third-party dependency -- the struct packing is hand-rolled like the reader.
    """
    arr = np.asarray(data)
    le_dtype = arr.dtype.newbyteorder("<") if arr.dtype in _DTYPE_TO_TYPE \
        else np.dtype("<f8")
    arr = np.asarray(arr, dtype=le_dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > MAXDIMS:
        raise ValueError(f"IBW supports up to {MAXDIMS} dims, got {arr.ndim}")
    wave_type = _DTYPE_TO_TYPE[le_dtype.newbyteorder("=")]

    dims = list(arr.shape)
    n_dim = (dims + [0, 0, 0, 0])[:4]
    npnts = int(np.prod(dims))
    # Igor order: fastest-varying (first) dimension contiguous == Fortran order.
    data_bytes = arr.tobytes(order="F")

    scale_a = [1.0, 1.0, 1.0, 1.0]
    scale_b = [0.0, 0.0, 0.0, 0.0]
    for i, v in enumerate((sfA or [])[:4]):
        scale_a[i] = float(v)
    for i, v in enumerate((sfB or [])[:4]):
        scale_b[i] = float(v)

    # WaveHeader5: a fixed 320-byte struct; only the fields read_ibw consumes are
    # filled (documented offsets), plus whVersion=1 at +26. wData follows at +320.
    wh = bytearray(320)
    struct.pack_into("<i", wh, 12, npnts)
    struct.pack_into("<h", wh, 16, wave_type)
    struct.pack_into("<h", wh, 26, 1)                       # whVersion
    struct.pack_into("<32s", wh, 28, name.encode("ascii", "replace")[:31])
    struct.pack_into("<4i", wh, 68, *n_dim)
    struct.pack_into("<4d", wh, 84, *scale_a)
    struct.pack_into("<4d", wh, 116, *scale_b)

    note_b = note.encode("latin-1", "replace")
    du_b = data_units.encode("latin-1", "replace")
    du_list = list(dim_units or [])
    dimu_b = [(du_list[i] if i < len(du_list) else "").encode("latin-1", "replace")
              for i in range(MAXDIMS)]

    # BinHeader5 (64 bytes). wfmSize = 320 (WaveHeader5) + wave data; formulaSize=0
    # so the note begins right after the data (matches real Asylum files).
    wfm_size = 320 + len(data_bytes)
    bh = struct.pack(
        "<hhiiii4i4iiii",
        5, 0,                                               # version, checksum(0)
        wfm_size, 0, len(note_b), len(du_b),                # wfm, formula, note, dataEU
        *[len(x) for x in dimu_b],                          # dimEUnitsSize[4]
        0, 0, 0, 0,                                         # dimLabelsSize[4]
        0, 0, 0)                                            # sIndices, options1/2
    header = bytearray(bh) + wh                             # 64 + 320 = 384 bytes
    # Igor's checksum: the 16-bit sum of the 192 shorts over BinHeader+WaveHeader
    # must be 0; solve for the checksum field (offset 2).
    chksum = (-sum(struct.unpack_from("<192h", header, 0))) & 0xFFFF
    struct.pack_into("<H", header, 2, chksum)

    blob = bytes(header) + data_bytes + note_b + du_b + b"".join(dimu_b)
    path = Path(path)
    path.write_bytes(blob)
    return path


def export_capture_ibw(runtime_dir, session_id: str, cap: dict, config,
                       cal: dict, *, out_dir=None) -> Path | None:
    """Project a RAW capture into its channels and write ONE `<session_id>.ibw`
    next to the session's npz (`captures/`, or `out_dir` if given) -- the `.ibw`
    twin of `export_capture_gwy`.

    FORM: the projected channels (all the same grid shape, from
    `capture_channels_2d`) are stacked into a single 3-D wave `(pixels, rows,
    n_channels)` -- Asylum's own convention (channels in the 3rd dimension) -- with
    a `ChannelList` note key so `read_ibw().channel_names` recovers the titles, and
    a `ChannelUnits` key recording each channel's z unit. XY scaling is the real
    field size; base SI (ZSensor topography in metres, others in volts). NaN/inf are
    sanitized to 0.0 like the gwy exporter. Returns the path, or None if empty.
    """
    from .capture_grid import capture_channels_2d

    channels = capture_channels_2d(cap, config, cal)
    if not channels:
        return None
    names = [ch["name"] for ch in channels]
    units = [ch["z_units"] for ch in channels]
    grids = [np.nan_to_num(np.asarray(ch["grid2d"], dtype=np.float64),
                           nan=0.0, posinf=0.0, neginf=0.0) for ch in channels]
    # every channel grid is (rows, pixels); transpose to (pixels, rows) so the
    # fast (x) axis is dim-0, then stack channels along dim-2.
    rows, pixels = grids[0].shape
    stack = np.stack([g.T for g in grids], axis=-1)         # (pixels, rows, nch)

    x_real = float(channels[0]["x_real_m"])
    y_real = float(channels[0]["y_real_m"])
    sfA = [x_real / pixels if pixels else 1.0,
           y_real / rows if rows else 1.0, 1.0]
    common = units[0] if len(set(units)) == 1 else ""       # one wave, one z unit
    note = f"ChannelList: {';'.join(names)}\r" \
           f"ChannelUnits: {';'.join(units)}\r"

    out_dir = Path(out_dir) if out_dir is not None else Path(runtime_dir) / "captures"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{session_id}.ibw"
    return write_ibw(path, stack, name=_wave_name(session_id), sfA=sfA,
                     sfB=[0.0, 0.0, 0.0], data_units=common,
                     dim_units=["m", "m", ""], note=note)
