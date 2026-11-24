"""
geo_compat.py
=============
Lapisan kompatibilitas untuk dua gaya API TorchGeo:

  (a) versi lama : sampler menghasilkan `BoundingBox(minx, maxx, miny, maxy, mint, maxt)`
                   dan `dataset.index` adalah R-tree.
  (b) versi baru : sampler menghasilkan tuple of slices `(slice_x, slice_y, slice_t)`
                   dan `dataset.index` adalah GeoDataFrame.

Semua kode lain di proyek ini TIDAK menyentuh detail bbox secara langsung,
cukup lewat fungsi di sini. Jalankan `scripts/phase0_check_setup.py` untuk
memastikan lapisan ini bekerja dengan versi TorchGeo yang terpasang.
"""

import re
from dataclasses import dataclass
from pathlib import Path


def bbox_coords(b):
    """Return (minx, maxx, miny, maxy, mint, maxt) dari bbox gaya apa pun."""
    if hasattr(b, "minx"):
        return b.minx, b.maxx, b.miny, b.maxy, b.mint, b.maxt
    if isinstance(b, tuple) and len(b) >= 2 and all(isinstance(s, slice) for s in b[:2]):
        xs, ys = b[0], b[1]
        ts = b[2] if len(b) > 2 and isinstance(b[2], slice) else slice(None, None)
        return xs.start, xs.stop, ys.start, ys.stop, ts.start, ts.stop
    raise TypeError(f"Format bbox tidak dikenali: {type(b)} -> {b!r}")


def shift_bbox(b, dx: float, dy: float):
    """Geser bbox sejauh (dx, dy) dalam satuan CRS, mempertahankan tipe aslinya."""
    if hasattr(b, "minx"):
        return type(b)(b.minx + dx, b.maxx + dx, b.miny + dy, b.maxy + dy, b.mint, b.maxt)
    xs, ys, *rest = b
    return (
        slice(xs.start + dx, xs.stop + dx, xs.step),
        slice(ys.start + dy, ys.stop + dy, ys.step),
        *rest,
    )


@dataclass
class Scene:
    scene_id: str
    path: str
    minx: float
    maxx: float
    miny: float
    maxy: float
    mint: object = None
    maxt: object = None


def _scene_id_from_path(path: str) -> str:
    p = Path(str(path))
    return f"{p.parent.name}/{p.name}"


def list_scenes(view) -> list:
    """Daftar scene (footprint + path file) di sebuah dataset/view TorchGeo."""
    idx = view.index
    scenes = []
    # GeoDataFrame dicek DULU: ia juga punya atribut `intersection` & `bounds`
    # (method GeoPandas), sehingga bisa salah terdeteksi sebagai R-tree.
    if hasattr(idx, "geometry"):  # GeoDataFrame (baru, mis. TorchGeo 0.8)
        # Dataset gabungan (mis. L8Biome = image & mask) punya filepath_1, filepath_2, ...
        # -> pakai kolom filepath pertama (image) sebagai identitas scene.
        path_cols = sorted(c for c in idx.columns if str(c).startswith("filepath"))
        for interval, row in idx.iterrows():
            minx, miny, maxx, maxy = row.geometry.bounds
            path = row[path_cols[0]] if path_cols else str(interval)
            mint = getattr(interval, "left", None)
            maxt = getattr(interval, "right", None)
            scenes.append(Scene(_scene_id_from_path(path), str(path), minx, maxx, miny, maxy, mint, maxt))
    elif hasattr(idx, "intersection") and hasattr(idx, "bounds"):  # R-tree (lama)
        for hit in idx.intersection(idx.bounds, objects=True):
            minx, maxx, miny, maxy, mint, maxt = hit.bounds
            path = hit.object if isinstance(hit.object, str) else str(hit.object)
            scenes.append(Scene(_scene_id_from_path(path), path, minx, maxx, miny, maxy, mint, maxt))
    else:
        raise RuntimeError(
            "Tidak bisa membaca index TorchGeo (bukan R-tree maupun GeoDataFrame). "
            "Laporkan versi TorchGeo kamu supaya geo_compat.py bisa disesuaikan."
        )
    # Satu scene bisa punya >1 file (mis. image & mask) -> gabungkan per scene_id
    merged = {}
    for s in scenes:
        if s.scene_id not in merged:
            merged[s.scene_id] = s
    return list(merged.values())


# .../p116_r50/L7...TIF atau .../p143_r21_3/L7...TIF (akuisisi ke-3 di lokasi yang sama)
_L7_PATHROW = re.compile(r"[\\/]p(\d+)_r(\d+)(?:_\d+)?[\\/]")
_L8_PATHROW = re.compile(r"^L[COTEM]0?8(\d{3})(\d{3})")    # LC81580172013201LGN00.TIF


def location_key(path: str) -> str:
    """Lokasi WRS-2 sebuah scene, 'p<path>_r<row>'. Fallback: nama file (= 1 scene 1 lokasi)."""
    m = _L7_PATHROW.search(str(path))
    if m:
        return f"p{int(m.group(1)):03d}_r{int(m.group(2)):03d}"
    m = _L8_PATHROW.match(Path(str(path)).name)
    if m:
        return f"p{int(m.group(1)):03d}_r{int(m.group(2)):03d}"
    return Path(str(path)).name


def _time_overlap(a0, a1, b0, b1) -> bool:
    if None in (a0, a1, b0, b1):
        return True
    try:
        return not (a1 < b0 or b1 < a0)
    except TypeError:  # tipe waktu beda (float vs Timestamp) -> abaikan dimensi waktu
        return True


class SceneLookup:
    """Mencari scene asal sebuah patch (berdasarkan overlap footprint terbesar)."""

    def __init__(self, scenes: list):
        self.scenes = scenes

    def find(self, bbox) -> str:
        minx, maxx, miny, maxy, mint, maxt = bbox_coords(bbox)
        best_id, best_area = "unknown", 0.0
        for s in self.scenes:
            if not _time_overlap(mint, maxt, s.mint, s.maxt):
                continue
            w = min(maxx, s.maxx) - max(minx, s.minx)
            h = min(maxy, s.maxy) - max(miny, s.miny)
            if w > 0 and h > 0 and w * h > best_area:
                best_id, best_area = s.scene_id, w * h
        return best_id


def footprint_overlaps(scenes_a: list, scenes_b: list) -> list:
    """Pasangan scene (a, b) yang footprint-nya beririsan secara spasial (abaikan waktu)."""
    pairs = []
    for a in scenes_a:
        for b in scenes_b:
            w = min(a.maxx, b.maxx) - max(a.minx, b.minx)
            h = min(a.maxy, b.maxy) - max(a.miny, b.miny)
            if w > 0 and h > 0:
                area_a = (a.maxx - a.minx) * (a.maxy - a.miny)
                pairs.append((a.scene_id, b.scene_id, (w * h) / max(area_a, 1e-9)))
    return pairs
