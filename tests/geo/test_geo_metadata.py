"""Unit tests for EXIF geo/time extraction (spec-022, Slice 1).

Two layers:
  * ut_ExtractFromExif    - pure parse rules, hand-built EXIF dicts (no IO).
  * ut_GeoMetadataExtractor - real JPEG fixtures written via PIL (IO glue).
"""

from datetime import datetime

from PIL import Image

from geo_cluster.exif_reader import (
    ExifInputs,
    GeoMetadataExtractor,
    extract_from_exif,
)

# Fixed "now" so date-bound assertions are deterministic.
NOW = datetime(2026, 1, 1, 12, 0, 0)

_DTO = 0x9003   # DateTimeOriginal
_GPS = 0x8825   # GPSInfo


def _dms(value: float):
    """Decimal degrees -> EXIF (deg, min, sec) floats (matches PIL round-trip)."""
    deg = int(value)
    minute = int((value - deg) * 60)
    sec = round((value - deg - minute / 60.0) * 3600.0, 2)
    return (float(deg), float(minute), float(sec))


def _write_jpeg(path, *, dt=None, lat=None, lon=None):
    """Write a tiny JPEG carrying the requested EXIF tags."""
    img = Image.new("RGB", (8, 8), (120, 120, 120))
    exif = img.getexif()
    if dt is not None:
        exif.get_ifd(0x8769)[_DTO] = dt
    if lat is not None and lon is not None:
        gps = exif.get_ifd(_GPS)
        gps[1] = "N" if lat >= 0 else "S"
        gps[2] = _dms(abs(lat))
        gps[3] = "E" if lon >= 0 else "W"
        gps[4] = _dms(abs(lon))
    img.save(path, exif=exif)
    return str(path)


class ut_ExtractFromExif:
    """The pure parse function: rationals, hemispheres, date bounds."""

    def test_gps_and_time(self):
        exif = {
            _DTO: "2024:06:15 10:30:00",
            _GPS: {1: "N", 2: (47.0, 30.0, 0.0), 3: "E", 4: (19.0, 3.0, 0.0)},
        }
        ts, lat, lon = extract_from_exif(exif, min_year=1990, now=NOW)
        assert ts == datetime(2024, 6, 15, 10, 30, 0)
        assert abs(lat - 47.5) < 1e-6
        assert abs(lon - 19.05) < 1e-6

    def test_southern_western_hemisphere_is_negative(self):
        exif = {_GPS: {1: "S", 2: (33.0, 51.0, 0.0), 3: "W", 4: (70.0, 39.0, 0.0)}}
        ts, lat, lon = extract_from_exif(exif, min_year=1990, now=NOW)
        assert ts is None
        assert lat < 0 and lon < 0

    def test_time_only(self):
        ts, lat, lon = extract_from_exif(
            {_DTO: "2024:01:02 08:00:00"}, min_year=1990, now=NOW
        )
        assert ts == datetime(2024, 1, 2, 8, 0, 0)
        assert lat is None and lon is None

    def test_gps_only(self):
        exif = {_GPS: {1: "N", 2: (10.0, 0.0, 0.0), 3: "E", 4: (20.0, 0.0, 0.0)}}
        ts, lat, lon = extract_from_exif(exif, min_year=1990, now=NOW)
        assert ts is None
        assert lat == 10.0 and lon == 20.0

    def test_neither(self):
        assert extract_from_exif(None, min_year=1990, now=NOW) == (None, None, None)
        assert extract_from_exif({}, min_year=1990, now=NOW) == (None, None, None)

    def test_pre_min_year_rejected(self):
        ts, _, _ = extract_from_exif(
            {_DTO: "1980:05:01 00:00:00"}, min_year=1990, now=NOW
        )
        assert ts is None

    def test_future_date_rejected(self):
        ts, _, _ = extract_from_exif(
            {_DTO: "2099:05:01 00:00:00"}, min_year=1990, now=NOW
        )
        assert ts is None

    def test_malformed_date_rejected(self):
        ts, _, _ = extract_from_exif({_DTO: "not-a-date"}, min_year=1990, now=NOW)
        assert ts is None

    def test_malformed_gps_does_not_raise(self):
        exif = {_GPS: {1: "N", 2: "garbage", 3: "E", 4: (1.0, 2.0, 3.0)}}
        ts, lat, lon = extract_from_exif(exif, min_year=1990, now=NOW)
        assert lat is None  # bad triple -> None, no exception


class ut_GeoMetadataExtractor:
    """End-to-end read of real JPEG fixtures."""

    def _extractor(self):
        return GeoMetadataExtractor(min_year=1990, now=NOW)

    def test_reads_gps_and_time_from_file(self, tmp_path):
        p = _write_jpeg(tmp_path / "a.jpg", dt="2024:06:15 10:30:00",
                        lat=47.5, lon=19.05)
        meta = self._extractor().read_one(p)
        assert meta.has_geo and meta.has_time
        assert abs(meta.lat - 47.5) < 1e-4
        assert abs(meta.lon - 19.05) < 1e-4
        assert meta.timestamp == datetime(2024, 6, 15, 10, 30, 0)

    def test_no_exif_file_degrades_to_none(self, tmp_path):
        p = tmp_path / "plain.jpg"
        Image.new("RGB", (8, 8), (10, 20, 30)).save(p)
        meta = self._extractor().read_one(str(p))
        assert not meta.has_geo and not meta.has_time
        assert meta.lat is None and meta.lon is None and meta.timestamp is None

    def test_missing_file_does_not_raise(self, tmp_path):
        meta = self._extractor().read_one(str(tmp_path / "nope.jpg"))
        assert not meta.has_geo and not meta.has_time

    def test_calc_returns_dict_keyed_by_path(self, tmp_path):
        a = _write_jpeg(tmp_path / "a.jpg", dt="2024:06:15 10:30:00",
                        lat=47.5, lon=19.05)
        b = _write_jpeg(tmp_path / "b.jpg", dt="2024:06:16 09:00:00")  # time only
        result = self._extractor().calc(ExifInputs(image_paths=[a, b]))
        assert set(result.metadata.keys()) == {a, b}
        assert result.metadata[a].has_geo
        assert not result.metadata[b].has_geo and result.metadata[b].has_time
