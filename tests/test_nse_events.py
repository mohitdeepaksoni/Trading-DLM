"""NSE event-feed helpers: broadcast timestamps in NSE's several formats, month chunking, numeric parsing."""
import datetime as dt
from emporion.data.loaders.nse_events import _ts, _months, _num


def test_timestamp_formats_seen_on_the_feeds():
    assert _ts("08-Sep-2026 16:32:23") == dt.datetime(2026, 9, 8, 16, 32, 23)      # pledge broadcastDt
    assert _ts("08-Sep-2026 12:35") == dt.datetime(2026, 9, 8, 12, 35)             # sast timestamp
    assert _ts("08-Sep-2026, 12-35") == dt.datetime(2026, 9, 8, 12, 35)            # sast 'time'
    assert _ts("01-SEP-2026") == dt.datetime(2026, 9, 1)                            # acquirerDate half
    assert _ts("2026-09-08 23:59:18") == dt.datetime(2026, 9, 8, 23, 59, 18)       # announcements sort_date
    assert _ts("18-Feb-2026 19:06") == dt.datetime(2026, 2, 18, 19, 6)             # insider 'date'
    assert _ts("") is None and _ts(None) is None


def test_month_chunks_and_numbers():
    ch = list(_months("2026-01-15", "2026-03-10"))
    assert ch == [(dt.date(2026, 1, 15), dt.date(2026, 1, 31)), (dt.date(2026, 2, 1), dt.date(2026, 2, 28)), (dt.date(2026, 3, 1), dt.date(2026, 3, 10))]
    assert _num("1,428,094") == 1428094.0 and _num("    20.02") == 20.02 and _num("-") is None and _num(None) is None
