Fixed VideoCacheStats.newest_mtime precision bug in _iter_video_cache_entries().

Changed st_mtime (float, second-level precision) to st_mtime_ns (integer, nanosecond precision)
divided by 1e9 to produce a float, providing higher precision timestamps.

Only app/services/cache_manager.py was modified as requested.