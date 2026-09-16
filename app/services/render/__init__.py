"""Render package: the pieces of the final video render extracted from
``app/services/video.py`` one module at a time.

Every module here is imported by ``video.py`` and re-exported under the old
function names, so callers keep working while the monolith shrinks.
"""
