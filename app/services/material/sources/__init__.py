"""Material source providers.

One module per external material source (stock video, AI generation). Each
provider exposes ``search_videos_*`` or ``generate_*`` with the same
``(search_term, minimum_duration, video_aspect)`` signature so the
``download_videos`` dispatcher can call them uniformly. Shared helpers
(api key rotation, TLS, aspect filter, secret redaction, persistence)
live in ``app.services.material._shared``.
"""

from __future__ import annotations
