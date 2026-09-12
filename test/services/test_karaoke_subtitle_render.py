from pathlib import Path

import numpy as np

from app.services import subtitle_styles
from app.services import video
from app.utils import utils


def test_karaoke_renderer_draws_normal_and_active_word_colors():
    preset = subtitle_styles.SUBTITLE_PRESETS["binance_karaoke"]
    clip = video._render_highlighted_subtitle_clip(
        f"Trade {subtitle_styles.HIGHLIGHT_OPEN}smarter{subtitle_styles.HIGHLIGHT_CLOSE}",
        font_path=str(Path(utils.font_dir()) / preset["font_name"]),
        font_size=48,
        max_width=700,
        text_color="#FFFFFF",
        highlight_color="#F0B90B",
        stroke_color="#000000",
        stroke_width=2,
        background_color=None,
        rounded_background=False,
    )

    try:
        colors = clip.get_frame(0).reshape(-1, 3)
        assert np.any(np.all(colors == (255, 255, 255), axis=1))
        assert np.any(np.all(colors == (240, 185, 11), axis=1))
    finally:
        video.close_clip(clip)
