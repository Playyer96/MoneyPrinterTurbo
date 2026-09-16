import warnings
from enum import Enum
from typing import Any, List, Literal, Optional, Union

import pydantic
from pydantic import BaseModel, ConfigDict, Field

from app.config import config
from app.models import const

# Silence one specific Pydantic warning.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Field name.*shadows an attribute in parent.*",
)


class VideoConcatMode(str, Enum):
    random = "random"
    sequential = "sequential"


class VideoTransitionMode(str, Enum):
    none = None
    shuffle = "Shuffle"
    fade_in = "FadeIn"
    fade_out = "FadeOut"
    slide_in = "SlideIn"
    slide_out = "SlideOut"
    zoom_in = "ZoomIn"
    zoom_out = "ZoomOut"


class VideoAspect(str, Enum):
    landscape = "16:9"
    portrait = "9:16"
    square = "1:1"

    def to_resolution(self):
        if self == VideoAspect.landscape:
            return 1920, 1080
        elif self == VideoAspect.portrait:
            return 1080, 1920
        elif self == VideoAspect.square:
            return 1080, 1080
        raise ValueError(f"unsupported video aspect: {self}")


class VideoFitMode(str, Enum):
    """How source clips with a different aspect ratio fill the output canvas."""

    cover = "cover"
    contain = "contain"


SubtitleDisplayMode = Literal[
    "sentence",
    "word_by_word",
    "two_words",
    "three_words",
    "progressive",
    "karaoke",
]
SubtitleAnimation = Literal[
    "none",
    "pop_spring",
    "scale_up",
    "fade",
    "slide_up",
    "shake",
]
_SUBTITLE_DISPLAY_MODES = (
    "sentence",
    "word_by_word",
    "two_words",
    "three_words",
    "progressive",
    "karaoke",
)
_SUBTITLE_ANIMATIONS = (
    "none",
    "pop_spring",
    "scale_up",
    "fade",
    "slide_up",
    "shake",
)


def _get_valid_ui_choice(key: str, allowed_values: tuple[str, ...], default: str) -> str:
    """
    Read a validated WebUI enum setting, tolerating invalid values left by
    older installs.

    Request bodies are validated strictly by Pydantic's Literal, so a typo
    returns a clear field validation error. Config files need the forgiving
    path instead: a hand-edited legacy value must not stop the whole service
    from starting after an upgrade. The HTTP status code is decided by the
    application's shared validation exception handler, not pinned here.
    """
    configured_value = config.ui.get(key, default)
    return configured_value if configured_value in allowed_values else default


_Config = ConfigDict(
    arbitrary_types_allowed=True,
    # Note: ensure your key names match renamed V2 parameters if needed
)


@pydantic.dataclasses.dataclass(config=_Config)
class MaterialInfo:
    provider: str = "pexels"
    url: str = ""
    duration: int = 0
    # Online material searches carry filtered public source info, reused by the
    # search cache and the task record. Locally uploaded materials leave it
    # empty; the value is rebuilt from a field allowlist before it is written to
    # the task file, so signed URLs, credentials, or unrelated fields sent by an
    # external request never reach persisted data.
    source_info: Optional[dict[str, Any]] = None


class VideoParams(BaseModel):
    """
    {
      "video_subject": "",
      "video_aspect": "landscape 16:9 (Xigua Video)",
      "voice_name": "",
      "bgm_name": "random",
      "font_name": "STHeitiMedium (Heiti SC Medium)",
      "text_color": "#FFFFFF",
      "font_size": 60,
      "stroke_color": "#000000",
      "stroke_width": 1.5
    }
    """

    video_subject: str
    video_script: str = ""  # Script used to generate the video
    video_terms: Optional[str | list] = None  # Keywords used to generate the video
    video_aspect: Optional[VideoAspect] = VideoAspect.portrait.value
    video_fit_mode: VideoFitMode = VideoFitMode.cover
    video_concat_mode: Optional[VideoConcatMode] = VideoConcatMode.random.value
    video_transition_mode: Optional[VideoTransitionMode] = None
    # ``video_clip_duration`` is the max seconds per material slice. When
    # ``video_clip_duration_auto`` is true, the pipeline derives a value from
    # the audio length (around audio/10s, clamped to [2, 10]) instead of
    # honouring ``video_clip_duration``. Both kept on the model so the API
    # surface stays stable while the WebUI's "Auto" option only flips the
    # boolean.
    video_clip_duration: int = Field(default=5, ge=1)
    video_clip_duration_auto: bool = False
    video_clip_speed: Optional[float] = 1.0
    match_materials_to_script: bool = False
    video_count: int = Field(default=1, ge=1)

    video_source: Optional[str] = "pexels"
    video_materials: Optional[List[MaterialInfo]] = (
        None  # Materials used to generate the video
    )

    custom_audio_file: Optional[str] = (
        None  # Custom audio file path, will ignore TTS and can still use Whisper subtitles
    )
    video_language: Optional[str] = ""  # auto detect

    voice_name: Optional[str] = ""
    voice_volume: Optional[float] = 1.0
    voice_rate: Optional[float] = 1.0
    # Free-text emotion / delivery instruction forwarded to TTS providers
    # that accept one (today: OmniVoice). Other providers silently ignore it.
    voice_style: Optional[str] = ""
    bgm_type: Optional[str] = "random"
    bgm_file: Optional[str] = ""
    bgm_volume: Optional[float] = 0.2
    # When True, the source video clip's audio track is preserved in the final
    # mix instead of being stripped. Combine with the existing voice_name /
    # bgm_type fields to compose: voice-over-original, BGM-under-original,
    # or original-only (no voiceover, no BGM). Default False preserves the
    # original "voice + optional BGM, source audio discarded" behaviour.
    keep_original_audio: bool = False
    # Volume scale applied to the preserved source audio track. Honoured
    # only when ``keep_original_audio`` is True. Defaults to 1.0 so the
    # original audio plays at its native level.
    original_audio_volume: float = Field(default=1.0, ge=0.0)
    # Shared music prompt across video music providers; new WebUI tasks always
    # write this field. The Sonilo-specific field below stays for compatibility
    # with old task records and the existing CLI argument.
    video_music_prompt: str = Field(default="", max_length=2000)
    sonilo_bgm_prompt: str = Field(default="", max_length=2000)

    subtitle_enabled: Optional[bool] = True
    # "srt" keeps the legacy subtitle format; "ass" writes an Advanced
    # SubStation Alpha file alongside the SRT so the user can edit the
    # on-disk captions directly; "both" writes both files (default).
    subtitle_format: Optional[str] = "both"
    subtitle_position: Optional[str] = config.ui.get(
        "subtitle_position", "bottom"
    )  # top, bottom, center, custom, two_thirds_bottom
    subtitle_display_mode: SubtitleDisplayMode = _get_valid_ui_choice(
        "subtitle_display_mode", _SUBTITLE_DISPLAY_MODES, "sentence"
    )
    subtitle_animation: SubtitleAnimation = _get_valid_ui_choice(
        "subtitle_animation", _SUBTITLE_ANIMATIONS, "none"
    )
    subtitle_style_preset: Optional[str] = config.ui.get(
        "subtitle_style_preset", "custom"
    )
    subtitle_casing: Optional[str] = config.ui.get("subtitle_casing", "as_is")
    custom_position: float = config.ui.get("custom_position", 70.0)
    font_name: Optional[str] = "STHeitiMedium.ttc"
    text_fore_color: Optional[str] = "#FFFFFF"
    text_background_color: Union[bool, str] = False
    rounded_subtitle_background: bool = False

    font_size: int = 60
    stroke_color: Optional[str] = "#000000"
    stroke_width: float = 1.5

    # Raw ASS overrides. Accept full ``[V4+ Styles]`` blocks, per-event
    # override tag strings (``\\an2\\pos(x,y)\\fscx105\\t(...)``), or named
    # transform primitives (shadow / blur / rotation). Falsy values leave
    # the rendered cue on the base style. Sized to a few hundred KB so a
    # user can paste any reasonable ASS snippet without truncation.
    subtitle_ass_style_override: Optional[str] = ""
    subtitle_ass_event_overrides: Optional[str] = ""
    subtitle_ass_shadow: Optional[float] = 0
    subtitle_ass_blur: Optional[float] = 0
    subtitle_ass_rotation: Optional[float] = 0
    subtitle_ass_background_color: Optional[str] = ""
    n_threads: Optional[int] = 2
    paragraph_number: int = Field(default=1, ge=1, le=10)
    video_script_prompt: str = Field(default="", max_length=2000)
    custom_system_prompt: str = Field(default="", max_length=8000)

    # Per-paragraph delivery cues. ``None`` means "no cues were generated", an
    # empty list means "cues were requested but the model produced none", and a
    # populated list has one entry per paragraph of ``video_script``.
    paragraph_cues: Optional[List[str]] = None
    delivery_cues_enabled: bool = False

    # Video title / hook banner overlay settings
    title_enabled: bool = False
    title_text: Optional[str] = ""
    title_style: Optional[str] = "tiktok_yellow"
    title_position: Optional[str] = "top"
    title_duration: Optional[str] = "intro"
    title_animation: Optional[str] = "pop_spring"
    title_font_name: Optional[str] = None
    title_font_size: Optional[int] = None

    # Intro / outro blurred overlay. Renders a heavily blurred full-screen
    # background with multi-line centered text, prepended or appended to the
    # final video. Both are optional and disabled by default to preserve the
    # legacy behaviour. ``intro_text`` / ``outro_text`` accept multi-line text
    # via ``\n``; each non-empty line becomes its own line on the overlay.
    # Intro / outro overlay defaults to ON. The previous opt-in default let
    # users forget to enable them, so the overlay was missing in real
    # renders even though the UI panel was exposed. Now ON with default text
    # that uses the video_subject (and the first series_outline entry, if
    # series is on); users can still disable or override the text per task.
    intro_enabled: bool = True
    intro_text: Optional[str] = ""
    intro_duration: float = Field(default=5.0, ge=0.5, le=15.0)
    intro_blur_strength: int = Field(default=35, ge=5, le=120)
    intro_text_color: Optional[str] = "#FFFFFF"
    intro_animation: Optional[str] = "fade"
    intro_tts_enabled: bool = True
    outro_enabled: bool = True
    outro_text: Optional[str] = ""
    outro_duration: float = Field(default=6.0, ge=0.5, le=15.0)
    outro_blur_strength: int = Field(default=35, ge=5, le=120)
    outro_text_color: Optional[str] = "#FFFFFF"
    outro_animation: Optional[str] = "fade"
    outro_tts_enabled: bool = True

    # Series mode: turn one subject into an ordered set of chapter videos.
    # ``series_parts`` is the single count input: 0 lets the model decide how
    # many chapters the subject actually needs, any other value pins it.
    series_enabled: bool = False
    series_parts: int = Field(default=0, ge=0, le=const.MAX_SERIES_PARTS)
    # Pre-approved chapter subjects. Empty means the outline is generated when
    # the task runs, so the WebUI can stay optional.
    series_outline: List[str] = Field(default_factory=list)
    # Tell each part which chapters come before and after it, so the series
    # reads as one arc instead of unrelated videos on the same subject.
    series_continuity: bool = True


class SubtitleRequest(BaseModel):
    video_script: str
    video_language: Optional[str] = ""
    voice_name: Optional[str] = "zh-CN-XiaoxiaoNeural-Female"
    voice_volume: Optional[float] = 1.0
    voice_rate: Optional[float] = 1.2
    voice_style: Optional[str] = ""
    bgm_type: Optional[str] = "random"
    bgm_file: Optional[str] = ""
    bgm_volume: Optional[float] = 0.2
    subtitle_position: Optional[str] = config.ui.get("subtitle_position", "bottom")
    subtitle_display_mode: SubtitleDisplayMode = _get_valid_ui_choice(
        "subtitle_display_mode", _SUBTITLE_DISPLAY_MODES, "sentence"
    )
    subtitle_animation: SubtitleAnimation = _get_valid_ui_choice(
        "subtitle_animation", _SUBTITLE_ANIMATIONS, "none"
    )
    subtitle_style_preset: Optional[str] = "custom"
    subtitle_casing: Optional[str] = "as_is"
    font_name: Optional[str] = "STHeitiMedium.ttc"
    text_fore_color: Optional[str] = "#FFFFFF"
    text_background_color: Union[bool, str] = False
    rounded_subtitle_background: bool = False
    font_size: int = 60
    stroke_color: Optional[str] = "#000000"
    stroke_width: float = 1.5
    subtitle_format: Optional[str] = "both"
    subtitle_ass_style_override: Optional[str] = ""
    subtitle_ass_event_overrides: Optional[str] = ""
    subtitle_ass_shadow: Optional[float] = 0
    subtitle_ass_blur: Optional[float] = 0
    subtitle_ass_rotation: Optional[float] = 0
    subtitle_ass_background_color: Optional[str] = ""
    video_source: Optional[str] = "local"
    subtitle_enabled: Optional[str] = "true"


class AudioRequest(BaseModel):
    video_script: str
    video_language: Optional[str] = ""
    voice_name: Optional[str] = "zh-CN-XiaoxiaoNeural-Female"
    voice_volume: Optional[float] = 1.0
    voice_rate: Optional[float] = 1.2
    bgm_type: Optional[str] = "random"
    bgm_file: Optional[str] = ""
    bgm_volume: Optional[float] = 0.2
    video_source: Optional[str] = "local"


class VideoScriptParams:
    """
    {
      "video_subject": "Spring flowers in bloom",
      "video_language": "",
      "paragraph_number": 1,
      "video_script_prompt": "",
      "custom_system_prompt": "",
      "delivery_cues_enabled": false
    }
    """

    video_subject: Optional[str] = "Spring flowers in bloom"
    video_language: Optional[str] = ""
    paragraph_number: int = Field(default=1, ge=1, le=10)
    video_script_prompt: str = Field(default="", max_length=2000)
    custom_system_prompt: str = Field(default="", max_length=8000)
    # When True, the LLM emits a `[Cue: <delivery>]` line before each paragraph
    # and the response payload includes `paragraph_cues`. The audio stage can
    # then forward each cue as a per-paragraph voice_style to TTS providers that
    # accept one (today: OmniVoice).
    delivery_cues_enabled: bool = False


class VideoTermsParams:
    """
    {
      "video_subject": "",
      "video_script": "",
      "amount": 5,
      "match_materials_to_script": false
    }
    """

    video_subject: Optional[str] = "Spring flowers in bloom"
    video_script: Optional[str] = (
        "The spring sea of flowers unfolds like a picturesque painting before our eyes. In the season of rebirth, nature puts on a vibrant and colorful gown. Golden forsythia, soft pink cherry blossoms, pure white pear blossoms, and radiant tulips bloom together in harmony..."
    )
    amount: Optional[int] = 5
    match_materials_to_script: bool = False


class VideoSocialMetadataParams:
    """
    {
      "video_subject": "A day in Shanghai",
      "video_script": "",
      "language": "auto",
      "platform": "tiktok"
    }
    """

    video_subject: Optional[str] = Field(default="A day in Shanghai", max_length=500)
    video_script: Optional[str] = Field(default="", max_length=8000)
    language: Optional[str] = Field(default="auto", max_length=64)
    platform: Optional[str] = Field(default="tiktok", max_length=64)


class TaskVideoRequest(VideoParams, BaseModel):
    pass


class TaskQueryRequest(BaseModel):
    pass


class VideoScriptRequest(VideoScriptParams, BaseModel):
    pass


class VideoTermsRequest(VideoTermsParams, BaseModel):
    pass


class VideoSocialMetadataRequest(VideoSocialMetadataParams, BaseModel):
    pass


# ---------------------------
# ----- RESPONSE MODELS -----
# ---------------------------
class BaseResponse(BaseModel):
    status: int = 200
    message: Optional[str] = "success"
    data: Any = None


# ---- DATA MODELS ----
class TaskResponseData(BaseModel):
    task_id: str


class TaskStatusData(BaseModel):
    """Stable fields guaranteed by the task query; legacy and extra fields pass through unchanged."""

    model_config = ConfigDict(extra="allow")

    task_id: str
    state: int
    progress: int = 0
    videos: Optional[List[str]] = None
    combined_videos: Optional[List[str]] = None
    failed_stage: Optional[str] = None
    error: Optional[str] = None
    cross_post_state: Optional[
        Literal["pending", "processing", "complete", "failed"]
    ] = None
    cross_post_results: Optional[List[dict[str, Any]]] = None
    cross_post_error: Optional[str] = None


class TaskListData(BaseModel):
    """Paginated task list payload."""

    tasks: List[TaskStatusData]
    total: int
    page: int
    page_size: int


class VideoScriptData(BaseModel):
    video_script: str
    # Per-paragraph delivery cues returned by the script generator. Empty
    # strings fall back to the configured `voice_style`. Length matches the
    # paragraph count in `video_script` when delivery cues are enabled.
    paragraph_cues: Optional[List[str]] = None


class VideoTermsData(BaseModel):
    video_terms: List[str]


class VideoSocialMetadataData(BaseModel):
    title: str
    caption: str
    hashtags: List[str]


class FileData(BaseModel):
    name: str
    size: int
    file: str


class BgmRetrieveData(BaseModel):
    files: List[FileData]


class BgmUploadData(BaseModel):
    file: str


class VideoMaterialRetrieveData(BaseModel):
    files: List[FileData]


class VideoMaterialUploadData(BaseModel):
    file: str


# ---- RESPONSE MODELS ----
class TaskResponse(BaseResponse):
    data: TaskResponseData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "task_id": "6c85c8cc-a77a-42b9-bc30-947815aa0558",
                },
            },
        }
    )


class TaskQueryResponse(BaseResponse):
    """
    The task query returns the generation state plus an optional
    cross-platform publishing state.

    A failed generation includes `failed_stage` and `error`. When auto publish
    is enabled, `cross_post_state` moves through pending, processing, and then
    complete or failed once generation finishes.
    """

    data: TaskStatusData

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": 200,
                    "message": "success",
                    "data": {
                        "task_id": "6c85c8cc-a77a-42b9-bc30-947815aa0558",
                        "state": 1,
                        "progress": 100,
                        "videos": ["/tasks/example/final-1.mp4"],
                        "cross_post_state": "complete",
                        "cross_post_results": [{"success": True}],
                    },
                },
                {
                    "status": 200,
                    "message": "success",
                    "data": {
                        "task_id": "6c85c8cc-a77a-42b9-bc30-947815aa0558",
                        "state": -1,
                        "progress": 30,
                        "failed_stage": "audio",
                        "error": "TTS request timed out",
                    },
                },
            ],
        }
    )


class TaskListResponse(BaseResponse):
    """The task list has its own response model, so its schema stays separate from the single-task query."""

    data: TaskListData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "tasks": [
                        {
                            "task_id": "6c85c8cc-a77a-42b9-bc30-947815aa0558",
                            "state": 4,
                            "progress": 50,
                        }
                    ],
                    "total": 1,
                    "page": 1,
                    "page_size": 10,
                },
            }
        }
    )


class TaskDeletionResponse(BaseResponse):
    data: None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": None,
            },
        }
    )


class VideoScriptResponse(BaseResponse):
    data: VideoScriptData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "video_script": "The spring sea of flowers is nature's beautiful canvas. In this season, the earth revives, all living things grow, and flowers bloom in competition, creating a magnificent tapestry of colors..."
                },
            },
        }
    )


class VideoTermsResponse(BaseResponse):
    data: VideoTermsData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {"video_terms": ["sky", "tree"]},
            },
        }
    )


class VideoSocialMetadataResponse(BaseResponse):
    data: VideoSocialMetadataData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "title": "A Day in Shanghai You Should Not Miss",
                    "caption": "Save this quick Shanghai inspiration and follow for more short travel ideas.",
                    "hashtags": ["#shorts", "#travel", "#shanghai", "#viral", "#fyp"],
                },
            },
        }
    )


class BgmRetrieveResponse(BaseResponse):
    data: BgmRetrieveData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "files": [
                        {
                            "name": "4fca18fce7344f3aa824777a40d45c8c.mp3",
                            "size": 1891269,
                            "file": "4fca18fce7344f3aa824777a40d45c8c.mp3",
                        }
                    ]
                },
            },
        }
    )


class BgmUploadResponse(BaseResponse):
    data: BgmUploadData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {"file": "4fca18fce7344f3aa824777a40d45c8c.mp3"},
            },
        }
    )


class VideoMaterialRetrieveResponse(BaseResponse):
    data: VideoMaterialRetrieveData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "files": [
                        {
                            "name": "example.mp4",
                            "size": 12345678,
                            "file": "/MoneyPrinterTurbo/resource/videos/example.mp4",
                        }
                    ]
                },
            },
        }
    )


class VideoMaterialUploadResponse(BaseResponse):
    data: VideoMaterialUploadData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "file": "/MoneyPrinterTurbo/resource/videos/example.mp4",
                },
            },
        }
    )
