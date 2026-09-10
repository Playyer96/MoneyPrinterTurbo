PUNCTUATIONS = [
    "?",
    ",",
    ".",
    "、",
    ";",
    ":",
    "!",
    "…",
    "？",
    "，",
    "。",
    "、",
    "；",
    "：",
    "！",
    "...",
    # Common Arabic punctuation must count as a natural sentence break too,
    # otherwise the script text and the pause boundaries edge-tts returns drift
    # apart and the later line-by-line matching fails.
    "،",
    "؛",
    "؟",
]

# Series mode ceiling. Automatic mode lets the model pick the chapter count, so
# this is only a runaway guard against a malformed response, not a feature cap.
MAX_SERIES_PARTS = 100

TASK_STATE_FAILED = -1
TASK_STATE_COMPLETE = 1
TASK_STATE_PROCESSING = 4

CROSS_POST_STATE_PENDING = "pending"
CROSS_POST_STATE_PROCESSING = "processing"
CROSS_POST_STATE_COMPLETE = "complete"
CROSS_POST_STATE_FAILED = "failed"

FILE_TYPE_VIDEOS = ["mp4", "mov", "mkv", "webm"]
FILE_TYPE_IMAGES = ["jpg", "jpeg", "png", "bmp"]
