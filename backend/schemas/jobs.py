from typing import Literal

JobStatus = Literal["queued", "running", "completed", "failed"]
JobOperation = Literal["video_init", "mask_propagation", "prompt_tracking"]
