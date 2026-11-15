"""Names the layers must agree on because they are written into a session's files.

These are not settings. The session directory (a working copy and a checkpoint share one layout) and `session.json`
hold these names, so renaming any of them needs a migration of saved sessions; nothing here is meant to be changed
or configured. Sessions, video and tracking all read and write the same files, so the names live here, below all three.
"""

# The folders and files of a session directory.
FRAMES_DIR = "frames"
MASKS_DIR = "masks"
TRACKING_DIR = "tracking"
SESSION_FILE = "session.json"
# Inside MASKS_DIR: the masks the user drew, the mask each click showed, one propagation result per manifest file,
# and the overlay images of a propagation.
OVERRIDES_DIR = "overrides"
INTERACTIVE_DIR = "interactive"
MANIFEST_PREFIX = "manifest-"
OVERLAY_DIR_PREFIX = "overlays_"

# Session-metadata entries that name the latest tracking result (the result file lives in TRACKING_DIR).
TRACKING_RESULT_ID = "latest_tracking_result_id"
TRACKING_RESULT_PATH = "latest_tracking_result_path"
TRACKING_RESULT_UPDATED_AT = "latest_tracking_result_updated_at"
TRACKING_RESULT_SUMMARY = "latest_tracking_result_summary"
TRACKING_METADATA_KEYS = (TRACKING_RESULT_ID, TRACKING_RESULT_PATH, TRACKING_RESULT_UPDATED_AT,
                          TRACKING_RESULT_SUMMARY)
