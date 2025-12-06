from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = "Image"
VIDEO = "Video"
RTSP = "RTSP"
YOUTUBE = "YouTube"

SOURCES_LIST = [IMAGE, VIDEO, RTSP, YOUTUBE]

# Images config
IMAGES_DIR = ROOT / "images"
DEFAULT_IMAGE = IMAGES_DIR / "WIN_20220419_21_14_35_Pro.jpg"
DEFAULT_DETECT_IMAGE = (
    IMAGES_DIR / "d333b6ab25857aa73e9c942a0c950df99f775de8e3b4b2a838491c33.jpg"
)

# Videos config
VIDEO_DIR = ROOT / "videos"
DEFAULT_VIDEO = VIDEO_DIR / "video_1.mp4"
DEFAULT_DETECT_VIDEO = VIDEO_DIR / "object_counting_output.mp4"

# ML Model config
MODEL_DIR = ROOT / "weights"
DETECTION_MODEL = MODEL_DIR / "last.pt"
# In case of your custome model comment out the line above and
# Place your custom model pt file name at the line below
# DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'
