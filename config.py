# config.py
import os

# --- Paths ---
# Base path for your dataset.
# IMPORTANT: This path needs to be correct for your system.
# It was "E:\sem 6\face recg wts\compact feed\images" in your original Gui_withdeeep.py. 
DATASET_BASE_PATH = r"E:\sem 6\face recg wts\changingview\images"

# Other data file paths. These names are derived from your existing scripts.
LOG_FILE_PATH = "entry_log_dataset2.txt"  # Used in gpt31standalone.py 
ENCODINGS_FILE = "encodings1.pkl"         # Used in gpt31standalone.py , delete_person.py , update_face_encodings.py 
INTRUDERS_FOLDER = "intruders"            # Used in gpt31standalone.py 
INTRUDER_ENCODINGS_FILE = "intruder_encodings.pkl" # Used in intruder_tracker.py 


# --- Debounce Times (in seconds) ---
# Time in seconds for capturing intruder images and logging "Unknown" faces.
UNKNOWN_CAPTURE_DEBOUNCE_TIME = 10 # From gpt31standalone.py 
# Time in seconds per person for logging known faces.
KNOWN_FACE_LOG_DEBOUNCE_TIME = 25   # From gpt31standalone.py 

# --- Face Recognition Settings ---
# 'hog' is faster but less accurate; 'cnn' is slower but more accurate.
FACE_DETECTION_MODEL = "hog" # From gpt31standalone.py 
# Matching tolerance (lower = stricter match, fewer false positives).
RECOGNITION_TOLERANCE = 0.6 # From gpt31standalone.py 

# --- Camera Performance Settings (NEW SECTION) ---
# Desired camera frame width and height. Note: IP cameras might override these.
# Common options: (320, 240) - Low, (640, 480) - Medium, (1280, 720) - High
CAMERA_FRAME_WIDTH = 320 # Default to your current setting for consistency
CAMERA_FRAME_HEIGHT = 240 # Default to your current setting for consistency

# Number of frames to skip before processing (face detection/recognition).
# 1 = process every frame, 2 = process every 2nd frame, 4 = process every 4th frame (your current)
FRAME_PROCESS_SKIP_RATE = 4 # Default to your current setting

# --- Email Configuration ---
EMAIL_CONFIG = { # From gpt31standalone.py 
    'sender': 'kewalprojects@gmail.com',
    'password': 'ttci bdap rfzh yefw', # Remember to use an App Password if using Gmail
    'receiver': 'techexplorer433@gmail.com',
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'subject': ' INTRUDER ALERT - kewal Face Recognition System'
}

# --- Camera Configuration ---
# For USB webcams, add their numerical indices here (e.g., [0, 1]).
# If you have a built-in webcam, it's usually 0.
USB_WEBCAM_INDICES = [] # Add your USB camera indices here, e.g., [0]

# For IP cameras, add their full stream URLs (RTSP, HTTP, etc.).
IP_CAMERA_URLS = [ # From gpt31standalone.py 
    "http://192.168.6.21:8080/video",  # Example: IP Camera 1
    #"http://192.168.6.252:8080/video",  # Example: IP Camera 1 
    
]

# --- Ensure necessary folders exist ---
# This creates the 'intruders' folder if it doesn't already exist. 
# It's good practice to ensure output directories are ready.
os.makedirs(INTRUDERS_FOLDER, exist_ok=True)