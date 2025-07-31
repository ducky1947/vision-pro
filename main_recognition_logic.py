# main_recognition_logic.py
import cv2
import threading # For managing the CameraStream threads
import queue # For passing frames between CameraStream and display logic
import sys
import time # For sleep and overall timing
import select # For non-blocking console input in standalone test mode (Linux/macOS)
import os # To check for dummy camera existence in test mode

# Import the newly created/refactored modules
import config # Centralized configurations
import face_data_manager # For loading known faces
import logging_manager # For ensuring log header is present
from camera_stream import CameraStream # The refactored CameraStream class

# --- Global Data (for this module, to pass to CameraStream instances) ---
# These will be loaded once when the recognition logic starts.
# They are declared here to hold the data loaded by face_data_manager.
_KNOWN_FACE_ENCODINGS = []
_KNOWN_FACE_NAMES = []
_KNOWN_FACE_GENDERS = []

# --- Main Recognition System Logic ---
def start_live_face_recognition(global_stop_event, camera_display_queues):
    """
    Starts the live face recognition system across multiple cameras.
    This function is designed to be run in a separate thread (e.g., a QThread from the GUI).

    Args:
        global_stop_event (threading.Event): An event object to signal all camera streams to stop gracefully.
        camera_display_queues (dict): A dictionary where keys are camera inputs (int/str)
                                      and values are queue.Queue objects. CameraStream instances will
                                      put processed frames into these queues for a display thread to retrieve.
                                      This module does NOT manage OpenCV windows itself.
    """
    global _KNOWN_FACE_ENCODINGS, _KNOWN_FACE_NAMES, _KNOWN_FACE_GENDERS

    # IMPORTANT: Tell OpenCV not to manage its own threads.
    # This can prevent conflicts when you're using Python's threading.
    cv2.setNumThreads(0)
    print("DEBUG: main_recognition_logic: cv2.setNumThreads(0) called.", flush=True)

    print("DEBUG: main_recognition_logic: Starting application core.", flush=True)

    # Load known faces once at startup (delegated to face_data_manager module)
    _KNOWN_FACE_ENCODINGS, _KNOWN_FACE_NAMES, _KNOWN_FACE_GENDERS = face_data_manager.load_known_face_encodings()
    
    # Ensure the log file header is written once (delegated to logging_manager module)
    logging_manager.write_log_header_if_needed()

    if not _KNOWN_FACE_ENCODINGS:
        print("SYSTEM: main_recognition_logic: No known faces loaded. The recognition system will only detect 'Unknown' faces.", flush=True)

    # --- Camera Configuration and Verification ---
    available_cameras_inputs = []
    # Add USB webcams defined in config.py
    available_cameras_inputs.extend(config.USB_WEBCAM_INDICES)
    # Add IP cameras defined in config.py
    available_cameras_inputs.extend(config.IP_CAMERA_URLS)

    if not available_cameras_inputs:
        print("ERROR: main_recognition_logic: No cameras configured in config.py. Exiting live face recognition system.", flush=True)
        global_stop_event.set() # Signal immediate stop if no cameras are configured
        return

    final_active_camera_inputs = [] # List to store inputs for cameras that successfully open
    print("\nDEBUG: main_recognition_logic: Verifying configured cameras...", flush=True)
    for cam_input in available_cameras_inputs:
        # Temporarily attempt to open each camera to check if it's accessible
        temp_cap = cv2.VideoCapture(cam_input)
        if temp_cap.isOpened():
            print(f"INFO: main_recognition_logic: Camera '{cam_input}' detected and opened successfully.", flush=True)
            final_active_camera_inputs.append(cam_input)
            temp_cap.release() # Release the temporary capture object
        else:
            print(f"ERROR: main_recognition_logic: Camera '{cam_input}' could not be opened. Skipping this camera.", flush=True)
            # This could be due to wrong URL, credentials, network issues, or camera already in use.
            
    if not final_active_camera_inputs:
        print("ERROR: main_recognition_logic: No active cameras found after verification. Exiting live face recognition system.", flush=True)
        global_stop_event.set() # Signal immediate stop if no active cameras were found
        return

    # Verify email configuration for alert system (for debugging/info purposes)
    print("\nDEBUG: main_recognition_logic: Checking email alert configuration...", flush=True)
    if all(key in config.EMAIL_CONFIG and config.EMAIL_CONFIG[key] for key in ['sender', 'password', 'receiver']):
        print("DEBUG: main_recognition_logic: Email alerts ENABLED with valid configuration.", flush=True)
    else:
        print("WARNING: main_recognition_logic: Email alerts DISABLED - Missing required configuration (sender, password, or receiver).", flush=True)


    # Prepare a list to hold CameraStream objects, one for each active camera
    camera_streams = [] 

    # Create and start a CameraStream instance for each active camera
    print("\nDEBUG: main_recognition_logic: Initializing camera streams.", flush=True)
    for cam_input in final_active_camera_inputs:
        # Ensure a queue exists in the shared camera_display_queues dictionary for this camera input.
        # This queue will be used by the CameraStream to send frames to the display loop in GUI.
        if cam_input not in camera_display_queues:
            camera_display_queues[cam_input] = queue.Queue(maxsize=1) # maxsize=1 ensures only freshest frame is kept

        # Create CameraStream instance, passing it the necessary data and control objects
        stream = CameraStream(
            camera_input=cam_input, # The camera identifier (index or URL)
            known_encs=_KNOWN_FACE_ENCODINGS, # Loaded known face encodings
            known_names=_KNOWN_FACE_NAMES,   # Loaded known face names
            known_genders=_KNOWN_FACE_GENDERS, # Loaded known face genders
            stop_event=global_stop_event, # The shared event to signal all streams to stop
            frame_queue=camera_display_queues[cam_input] # The specific queue for this stream's output frames
        )
        camera_streams.append(stream) # Add the stream object to our list
        stream.start() # Start the reader and processing threads for this camera

    print("\nINFO: main_recognition_logic: Live multi-camera face recognition system core running. Waiting for stop signal.", flush=True)

    # This thread (where start_live_face_recognition runs) now simply waits
    # for the global_stop_event to be set. This allows all CameraStream threads
    # to run in the background without the main recognition logic actively doing work here.
    global_stop_event.wait() # Blocks until the stop event is set (by GUI or standalone test loop)

    print("INFO: main_recognition_logic: Stop signal received. Cleaning up camera threads...", flush=True)

    # Ensure all camera threads are properly terminated and their resources are released.
    for stream in camera_streams:
        stream.stop() # Signal each individual CameraStream object to stop its internal threads
        
        # Attempt to join (wait for) the reader thread to finish gracefully
        print(f"DEBUG: main_recognition_logic: Joining Camera {stream.camera_input} reader thread.", flush=True)
        if stream.reader_thread.is_alive(): # Check if thread is still running
            stream.reader_thread.join(timeout=5) # Wait for up to 5 seconds for it to finish
            if stream.reader_thread.is_alive():
                print(f"WARNING: main_recognition_logic: Camera {stream.camera_input} reader thread did not terminate gracefully within timeout (still alive).", flush=True)
            else:
                print(f"DEBUG: main_recognition_logic: Camera {stream.camera_input} reader thread joined successfully.", flush=True)
        else:
            print(f"DEBUG: main_recognition_logic: Camera {stream.camera_input} reader thread was already stopped.", flush=True)
        
        # Attempt to join (wait for) the processing thread to finish gracefully
        print(f"DEBUG: main_recognition_logic: Joining Camera {stream.camera_input} processing thread.", flush=True)
        if stream.processing_thread.is_alive(): # Check if thread is still running
            stream.processing_thread.join(timeout=5) # Wait for up to 5 seconds for it to finish
            if stream.processing_thread.is_alive():
                print(f"WARNING: main_recognition_logic: Camera {stream.camera_input} processing thread did not terminate gracefully within timeout (still alive).", flush=True)
            else:
                print(f"DEBUG: main_recognition_logic: Camera {stream.camera_input} processing thread joined successfully.", flush=True)
        else:
            print(f"DEBUG: main_recognition_logic: Camera {stream.camera_input} processing thread was already stopped.", flush=True)

    print("INFO: main_recognition_logic: All camera threads cleaned up. Live recognition logic finished.", flush=True)


# --- Standalone Test for main_recognition_logic.py (Non-Displaying) ---
# This block is executed only when main_recognition_logic.py is run directly.
# It verifies the background camera stream processing and data flow, but does NOT open OpenCV windows.
if __name__ == "__main__":
    print("\n--- Testing main_recognition_logic.py (Standalone - Non-Displaying) ---", flush=True)
    print("NOTE: This test runs the background recognition logic but will NOT show video feeds.", flush=True)
    print("ACTION: Wait for 15 seconds, then the process will attempt to stop automatically.", flush=True)
    print("Alternatively, press Ctrl+C in the console to force stop earlier.", flush=True)

    test_global_stop_event = threading.Event() # Event for signaling stop
    test_camera_display_queues = {} # Dummy dictionary, frames will be put here but not retrieved

    # Configure some test cameras if not already in config.py for testing purposes
    if not config.USB_WEBCAM_INDICES and not config.IP_CAMERA_URLS:
        print("WARNING: main_recognition_logic test: No cameras configured in config.py. Attempting to find a default camera for test.", flush=True)
        found_test_cam = False
        
        # Try USB webcam 0 first
        temp_cap_usb = cv2.VideoCapture(0)
        if temp_cap_usb.isOpened():
            config.USB_WEBCAM_INDICES.append(0) # Temporarily add to config for this test run
            temp_cap_usb.release()
            print("INFO: main_recognition_logic test: Using default USB webcam 0.", flush=True)
            found_test_cam = True
        else:
            # If USB 0 fails, try the IP camera URL you originally had
            test_ip_cam_url = "http://192.168.41.96:8080/video"
            temp_cap_ip = cv2.VideoCapture(test_ip_cam_url)
            if temp_cap_ip.isOpened():
                config.IP_CAMERA_URLS.append(test_ip_cam_url) # Temporarily add to config for this test run
                temp_cap_ip.release()
                print(f"INFO: main_recognition_logic test: Using IP camera {test_ip_cam_url}.", flush=True)
                found_test_cam = True
            else:
                print("ERROR: main_recognition_logic test: Could not find any active camera. Please configure one in config.py or check camera connection.", flush=True)

        if not found_test_cam:
            print("CRITICAL: Exiting standalone test as no active cameras were found for testing.", flush=True)
            sys.exit(1)

    # Start the main recognition logic in a separate thread
    recognition_thread = threading.Thread(
        target=start_live_face_recognition,
        args=(test_global_stop_event, test_camera_display_queues),
        daemon=True
    )
    recognition_thread.start()

    # --- Main thread simply waits for a duration or Ctrl+C ---
    test_duration_seconds = 15
    print(f"DEBUG: main_recognition_logic test: Running background threads for {test_duration_seconds} seconds.", flush=True)
    start_time = time.time()
    try:
        while (time.time() - start_time) < test_duration_seconds and recognition_thread.is_alive():
            time.sleep(0.5) # Periodically check time and thread status
    except KeyboardInterrupt:
        print("INFO: main_recognition_logic test: Ctrl+C pressed. Signaling stop.", flush=True)
    
    # Ensure stop signal is sent to background threads
    test_global_stop_event.set() 

    # --- Cleanup ---
    print("INFO: main_recognition_logic test: Signaling threads for final cleanup...", flush=True)
    if recognition_thread.is_alive():
        print("DEBUG: main_recognition_logic test: Waiting for recognition thread to join...", flush=True)
        recognition_thread.join(timeout=10) # Give it time to shut down
        if recognition_thread.is_alive():
            print("WARNING: main_recognition_logic test: Recognition thread did not terminate gracefully within timeout.", flush=True)
        else:
            print("DEBUG: main_recognition_logic test: Recognition thread joined successfully.", flush=True)
    else:
        print("DEBUG: main_recognition_logic test: Recognition thread was already stopped.", flush=True)

    # No cv2.destroyAllWindows() here as this block doesn't create windows.
    print("INFO: main_recognition_logic.py Standalone Test Complete. Application finished.", flush=True)