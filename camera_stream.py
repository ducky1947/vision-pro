# camera_stream.py
import cv2
import face_recognition
import numpy as np
import datetime
import time
import threading
import queue
import os

# Import the newly created/refactored modules
import config # For all configuration settings
import logging_manager # For centralized logging operations
import intruder_tracker # For managing intruder data (now simplified)
import email_sender # For sending email alerts

# Global debounce timestamp for unknown faces. This MUST be a single, shared global variable,
# managed with a lock to be thread-safe across all CameraStream instances.
_last_unknown_capture_time = 0
_unknown_capture_debounce_lock = threading.Lock() # Lock for _last_unknown_capture_time

class CameraStream:
    """
    Manages a single camera feed in a separate thread for face detection and recognition.
    It includes an internal reader thread to prevent cap.read() from blocking the processing thread.
    Processed frames are put into a queue for display in the main thread (GUI).
    """
    def __init__(self, camera_input, known_encs, known_names, known_genders,
                 stop_event, frame_queue): 
        self.camera_input = camera_input
        self.cap = None # OpenCV VideoCapture object
        
        # Main processing thread for face recognition
        self.processing_thread = threading.Thread(target=self._run_processing_loop, name=f"CamProcThread-{camera_input}")
        
        # Internal reader thread for reading frames from the camera
        self.reader_thread = threading.Thread(target=self._run_reader_loop, name=f"CamReaderThread-{camera_input}")
        
        self.stop_event = stop_event # Event to signal all threads to stop gracefully
        self.frame_queue = frame_queue # Queue to send processed frames to the main thread for display
        self.internal_frame_queue = queue.Queue(maxsize=1) # Internal queue from reader to processing thread

        # References to shared/global resources for use within the thread
        self.known_face_encodings = known_encs
        self.known_face_names = known_names
        self.known_face_genders = known_genders
        self.recognition_tolerance = config.RECOGNITION_TOLERANCE # Access directly from config

        # Per-stream debounce for logging known faces. This is specific to each CameraStream instance.
        self.last_logged_times = {}

    def _run_reader_loop(self):
        """
        Internal thread loop for continuously reading frames from the camera.
        This prevents cap.read() from blocking the main processing thread.
        """
        print(f"DEBUG: Camera {self.camera_input}: Reader thread entered.", flush=True)
        
        self.cap = cv2.VideoCapture(self.camera_input)

        if not self.cap.isOpened():
            print(f"ERROR: Camera {self.camera_input}: Reader thread could not open webcam. Signaling global stop.", flush=True)
            self.stop_event.set() 
            return

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        print(f"DEBUG: Camera {self.camera_input}: Reader thread capture device opened.", flush=True)

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print(f"WARNING: Camera {self.camera_input}: Reader thread couldn't read frame. Attempting to re-open...", flush=True)
                self.cap.release()
                time.sleep(0.5) 
                self.cap = cv2.VideoCapture(self.camera_input) 
                if not self.cap.isOpened():
                    print(f"ERROR: Camera {self.camera_input}: Reader thread failed to re-open. Signaling global stop.", flush=True)
                    self.stop_event.set()
                    break 
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if isinstance(self.camera_input, int): 
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                continue
            
            try:
                if not self.internal_frame_queue.empty():
                    self.internal_frame_queue.get_nowait() 
                self.internal_frame_queue.put_nowait(frame) 
            except queue.Full:
                print(f"WARNING: Camera {self.camera_input}: Reader internal queue full. Dropping frame.", flush=True)
                pass 
            except Exception as e:
                print(f"ERROR: Camera {self.camera_input}: Reader thread queue operation error: {e}", flush=True)
            
            time.sleep(0.001) 

        print(f"DEBUG: Camera {self.camera_input}: Reader thread loop exited. Releasing camera.", flush=True)
        self.cap.release()
        print(f"DEBUG: Camera {self.camera_input}: Reader thread camera resource released.", flush=True)

    def _run_processing_loop(self):
        """
        The main loop for the camera processing thread.
        Gets frames from internal queue, performs face recognition, draws results, and puts frames into display queue.
        """
        print(f"DEBUG: Camera {self.camera_input}: Processing thread entered.", flush=True)
        
        frame_counter = 0 

        while not self.stop_event.is_set(): 
            if self.stop_event.is_set():
                print(f"DEBUG: Camera {self.camera_input}: Processing detected stop event at loop start. Breaking loop.", flush=True)
                break 

            frame = None
            try:
                frame = self.internal_frame_queue.get(timeout=0.05) 
            except queue.Empty:
                time.sleep(0.005) 
                continue 
            except Exception as e:
                print(f"ERROR: Camera {self.camera_input}: Processing thread frame retrieval error: {e}. Skipping frame.", flush=True)
                time.sleep(0.005) 
                continue

            if frame is None:
                print(f"WARNING: Camera {self.camera_input}: Processing received None frame. Skipping.", flush=True)
                time.sleep(0.005) 
                continue

            frame_counter += 1

            process_this_frame = (frame_counter % 4 == 0)

            drawn_frame = frame.copy() 

            if process_this_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                if rgb_small_frame is None or rgb_small_frame.size == 0:
                    print(f"WARNING: Camera {self.camera_input}: rgb_small_frame is invalid. Skipping face detection.", flush=True)
                    try:
                        self.frame_queue.put_nowait(drawn_frame) 
                    except queue.Full:
                        pass
                    time.sleep(0.005)
                    continue

                face_locations = []
                face_encodings = []
                try:
                    face_locations = face_recognition.face_locations(rgb_small_frame, model=config.FACE_DETECTION_MODEL)
                    
                    MAX_REASONABLE_FACES = 10 
                    if len(face_locations) > MAX_REASONABLE_FACES:
                        print(f"ERROR: Camera {self.camera_input}: Detected {len(face_locations)} faces (max {MAX_REASONABLE_FACES} allowed). Resetting to 0.", flush=True)
                        face_locations = []
                        continue
                        
                    valid_face_locations = []
                    min_face_size = 20  
                    for (top, right, bottom, left) in face_locations:
                        face_height = bottom - top
                        face_width = right - left
                        if face_height >= min_face_size and face_width >= min_face_size:
                            valid_face_locations.append((top, right, bottom, left))
                        else:
                            print(f"WARNING: Camera {self.camera_input}: Discarding too small face ({face_width}x{face_height}px) in frame {frame_counter}.", flush=True)
                    face_locations = valid_face_locations

                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                except Exception as e:
                    print(f"CRITICAL ERROR: Camera {self.camera_input}: Unhandled face detection/encoding exception: {e}. THIS IS LIKELY A STABILITY ISSUE.", flush=True)
                    self.stop_event.set()
                    try:
                        self.frame_queue.put_nowait(drawn_frame)
                    except queue.Full:
                        pass
                    time.sleep(0.005)
                    break 
                
                if self.stop_event.is_set():
                    print(f"DEBUG: Camera {self.camera_input}: Processing detected stop event after face recognition. Breaking loop.", flush=True)
                    break 

                faces_to_display_on_frame = []
                current_frame_has_unknown_intruder = False # Renamed for clarity

                for face_encoding in face_encodings:
                    recognized_gender_only = "Intruder" # Default if unknown
                    recognized_name_with_gender = "Unknown" # Default if unknown
                    recognized_person_name_only = "Unknown" # Default for display

                    if self.known_face_encodings: 
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.recognition_tolerance)
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                recognized_name_with_gender = self.known_face_names[best_match_index]
                                recognized_gender_only = self.known_face_genders[best_match_index]
                                recognized_person_name_only = recognized_name_with_gender.split('__')[0] \
                                                              if '__' in recognized_name_with_gender else recognized_name_with_gender

                    # If still unknown, it's an intruder
                    if recognized_name_with_gender == "Unknown":
                        current_frame_has_unknown_intruder = True # Flag that we saw an intruder in this frame

                        # --- Intruder Event Management (Capture, Log, Email - all debounced together) ---
                        # This section is executed ONLY IF an unknown face is detected in the current frame.
                        # The full event (capture, log, email) is controlled by debounce.

                        current_time_epoch = time.time()
                        
                        # This flag determines if a full intruder event should be triggered based on debounce
                        trigger_full_intruder_event = False 
                        
                        # Use the global debounce lock specifically for this full intruder event
                        global _last_unknown_capture_time, _unknown_capture_debounce_lock
                        with _unknown_capture_debounce_lock: 
                            if current_time_epoch - _last_unknown_capture_time >= config.UNKNOWN_CAPTURE_DEBOUNCE_TIME:
                                _last_unknown_capture_time = current_time_epoch # Update global debounce time
                                trigger_full_intruder_event = True # Allowed to trigger the full event
                            else:
                                print(f"DEBUG: Camera {self.camera_input}: Full intruder event debounce active. Skipping capture/log/email.", flush=True)

                        if trigger_full_intruder_event: 
                            try:
                                # 1. Get/Assign Intruder ID from intruder_tracker
                                # intruder_tracker now only returns the ID.
                                intruder_id_for_event = intruder_tracker.match_or_add_intruder(face_encoding)
                                print(f"INFO: Camera {self.camera_input}: [Intruder Event Triggered] ID: {intruder_id_for_event}", flush=True)

                                # 2. Capture a NEW image for this specific event
                                # Generate a unique filename and full path for this intruder snapshot.
                                timestamp_str_for_capture = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                newly_captured_intruder_filename = f"intruder_{timestamp_str_for_capture}.jpg"
                                full_path_for_new_capture = os.path.join(config.INTRUDERS_FOLDER, newly_captured_intruder_filename)
                                
                                cv2.imwrite(full_path_for_new_capture, frame) # Save the full frame of the intruder
                                print(f"INFO: Camera {self.camera_input}: Captured NEW intruder image: {full_path_for_new_capture}", flush=True)
                                
                                # 3. Log the event with the path to the NEWLY CAPTURED IMAGE
                                logging_manager.write_log_entry(intruder_id_for_event, "Intruder", image_link=full_path_for_new_capture)
                                print(f"INFO: Camera {self.camera_input}: Logged intruder entry with image: '{full_path_for_new_capture}'", flush=True)
                                
                                # 4. Send Email Alert
                                alert_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                email_sender.send_alert_in_thread(full_path_for_new_capture, alert_time, str(self.camera_input))
                                print(f"DEBUG: Camera {self.camera_input}: Started email alert thread.", flush=True)
                                
                            except Exception as e:
                                print(f"ERROR: Camera {self.camera_input}: Failed to handle full intruder event (capture/log/email): {e}", flush=True)
                        
                        # Update display name for this specific intruder
                        recognized_person_name_only = intruder_id_for_event # Display "Intruder_X"
                        # The gender is already "Intruder" by default for unknowns
                        
                    # --- End Intruder Event Management ---

                    # Add the recognized name to the list for display on frame
                    faces_to_display_on_frame.append(recognized_person_name_only)


                    # --- Logging Logic for Known Faces (debounced per person) ---
                    # This remains separate as it's a different debounce/logging trigger.
                    if recognized_name_with_gender != "Unknown": # This applies only if it's a known person
                        current_time_dt = datetime.datetime.now()
                        last_time = self.last_logged_times.get(recognized_name_with_gender, None)
                        if last_time is None or (current_time_dt - last_time).total_seconds() >= config.KNOWN_FACE_LOG_DEBOUNCE_TIME: # Use config
                            # Log known person. image_link is None for known persons.
                            logging_manager.write_log_entry(recognized_person_name_only, recognized_gender_only, image_link=None) 
                            self.last_logged_times[recognized_name_with_gender] = current_time_dt
                    
                # --- Display results (drawing on drawn_frame) ---
                # This part is outside the 'if unknown_face_detected_in_current_frame' block,
                # so it applies to both known and unknown faces that were detected.
                if len(face_locations) > 0:
                    for (top, right, bottom, left), display_name_on_frame in zip(face_locations, faces_to_display_on_frame):
                        top *= 4 ; right *= 4 ; bottom *= 4 ; left *= 4
                        color = (0, 255, 0) if display_name_on_frame != "Unknown" and not display_name_on_frame.startswith("Intruder_") else (0, 0, 255) # Green for known, Red for unknown/intruder
                        cv2.rectangle(drawn_frame, (left, top), (right, bottom), color, 2)

                        cv2.rectangle(drawn_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8
                        font_thickness = 2
                        text_color = (0, 0, 0) if display_name_on_frame != "Unknown" and not display_name_on_frame.startswith("Intruder_") else (255, 255, 255)
                        cv2.putText(drawn_frame, display_name_on_frame, (left + 6, bottom - 6), font, font_scale, text_color, font_thickness)
                else: 
                    text = "Non human object or no face detected"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    font_thickness = 2
                    text_x = 10 ; text_y = 30
                    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                    cv2.rectangle(drawn_frame, (text_x - 5, text_y - text_size[1] - 5),
                                (text_x + text_size[0] + 5, text_y + 5), (0, 0, 255), cv2.FILLED)
                    cv2.putText(drawn_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
            
            try:
                self.frame_queue.put_nowait(drawn_frame)
            except queue.Full:
                print(f"WARNING: Camera {self.camera_input}: Display queue full for main thread. Dropping frame.", flush=True)
                pass 

            time.sleep(0.001) 

        print(f"DEBUG: Camera {self.camera_input}: Processing thread loop exited cleanly.", flush=True) 

    def start(self):
        """Starts the camera stream's reader and processing threads."""
        self.reader_thread.start()
        self.processing_thread.start()
        print(f"DEBUG: Camera {self.camera_input}: Threads initiated and started.", flush=True)

    def stop(self):
        """Signals the camera stream threads to stop."""
        print(f"DEBUG: Camera {self.camera_input}: Stop method called. Setting stop_event.", flush=True)
        self.stop_event.set()