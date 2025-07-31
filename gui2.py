# gui.py
import sys
import os
import threading
import queue
import subprocess # IMPORTANT: Make sure this is imported for os.startfile / subprocess.run
from datetime import datetime, timedelta # Used for log filtering
import cv2 # Used for camera capture dialog and for displaying frames in OpenCV windows
from PIL import Image # PIL.Image is not directly used in the provided logic but is kept as it was in original gui.py
from PyQt5 import QtWidgets, QtGui, QtCore # Core PyQt5 library components

# Import your new modular components
import config # Centralized configuration for paths, debounce times, email, etc.
import face_data_manager # Module for loading, saving, updating, and deleting face encodings/data
import logging_manager # Module for writing to and reading from the activity log file
import main_recognition_logic # Module that orchestrates the live face recognition system backend


# --- Helper Function for Non-Blocking Temporary Messages ---
class TemporaryMessageBox(QtWidgets.QMessageBox):
    """
    A non-modal QMessageBox that closes automatically after a specified duration.
    Implements a singleton-like behavior to ensure only one is active at a time,
    preventing multiple overlapping message boxes from appearing.
    """
    _instance = None # Class variable to hold a single active instance

    def __init__(self, title, message, duration_ms=3000, parent=None):
        print(f"DEBUG: TemporaryMessageBox: Initializing '{title}' with message: '{message}'", flush=True)
        # If an instance already exists, close it before creating a new one
        if TemporaryMessageBox._instance is not None:
            print("DEBUG: TemporaryMessageBox: Closing existing instance.", flush=True)
            TemporaryMessageBox._instance.close()

        super().__init__(parent)
        TemporaryMessageBox._instance = self # Set this new instance as the singleton

        self.setWindowTitle(title) # Set the window title for the message box
        self.setText(message) # Set the message text to be displayed
        self.setStandardButtons(QtWidgets.QMessageBox.Ok) # Include an OK button for user acknowledgement
        self.setModal(False) # Non-modal: allows interaction with other windows while this one is open

        # Apply custom CSS styling for a more modern and consistent look
        self.setStyleSheet("QMessageBox { background-color: #e6f2ff; border: 1px solid #cceeff; border-radius: 5px; }"
                           "QLabel { color: #333333; font-size: 10pt; text-align: center; margin: 10px; }"
                           "QPushButton { background-color: #007bff; color: white; padding: 5px 15px; border-radius: 4px; }"
                           "QPushButton:hover { background-color: #0056b3; }")

        # Center the message box relative to its parent window or the desktop screen
        if parent:
            parent_rect = parent.geometry()
            x = parent_rect.x() + (parent_rect.width() - self.width()) // 2
            y = parent_rect.y() + (parent_rect.height() - self.height()) // 2
            self.move(x, y)
            print(f"DEBUG: TemporaryMessageBox: Centered relative to parent at ({x},{y}).", flush=True)
        else:
            screen_rect = QtWidgets.QApplication.desktop().screenGeometry()
            x = (screen_rect.width() - self.width()) // 2
            y = (screen_rect.height() - self.height()) // 2
            self.move(x, y)
            print(f"DEBUG: TemporaryMessageBox: Centered relative to screen at ({x},{y}).", flush=True)

        QtCore.QTimer.singleShot(duration_ms, self.close) # Set a timer to automatically close the message box after `duration_ms`
        self.finished.connect(self._clear_instance) # Connect the `finished` signal to clear the singleton instance
        self.open() # Show the message box (non-blocking)
        print(f"DEBUG: TemporaryMessageBox: Displaying message box. Auto-closing in {duration_ms}ms.", flush=True)

    def _clear_instance(self, result):
        """
        Slot connected to the `finished` signal of the message box.
        Resets the singleton instance reference when the message box is closed,
        allowing a new TemporaryMessageBox to be created later.
        """
        print(f"DEBUG: TemporaryMessageBox: Instance finished signal received with result {result}.", flush=True)
        if TemporaryMessageBox._instance == self: # Check if this is the active instance being closed
            TemporaryMessageBox._instance = None # Reset the singleton reference
            print("DEBUG: TemporaryMessageBox: Instance cleared (singleton reset).", flush=True)


# --- Main Application Window ---
class FaceRecognitionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        print("DEBUG: FaceRecognitionApp: Initializing main application window.", flush=True)
        self.setWindowTitle("Face Recognition Control Panel") # Set the window title
        self.setGeometry(100, 100, 800, 600) # Set initial position and size
        self.setWindowState(QtCore.Qt.WindowMaximized) # Start the application window maximized
        self.setStyleSheet("background-color: #e6f2ff;") # Apply a light blue background color

        self.central_widget = QtWidgets.QWidget() # Create a central widget for the main window
        self.setCentralWidget(self.central_widget) # Set it as the main window's central widget
        self.layout = QtWidgets.QVBoxLayout(self.central_widget) # Use a vertical layout for content
        self.layout.setAlignment(QtCore.Qt.AlignCenter) # Center content vertically within the layout

        self.create_menu_bar() # Setup the application's menu bar
        self.add_welcome_widgets() # Add initial welcome messages to the GUI

        # --- QThread references for background processes ---
        # The QThread that will run the `main_recognition_logic` module (the core system)
        self.recognition_thread = None
        self.recognition_stop_event = threading.Event() # A standard Python Event to signal the recognition thread to stop
        self.camera_display_queues = {} # Dictionary: {camera_input: queue.Queue} for receiving frames from recognition thread
        self.camera_display_timers = {} # Dictionary: {camera_input: QTimer} for updating individual OpenCV display windows

        # QThread for the encoding update process (replaces the old `update_face_encodings.py` subprocess)
        self.encoding_update_thread = None 
        # QThread for the person deletion process (replaces the old `delete_person.py` subprocess)
        self.delete_person_thread = None

        # Variables for the 'Capture New Face' modal dialog
        self.camera_capture_dialog = None # Reference to the modal dialog itself
        self.capture_dialog_video_label = None # QLabel within the dialog to display the live video feed
        self.capture_dialog_camera_capture = None # cv2.VideoCapture object for direct camera access in the dialog
        self.capture_dialog_timer = None # QTimer for continuously updating frames in the capture dialog
        print("DEBUG: FaceRecognitionApp: Initialization complete.", flush=True)

    def create_menu_bar(self):
        """
        Sets up the application's menu bar with a new, more organized structure:
        File, Registration, and Log menus.
        """
        print("DEBUG: FaceRecognitionApp: Creating menu bar with new structure.", flush=True)
        menu_bar = self.menuBar()
        # Apply custom CSS styling for the menu bar and its items for a consistent look
        menu_bar.setStyleSheet("QMenuBar { background-color: #4CAF50; color: white; }" # Green background for menu bar
                               "QMenuBar::item:selected { background-color: #45a049; }" # Slightly darker green on selection
                               "QMenu { background-color: white; border: 1px solid #ccc; border-radius: 4px; }" # White background for dropdown menus
                               "QMenu::item { padding: 5px 20px; }" # Padding for individual menu items
                               "QMenu::item:selected { background-color: #f0f0f0; }") # Light gray on menu item selection

        # --- 1. File Menu ---
        file_menu = menu_bar.addMenu("&File")

        # Start Camera (initiates the live face recognition system)
        start_camera_action = QtWidgets.QAction("&Start Camera", self)
        start_camera_action.triggered.connect(self.start_camera)
        file_menu.addAction(start_camera_action)
        print("DEBUG: Menu: File -> 'Start Camera' action added.", flush=True)

        # Stop Camera (stops the live face recognition system)
        stop_camera_action = QtWidgets.QAction("&Stop Camera", self)
        stop_camera_action.triggered.connect(self.stop_camera)
        file_menu.addAction(stop_camera_action)
        file_menu.addSeparator() # Separator after camera controls
        print("DEBUG: Menu: File -> 'Stop Camera' action added.", flush=True)

        # Exit Application
        exit_action = QtWidgets.QAction("&Exit", self)
        exit_action.triggered.connect(self.close) # Connects to QMainWindow's built-in close event
        file_menu.addAction(exit_action)
        print("DEBUG: Menu: File -> 'Exit' action added.", flush=True)


        # --- 2. Registration Menu (NEW Top-Level Menu) ---
        registration_menu = menu_bar.addMenu("&Registration")

        # New Face Registration (offers options to capture from camera or scan folder)
        new_face_action = QtWidgets.QAction("&New Face Registration", self)
        new_face_action.triggered.connect(self.add_new_face)
        registration_menu.addAction(new_face_action)
        print("DEBUG: Menu: Registration -> 'New Face Registration' action added.", flush=True)

        # Delete Registration (removes a person's data)
        delete_face_action = QtWidgets.QAction("&Delete Registration", self)
        delete_face_action.triggered.connect(self.delete_face)
        registration_menu.addAction(delete_face_action)
        print("DEBUG: Menu: Registration -> 'Delete Registration' action added.", flush=True)


        # --- 3. Log Menu (NEW Top-Level Menu) ---
        log_menu = menu_bar.addMenu("&Log")

        # View Full Log
        view_full_log_action = QtWidgets.QAction("&View Full Log", self)
        view_full_log_action.triggered.connect(self._view_full_log)
        log_menu.addAction(view_full_log_action)
        print("DEBUG: Menu: Log -> 'View Full Log' action added.", flush=True)

        # Search Log by Name
        search_log_by_name_action = QtWidgets.QAction("&Search Log by Name", self)
        search_log_by_name_action.triggered.connect(self._search_log_by_name)
        log_menu.addAction(search_log_by_name_action)
        print("DEBUG: Menu: Log -> 'Search Log by Name' action added.", flush=True)

        # View Person History (as a shortcut to search)
        # view_person_history_action = QtWidgets.QAction("&View Person History", self)
        # view_person_history_action.triggered.connect(self._view_person_history)
        # log_menu.addAction(view_person_history_action)
        # print("DEBUG: Menu: Log -> 'View Person History' action added.", flush=True)
        # log_menu.addSeparator() # Separator before Download Log

        # Download Log (Duration)
        download_log_action = QtWidgets.QAction("&Download Log (Duration)", self)
        download_log_action.triggered.connect(self.download_log)
        log_menu.addAction(download_log_action)
        print("DEBUG: Menu: Log -> 'Download Log (Duration)' action added.", flush=True)


        print("DEBUG: FaceRecognitionApp: Menu bar creation complete.", flush=True)

    def add_welcome_widgets(self):
        """Adds a welcome title and informational message to the central widget of the main window."""
        print("DEBUG: FaceRecognitionApp: Adding welcome widgets.", flush=True)
        title_label = QtWidgets.QLabel("Face Recognition Control Panel")
        title_label.setFont(QtGui.QFont("Helvetica", 18, QtGui.QFont.Bold))
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setStyleSheet("color: #333333; margin-bottom: 20px;")
        self.layout.addWidget(title_label)

        info_label = QtWidgets.QLabel("Use the 'File' menu to manage operations.")
        info_label.setFont(QtGui.QFont("Helvetica", 10))
        info_label.setAlignment(QtCore.Qt.AlignCenter)
        info_label.setStyleSheet("color: #555555;")
        self.layout.addWidget(info_label)
        print("DEBUG: FaceRecognitionApp: Welcome widgets added.", flush=True)
        
    # --- Encoding Update via QThread (replaces QProcess for update_face_encodings.py) ---
    def run_manual_update_encodings(self):
        """
        Initiates the face encoding update process in a background QThread.
        This function is called when the user selects "Update Existing Encodings".
        """
        print("DEBUG: FaceRecognitionApp: Attempting to run manual update encodings.", flush=True)
        # Prevent starting multiple update threads simultaneously to avoid resource conflicts
        if self.encoding_update_thread and self.encoding_update_thread.isRunning():
            TemporaryMessageBox("Process Running", "Face encoding update is already in progress. Please wait.", parent=self)
            print("DEBUG: Encoding Update: Already running, not starting new.", flush=True)
            return

        # Create and start a new QThread dedicated to the encoding update operation
        self.encoding_update_thread = EncodingUpdateThread(self)
        # Connect signals from the thread to methods in the GUI for status feedback
        self.encoding_update_thread.finished_signal.connect(self._handle_encoding_finished) # On success
        self.encoding_update_thread.error_signal.connect(self._handle_encoding_error) # On error
        self.encoding_update_thread.start() # Start the background thread
        print("DEBUG: Encoding Update: Started EncodingUpdateThread.", flush=True)
        TemporaryMessageBox("Encoding Update", "Face encoding update process started in the background. This may take a while.", parent=self)

    def _handle_encoding_finished(self, message):
        """
        Slot to handle the successful completion of the encoding update thread.
        Displays a success message to the user.
        """
        print(f"DEBUG: Encoding Update: Finished with message: {message}", flush=True)
        TemporaryMessageBox("Encoding Update", message, parent=self)
        self.encoding_update_thread = None # Clear the thread reference after its completion

    def _handle_encoding_error(self, message):
        """
        Slot to handle errors encountered during the encoding update thread.
        Displays an error message to the user.
        """
        print(f"ERROR: Encoding Update: Failed with message: {message}", flush=True)
        TemporaryMessageBox("Encoding Update Error", message, duration_ms=7000, parent=self)
        self.encoding_update_thread = None # Clear the thread reference after an error


    # --- Camera Functions (Live Recognition System Control) ---
    def start_camera(self):
        """
        Starts the live face recognition system by launching the `main_recognition_logic`
        in a background QThread. This also sets up display timers for OpenCV windows.
        """
        print("DEBUG: FaceRecognitionApp: Attempting to start camera process.", flush=True)
        # Prevent starting multiple recognition threads simultaneously
        if self.recognition_thread and self.recognition_thread.isRunning():
            TemporaryMessageBox("Camera", "Live face recognition is already running.", parent=self)
            print("DEBUG: Camera Control: Already running, not starting new.", flush=True)
            return

        # Prepare for a new recognition session:
        self.recognition_stop_event.clear() # Ensure the stop event is clear before starting a new session
        self.camera_display_queues = {} # Clear previous camera queues (will be repopulated by recognition thread)
        
        # Stop and clear any existing display timers to avoid conflicts from a previous run
        for timer in self.camera_display_timers.values():
            if timer.isActive():
                timer.stop()
        self.camera_display_timers = {} # Clear timer references after stopping them

        # Create and start the QThread that will run `main_recognition_logic.start_live_face_recognition`
        self.recognition_thread = RecognitionSystemThread(
            stop_event=self.recognition_stop_event,
            camera_display_queues=self.camera_display_queues # Pass the shared queues dictionary
        )
        # Connect signals:
        # - When the recognition thread starts, set up the display timers.
        # - When the recognition thread finishes, handle final cleanup.
        self.recognition_thread.started.connect(self._setup_camera_display_timers) 
        self.recognition_thread.finished.connect(self._handle_camera_finished) 
        self.recognition_thread.start() # Start the background thread
        
        print("DEBUG: Camera Control: Started RecognitionSystemThread.", flush=True)
        TemporaryMessageBox("Camera Control", "Live face recognition process started. OpenCV window(s) should appear.", parent=self)

    def _setup_camera_display_timers(self):
        """
        Sets up QTimers for each active camera to pull frames from their respective queues
        and display them in OpenCV windows. This method is called in the GUI's main thread
        when the `RecognitionSystemThread` signals it has started.
        """
        print("DEBUG: FaceRecognitionApp: Setting up camera display timers.", flush=True)
        
        # Use a singleShot timer to give `main_recognition_logic` a brief moment (1 second)
        # to initialize cameras and populate initial `camera_display_queues` with keys.
        # This prevents the display loop from immediately trying to access empty or non-existent queues.
        QtCore.QTimer.singleShot(1000, self._start_display_for_active_cameras)

    def _start_display_for_active_cameras(self):
        """
        Iterates through the list of active cameras (obtained from `camera_display_queues`)
        and sets up a dedicated QTimer for each to manage its corresponding OpenCV display window.
        This allows multiple camera feeds to be displayed simultaneously in separate windows.
        """
        print("DEBUG: FaceRecognitionApp: Iterating cameras to start display timers and windows.", flush=True)
        # Iterate over a copy of keys, as `camera_display_queues` might be dynamically updated
        # by the `RecognitionSystemThread` if new cameras come online (though not implemented here).
        for cam_input in list(self.camera_display_queues.keys()): 
            # Only create a new timer and window if one doesn't already exist for this camera input
            if cam_input not in self.camera_display_timers: 
                # --- CRUCIAL: Create the OpenCV window in the GUI's main thread ---
                window_name = f'Live Face Recognition - Camera ({cam_input})'
                try:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Create a resizable OpenCV window
                    cv2.resizeWindow(window_name, 640, 480) # Set an initial default size for the window
                    print(f"DEBUG: FaceRecognitionApp: Created OpenCV display window '{window_name}'.", flush=True)
                except Exception as e:
                    print(f"ERROR: FaceRecognitionApp: Failed to create OpenCV window for '{cam_input}': {e}. Skipping display for this camera.", flush=True)
                    continue # If window creation fails, skip setting up its timer and move to next camera

                timer = QtCore.QTimer(self)
                # Use a lambda to capture the current `cam_input` value for the slot.
                # This ensures the correct camera's queue is accessed when the timer fires.
                timer.timeout.connect(lambda ci=cam_input: self._update_live_camera_display(ci))
                timer.start(30) # Start the timer to update every 30ms (approx 33 FPS) for smooth video
                self.camera_display_timers[cam_input] = timer # Store timer reference in the dictionary
                print(f"DEBUG: FaceRecognitionApp: Started display timer for camera {cam_input}.", flush=True)

    def _update_live_camera_display(self, cam_input):
        """
        Slot called by a QTimer for a specific camera.
        It pulls a processed frame from the camera's queue and displays it in its dedicated OpenCV window.
        Also handles window closure by the user and `q` key press.
        """
        window_name = f'Live Face Recognition - Camera ({cam_input})'

        # Check if the OpenCV window associated with this camera still exists.
        # cv2.getWindowProperty returns -1 if the window has been manually closed by the user's 'X' button.
        # If it's closed, stop the timer for this specific camera and clean up.
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) == -1:
            print(f"INFO: FaceRecognitionApp: OpenCV window '{window_name}' was closed by user. Stopping its timer.", flush=True)
            if cam_input in self.camera_display_timers:
                self.camera_display_timers[cam_input].stop() # Stop the QTimer for this closed window
                del self.camera_display_timers[cam_input] # Remove the timer reference from our map
            # We don't `return` here immediately. We still attempt to get frames from the queue
            # even if the display window is gone. This is important to prevent the queue from constantly
            # filling up and consuming memory, as the recognition thread is still pushing to it.
            
        try:
            # Get the latest frame from the queue (non-blocking).
            # `get_nowait()` raises `queue.Empty` if no item is immediately available.
            frame_to_display = self.camera_display_queues[cam_input].get_nowait()
            if frame_to_display is not None:
                # Ensure the frame has valid data (is not empty) before attempting to display it
                if frame_to_display.size > 0:
                    cv2.imshow(window_name, frame_to_display)
                    # `cv2.waitKey(1)` is crucial here:
                    # - It processes OpenCV window events (like mouse clicks, resizing, window closing).
                    # - It returns the ASCII value of any key pressed while an OpenCV window is active.
                    # - It introduces a tiny delay (1ms) preventing the loop from hogging CPU, while
                    #   still allowing for rapid updates.
                    key = cv2.waitKey(1) & 0xFF # Process events and get key press (mask with 0xFF for ASCII)
                    if key == ord('q'): # Check if 'q' key was pressed in this active OpenCV window
                        print(f"INFO: FaceRecognitionApp: 'q' pressed in window {window_name}. Signaling stop cameras.", flush=True)
                        self.stop_camera() # Call the GUI's method to gracefully stop the entire system
                else:
                    print(f"WARNING: FaceRecognitionApp: Received empty frame for {cam_input}. Skipping imshow.", flush=True)
        except queue.Empty:
            pass # No new frame from this camera yet; simply skip displaying for this iteration
        except Exception as e:
            print(f"ERROR: FaceRecognitionApp: Error displaying frame for Camera ({cam_input}): {e}", flush=True)
            # If a display error occurs, stop the timer for this problematic camera to prevent recurrence.
            if cam_input in self.camera_display_timers:
                self.camera_display_timers[cam_input].stop()
                del self.camera_display_timers[cam_input]


    def _handle_camera_finished(self):
        """
        Slot connected to the `recognition_thread.finished` signal.
        This method is called when the `RecognitionSystemThread` (the main backend thread)
        has completed its execution or has been stopped. It performs final cleanup actions.
        """
        print("DEBUG: FaceRecognitionApp: RecognitionSystemThread finished signal received.", flush=True)
        # Ensure all display timers are stopped and cleared
        for cam_input, timer in list(self.camera_display_timers.items()):
            if timer.isActive():
                timer.stop()
            print(f"DEBUG: FaceRecognitionApp: Stopped display timer for camera {cam_input}.", flush=True)
            del self.camera_display_timers[cam_input]
        self.camera_display_timers = {} # Clear the dictionary after processing all timers

        cv2.destroyAllWindows() # Close all OpenCV windows that might still be open
        TemporaryMessageBox("Camera Control", "Live face recognition process closed normally.", parent=self)
        self.recognition_thread = None # Clear the thread reference
        self.recognition_stop_event.clear() # Clear the event for a fresh start next time
        self.camera_display_queues = {} # Clear the communication queues
        print("DEBUG: FaceRecognitionApp: All camera resources and threads cleaned up after finish signal.", flush=True)


    def stop_camera(self):
        """
        Stops the live face recognition system.
        This method signals the background recognition thread to stop,
        cleans up display timers, and closes OpenCV windows.
        """
        print("DEBUG: FaceRecognitionApp: Attempting to stop camera process.", flush=True)
        if self.recognition_thread and self.recognition_thread.isRunning():
            print("DEBUG: Camera Control: Signaling recognition thread to stop...", flush=True)
            self.recognition_stop_event.set() # Signal the background thread to exit its internal loops
            
            # Stop all active display timers immediately to prevent new frames from being drawn
            for cam_input, timer in list(self.camera_display_timers.items()): # Iterate over a copy to allow deletion during loop
                if timer.isActive():
                    timer.stop()
                print(f"DEBUG: FaceRecognitionApp: Stopped display timer for camera {cam_input}.", flush=True)
                del self.camera_display_timers[cam_input] # Remove timer reference from map
            self.camera_display_timers = {} # Clear the dictionary completely

            # Wait for the recognition thread to finish gracefully
            # A timeout is provided to prevent the GUI from freezing indefinitely if the thread is stuck.
            if not self.recognition_thread.wait(5000): # Wait up to 5 seconds for thread to clean up
                print("WARNING: Camera Control: Recognition thread did not terminate gracefully within timeout. It might be unresponsive.", flush=True)
                # For critical unresponsiveness, you might use self.recognition_thread.terminate() here,
                # but it's less graceful and can leave resources in an uncertain state.
            else:
                print("DEBUG: Camera Control: Recognition thread terminated successfully.", flush=True)

            self.recognition_thread = None # Clear the thread object reference
            self.recognition_stop_event.clear() # Reset the event for the next time 'Start Camera' is clicked
            self.camera_display_queues = {} # Clear the queues used for frame communication

            cv2.destroyAllWindows() # Close all OpenCV windows that were opened by the system
            TemporaryMessageBox("Camera Control", "Live face recognition process stopped successfully.", parent=self)
            print("DEBUG: Camera Control: All camera resources and threads cleaned up after stop.", flush=True)
        else:
            TemporaryMessageBox("Camera Control", "Live face recognition is not currently running.", parent=self)
            print("DEBUG: Camera Control: Not running, no action needed.", flush=True)


    # --- New Face Registration (UI flow) ---
    def add_new_face(self):
        """
        Presents options for new face registration: capture from live camera feed
        or trigger a scan of the dataset folder for newly added images.
        """
        print("DEBUG: FaceRecognitionApp: Opening 'New Face Registration Options' dialog.", flush=True)
        choice_dialog = QtWidgets.QDialog(self)
        choice_dialog.setWindowTitle("New Face Registration Options")
        choice_dialog.setFixedSize(350, 180) # Fixed size for layout consistency
        choice_dialog.setModal(True) # Make it modal to block interaction with main window until closed

        # Center the dialog on the parent window for better user experience
        parent_rect = self.geometry()
        x = parent_rect.x() + (parent_rect.width() - choice_dialog.width()) // 2
        y = parent_rect.y() + (parent_rect.height() - choice_dialog.height()) // 2
        choice_dialog.move(x, y)
        print(f"DEBUG: New Face Dialog: Centered at ({x},{y}).", flush=True)

        # Layout for buttons in the choice dialog
        layout = QtWidgets.QVBoxLayout(choice_dialog)
        layout.addWidget(QtWidgets.QLabel("Choose registration method:",
                                          font=QtGui.QFont("Helvetica", 12, QtGui.QFont.Bold),
                                          alignment=QtCore.Qt.AlignCenter))

        # Option 1: Capture New Face from Camera (opens a separate capture dialog)
        btn_capture = QtWidgets.QPushButton("Capture New Face (Camera)")
        btn_capture.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; padding: 8px;")
        # Connect to a lambda function that first accepts the dialog, then calls the capture function
        btn_capture.clicked.connect(lambda: (choice_dialog.accept(), self.capture_new_face_from_camera()))
        btn_capture.setFixedSize(280, 35) # Fixed size for button consistency
        layout.addWidget(btn_capture, alignment=QtCore.Qt.AlignCenter)
        print("DEBUG: New Face Dialog: 'Capture New Face' button added.", flush=True)

        # Option 2: Update Existing Encodings (scans the dataset folder for new images)
        btn_manual = QtWidgets.QPushButton("Update Existing Encodings (Scan Folder)")
        btn_manual.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 8px;")
        # Connect to a lambda function that first accepts the dialog, then calls the manual update function
        btn_manual.clicked.connect(lambda: (choice_dialog.accept(), self.run_manual_update_encodings()))
        btn_manual.setFixedSize(280, 35) # Fixed size for button consistency
        layout.addWidget(btn_manual, alignment=QtCore.Qt.AlignCenter)
        print("DEBUG: New Face Dialog: 'Update Existing Encodings' button added.", flush=True)

        layout.addStretch(1) # Adds stretchable space to push buttons to the center vertically
        result = choice_dialog.exec_() # Show the dialog modally and wait for user interaction
        print(f"DEBUG: New Face Dialog: Dialog closed with result {result}.", flush=True)


    def capture_new_face_from_camera(self):
        """
        Opens a modal dialog to capture a new face image from the live camera feed.
        Allows the user to preview the feed and capture a photo which is then saved to the dataset.
        """
        print("DEBUG: FaceRecognitionApp: Opening 'Capture New Face' camera dialog.", flush=True)
        self.camera_capture_dialog = QtWidgets.QDialog(self) # Create a new QDialog instance
        self.camera_capture_dialog.setWindowTitle("Capture New Face") # Set dialog title
        self.camera_capture_dialog.setFixedSize(680, 600) # Fixed size for the dialog
        self.camera_capture_dialog.setModal(True) # Make it modal (blocks parent window)
        self.camera_capture_dialog.setLayout(QtWidgets.QVBoxLayout()) # Set a vertical layout for dialog content
        self.camera_capture_dialog.layout().setAlignment(QtCore.Qt.AlignCenter) # Center content within its layout

        # Center the dialog on the parent window for better user experience
        parent_rect = self.geometry()
        x = parent_rect.x() + (parent_rect.width() - self.camera_capture_dialog.width()) // 2
        y = parent_rect.y() + (parent_rect.height() - self.camera_capture_dialog.height()) // 2
        self.camera_capture_dialog.move(x, y)
        print(f"DEBUG: Capture Face Dialog: Centered at ({x},{y}).", flush=True)

        # QLabel to display the live video feed. It will show "Opening camera..." initially.
        self.capture_dialog_video_label = QtWidgets.QLabel("Opening camera...")
        self.capture_dialog_video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.capture_dialog_video_label.setFixedSize(640, 480) # Fixed size for the video display area
        self.capture_dialog_video_label.setStyleSheet("border: 1px solid gray; background-color: black; color: lightgray;")
        self.camera_capture_dialog.layout().addWidget(self.capture_dialog_video_label)
        print("DEBUG: Capture Face Dialog: Video label added.", flush=True)

        # Button to capture a photo from the current frame
        self.capture_button = QtWidgets.QPushButton("Capture Photo")
        self.capture_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        # Connect to a new function to process the captured frame after click
        self.capture_button.clicked.connect(self._process_captured_face_from_dialog) 
        self.camera_capture_dialog.layout().addWidget(self.capture_button, alignment=QtCore.Qt.AlignCenter)
        print("DEBUG: Capture Face Dialog: 'Capture Photo' button added.", flush=True)

        # Button to cancel the capture process and close the dialog
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
        self.cancel_button.clicked.connect(self.camera_capture_dialog.reject) # Reject closes dialog with rejected status
        self.camera_capture_dialog.layout().addWidget(self.cancel_button, alignment=QtCore.Qt.AlignCenter)
        print("DEBUG: Capture Face Dialog: 'Cancel' button added.", flush=True)

        # Initialize OpenCV VideoCapture for the dialog's camera.
        # Uses camera ID 0 (default webcam), or you could make this configurable (e.g., from config.py)
        self.capture_dialog_camera_capture = cv2.VideoCapture(0) 
        print("DEBUG: Capture Face Dialog: Attempting to open camera (ID 0).", flush=True)
        if not self.capture_dialog_camera_capture.isOpened():
            QtWidgets.QMessageBox.critical(self, "Camera Error", 
                                           "Failed to open camera (ID 0) for capture. "
                                           "Please check if camera is connected and not in use by another application.", 
                                           parent=self.camera_capture_dialog)
            print("ERROR: Capture Face Dialog: Failed to open camera ID 0 for capture.", flush=True)
            self._release_capture_dialog_camera() # Clean up resources
            self.camera_capture_dialog.close() # Close the dialog if camera failed to open
            return
        # Set resolution for the capture camera for consistency and quality
        self.capture_dialog_camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture_dialog_camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("DEBUG: Capture Face Dialog: Camera opened successfully for capture.", flush=True)

        # QTimer to continuously update the video feed in the QLabel of the dialog
        self.capture_dialog_timer = QtCore.QTimer(self)
        self.capture_dialog_timer.timeout.connect(self._update_capture_dialog_frame)
        self.capture_dialog_timer.start(30) # Update every 30ms (approx 33 FPS) for smooth preview
        print("DEBUG: Capture Face Dialog: QTimer started for frame updates.", flush=True)

        # Connect dialog's finished signal to cleanup function, ensuring camera is released regardless of how dialog closes
        self.camera_capture_dialog.finished.connect(self._release_capture_dialog_camera)
        
        # Show the dialog modally and wait for user interaction (accept or reject).
        # This blocks the main GUI until the dialog is closed.
        result = self.camera_capture_dialog.exec_() 
        print(f"DEBUG: Capture Face Dialog: Dialog closed with result {result}.", flush=True)


    def _update_capture_dialog_frame(self):
        # ... (This method remains unchanged) ...
        ret, frame = self.capture_dialog_camera_capture.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

            pixmap = QtGui.QPixmap.fromImage(qt_image).scaled(
                self.capture_dialog_video_label.width(), self.capture_dialog_video_label.height(),
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            self.capture_dialog_video_label.setPixmap(pixmap)
        else:
            self.capture_dialog_timer.stop()
            print("ERROR: Capture Face Dialog: Failed to read frame from camera. Stopping timer.", flush=True)
            QtWidgets.QMessageBox.critical(self, "Camera Feed Error", 
                                           "Failed to get frame from camera during capture. Closing capture window.", 
                                           parent=self.camera_capture_dialog)
            if self.camera_capture_dialog:
                self.camera_capture_dialog.close()


    def _release_capture_dialog_camera(self):
        # ... (This method remains unchanged) ...
        print("DEBUG: Capture Face Dialog: Releasing camera resources.", flush=True)
        if self.capture_dialog_timer and self.capture_dialog_timer.isActive():
            self.capture_dialog_timer.stop()
            print("DEBUG: Capture Face Dialog: QTimer stopped.", flush=True)
        if self.capture_dialog_camera_capture and self.capture_dialog_camera_capture.isOpened():
            self.capture_dialog_camera_capture.release()
            self.capture_dialog_camera_capture = None
            print("DEBUG: Capture Face Dialog: OpenCV camera released.", flush=True)
            self.capture_dialog_video_label.setPixmap(QtGui.QPixmap())
            self.capture_dialog_video_label.setText("Camera Feed Stopped.")
            print("DEBUG: Capture Face Dialog: Video label cleared and text updated.", flush=True)

    def _process_captured_face_from_dialog(self):
        # ... (This method remains unchanged) ...
        print("DEBUG: FaceRecognitionApp: Processing captured face initiated.", flush=True)
        ret, frame_to_save = self.capture_dialog_camera_capture.read()

        if not ret or frame_to_save is None or frame_to_save.size == 0:
            QtWidgets.QMessageBox.warning(self, "Capture Error", 
                                           "Failed to capture a valid image. Please ensure camera is working and try again.", 
                                           parent=self.camera_capture_dialog)
            print("ERROR: Capture Face Dialog: Failed to capture valid image.", flush=True)
            return

        if self.camera_capture_dialog:
            self.camera_capture_dialog.accept()
            print("DEBUG: Capture Face Dialog: Camera dialog accepted and closed.", flush=True)

        self._get_person_details_and_save(frame_to_save)
        print("DEBUG: FaceRecognitionApp: _get_person_details_and_save called.", flush=True)


    def _get_person_details_and_save(self, frame_to_save):
        # ... (This method remains unchanged) ...
        print("DEBUG: FaceRecognitionApp: Getting person details for saving.", flush=True)
        person_name, ok = QtWidgets.QInputDialog.getText(self, "Input Name", 
                                                        "Enter person's name (e.g., 'John Doe'):",
                                                        QtWidgets.QLineEdit.Normal, "")
        if not ok or not person_name:
            QtWidgets.QMessageBox.warning(self, "Input Cancelled", "Name is required for registration. Face not saved.")
            print("DEBUG: Save Face: Name input cancelled or empty.", flush=True)
            return
        person_name = person_name.strip()
        print(f"DEBUG: Save Face: Person name entered: '{person_name}'.", flush=True)

        gender, ok = QtWidgets.QInputDialog.getItem(self, "Input Gender", "Select gender:", ["male", "female", "other"], 0, False)
        if not ok or not gender:
            QtWidgets.QMessageBox.warning(self, "Input Cancelled", "Gender selection is required. Face not saved.")
            print("DEBUG: Save Face: Gender input cancelled or empty.", flush=True)
            return
        gender = gender.lower()
        print(f"DEBUG: Save Face: Gender selected: '{gender}'.", flush=True)

        try:
            image_path = face_data_manager.save_new_face_image(frame_to_save, person_name, gender)
            QtWidgets.QMessageBox.information(self, "Success", f"Face captured and saved to:\n{image_path}")
            print(f"DEBUG: Save Face: Image successfully saved to {image_path}.", flush=True)

            self.run_manual_update_encodings()
            print("DEBUG: Save Face: Triggered manual encoding update after save.", flush=True)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Failed to save image or trigger encoding update: {e}")
            print(f"ERROR: Save Face: Failed to save image or trigger encoding update: {e}", flush=True)


    def delete_face(self):
        # ... (This method remains unchanged) ...
        print("DEBUG: FaceRecognitionApp: Attempting to delete face.", flush=True)
        if self.delete_person_thread and self.delete_person_thread.isRunning():
            TemporaryMessageBox("Process Running", "Deletion process is already in progress. Please wait.", parent=self)
            print("DEBUG: Delete Person: Already running, not starting new.", flush=True)
            return

        name_to_delete, ok = QtWidgets.QInputDialog.getText(self, "Delete Person", 
                                                            "Enter person's full name to delete (e.g., 'John_Doe__male' or just 'John Doe'):")
        if not ok or not name_to_delete:
            print("DEBUG: Delete Person: Name input cancelled or empty.", flush=True)
            return
        name_to_delete = name_to_delete.strip()
        print(f"DEBUG: Delete Person: Name entered for deletion: '{name_to_delete}'.", flush=True)

        confirm = QtWidgets.QMessageBox.question(self, "Confirm Deletion",
                                                 f"Are you sure you want to delete all data for '{name_to_delete}'? This action cannot be undone.",
                                                 QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                 QtWidgets.QMessageBox.No)
        if confirm == QtWidgets.QMessageBox.No:
            TemporaryMessageBox("Deletion Cancelled", f"Deletion of '{name_to_delete}' cancelled.", parent=self)
            print("DEBUG: Delete Person: Deletion confirmed as NO.", flush=True)
            return
        print("DEBUG: Delete Person: Deletion confirmed as YES. Starting deletion thread.", flush=True)

        self.delete_person_thread = DeletePersonThread(name_to_delete, self)
        self.delete_person_thread.finished_signal.connect(self._handle_delete_finished)
        self.delete_person_thread.error_signal.connect(self._handle_delete_error)
        self.delete_person_thread.start()
        TemporaryMessageBox("Deleting Person", f"Attempting to delete '{name_to_delete}' in the background. Please wait.", parent=self)

    def _handle_delete_finished(self, message):
        # ... (This method remains unchanged) ...
        print(f"DEBUG: Delete Person: Finished with message: {message}", flush=True)
        TemporaryMessageBox("Delete Registration", message, parent=self)
        self.delete_person_thread = None

    def _handle_delete_error(self, message):
        # ... (This method remains unchanged) ...
        print(f"ERROR: Delete Person: Failed with message: {message}", flush=True)
        TemporaryMessageBox("Delete Registration Error", message, duration_ms=7000, parent=self)
        self.delete_person_thread = None


    # --- MODIFIED/NEW Log Viewing Functions ---

    def _display_log_content_dialog(self, title, content_lines):
        """
        Helper method to display log content in a common QDialog using a QTableWidget.
        This allows for columnar display and embedding interactive buttons (e.g., View Image, View History).
        """
        print(f"DEBUG: Log Display Dialog: Preparing to display '{title}' with {len(content_lines)} lines.", flush=True)

        result_dialog = QtWidgets.QDialog(self)
        result_dialog.setWindowTitle(title)
        result_dialog.setFixedSize(900, 600) # Increased size to accommodate new columns and buttons
        result_dialog.setModal(True)

        # Center the dialog on the parent window
        parent_rect = self.geometry()
        x = parent_rect.x() + (parent_rect.width() - result_dialog.width()) // 2
        y = parent_rect.y() + (parent_rect.height() - result_dialog.height()) // 2
        result_dialog.move(x, y)
        print(f"DEBUG: Log Display Dialog: Centered at ({x},{y}).", flush=True)

        dialog_layout = QtWidgets.QVBoxLayout(result_dialog)
        dialog_layout.addWidget(QtWidgets.QLabel(title + ":",
                                                  font=QtGui.QFont("Helvetica", 12, QtGui.QFont.Bold),
                                                  alignment=QtCore.Qt.AlignCenter))

        # --- Use QTableWidget instead of QTextEdit ---
        table_widget = QtWidgets.QTableWidget()
        table_widget.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers) # Make table read-only
        table_widget.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows) # Select entire rows
        table_widget.setWordWrap(False) # Disable word wrap for cells
        table_widget.horizontalHeader().setStretchLastSection(False) # Don't stretch last section by default
        table_widget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents) # Adjust column width to content

        # Read the header from the content_lines (first non-empty line)
        header_line = ""
        data_start_row = 0
        if len(content_lines) >= 2: # Ensure there's at least header and separator
            header_line = content_lines[0].strip()
            data_start_row = 2 # Header is line 0, separator is line 1, data starts at line 2

        if not header_line:
            print("WARNING: Log Display Dialog: No header line found in log content. Using default headers.", flush=True)
            # Default headers if header is missing (must match expected order)
            headers_from_file = ["NAME", "GENDER", "DAY", "DATE", "TIME", "IMAGE_LINK"]
        else:
            # Parse headers from the first line, stripping whitespace
            headers_from_file = [h.strip() for h in header_line.split('|')]
        
        # Define the exact headers for our table, including the interactive 'History' column
        display_headers = headers_from_file + ["HISTORY"] # Add HISTORY for display

        table_widget.setColumnCount(len(display_headers))
        table_widget.setHorizontalHeaderLabels(display_headers)

        # Map header names to their column indices for easier access
        col_map = {header: idx for idx, header in enumerate(headers_from_file)}
        
        NAME_COL_IDX = col_map.get("NAME", 0) # Default to 0 if not found, safety
        GENDER_COL_IDX = col_map.get("GENDER", 1)
        DATE_COL_IDX = col_map.get("DATE", 3)
        TIME_COL_IDX = col_map.get("TIME", 4)
        IMAGE_LINK_COL_IDX = col_map.get("IMAGE_LINK", 5) # Assuming it's the 6th column (index 5)
        HISTORY_COL_IDX = len(display_headers) - 1


        # Populate table data from log file lines
        table_data_lines = content_lines[data_start_row:]
        table_widget.setRowCount(len(table_data_lines))

        for row_idx, line in enumerate(table_data_lines):
            # Split line by '|' and strip whitespace from each part
            parts = [p.strip() for p in line.split('|')]
            
            # Ensure the line has enough parts for the expected columns from the file
            # If a line is malformed, log a warning and skip/remove the row.
            if len(parts) < len(headers_from_file):
                print(f"WARNING: Log Display Dialog: Skipping malformed log line (too few columns): '{line.strip()}'", flush=True)
                # table_widget.removeRow(row_idx) # This would shift rows and might cause issues, better to just set empty items
                for col in range(table_widget.columnCount()): # Fill remaining with empty items
                    table_widget.setItem(row_idx, col, QtWidgets.QTableWidgetItem(""))
                continue

            # Populate standard data columns
            for col_file_idx, data_part in enumerate(parts):
                if col_file_idx < len(headers_from_file): # Only process up to defined headers
                    item = QtWidgets.QTableWidgetItem(data_part)
                    table_widget.setItem(row_idx, col_file_idx, item)

            # --- Handle IMAGE_LINK column: Embed a button if it's a link ---
            image_link_data = parts[IMAGE_LINK_COL_IDX] if IMAGE_LINK_COL_IDX < len(parts) else "N/A"
            
            if image_link_data and image_link_data != "N/A":
                btn_view_image = QtWidgets.QPushButton("View Image")
                btn_view_image.setStyleSheet("background-color: #007bff; color: white; border-radius: 3px; padding: 2px 5px;")
                # Connect lambda to pass the specific image_link for this row. `checked` is an unused arg from clicked signal.
                btn_view_image.clicked.connect(lambda checked, path=image_link_data: self._open_image_file(path))
                table_widget.setCellWidget(row_idx, IMAGE_LINK_COL_IDX, btn_view_image)
                # Set tooltip to show full path on hover
                item = QtWidgets.QTableWidgetItem() # Create an empty item to hold the tooltip
                item.setToolTip(image_link_data)
                table_widget.setItem(row_idx, IMAGE_LINK_COL_IDX, item)
            else:
                # For "N/A" entries, just set the text item
                item = QtWidgets.QTableWidgetItem("N/A")
                table_widget.setItem(row_idx, IMAGE_LINK_COL_IDX, item)

            # --- Handle HISTORY column: Embed a button ---
            btn_view_history = QtWidgets.QPushButton("View History")
            btn_view_history.setStyleSheet("background-color: #28a745; color: white; border-radius: 3px; padding: 2px 5px;")
            
            # Get the person's name from the current row for the history search
            person_name_for_history = parts[NAME_COL_IDX].strip()
            # Connect lambda to pass the person's name for direct search.
            btn_view_history.clicked.connect(lambda checked, name=person_name_for_history: self._search_log_by_name_direct(name))
            table_widget.setCellWidget(row_idx, HISTORY_COL_IDX, btn_view_history)
            item = QtWidgets.QTableWidgetItem() # Create an empty item to hold the history button
            table_widget.setItem(row_idx, HISTORY_COL_IDX, item) # QTableWidget needs an item even if it has a widget


        # Adjust column widths to fit content, then stretch the last column (HISTORY)
        table_widget.resizeColumnsToContents() 
        table_widget.horizontalHeader().setSectionResizeMode(HISTORY_COL_IDX, QtWidgets.QHeaderView.Stretch) # Stretch HISTORY column
        
        dialog_layout.addWidget(table_widget)
        print("DEBUG: Log Display Dialog: QTableWidget populated and configured.", flush=True)

        close_button = QtWidgets.QPushButton("Close")
        close_button.setStyleSheet("background-color: #6c757d; color: white; padding: 8px;")
        close_button.clicked.connect(result_dialog.accept)
        dialog_layout.addWidget(close_button, alignment=QtCore.Qt.AlignCenter)
        print("DEBUG: Log Display Dialog: 'Close' button added.", flush=True)

        result_dialog.exec_()
        print(f"DEBUG: Log Display Dialog: Dialog '{title}' closed.", flush=True)


    # NEW helper method: To open image files (used by IMAGE_LINK buttons)
    def _open_image_file(self, image_path):
        """
        Opens the specified image file using the system's default image viewer.
        Handles both relative and absolute paths for cross-platform compatibility.
        """
        print(f"DEBUG: GUI: Attempting to open image file: {image_path}", flush=True)
        # Construct absolute path if image_path is relative
        if not os.path.isabs(image_path):
            full_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_path)
            full_image_path = os.path.normpath(full_image_path) # Normalize path for consistency
        else:
            full_image_path = image_path

        if not os.path.exists(full_image_path):
            QtWidgets.QMessageBox.warning(self, "Image Not Found", f"Image file not found:\n{full_image_path}")
            print(f"ERROR: GUI: Image file not found: {full_image_path}", flush=True)
            return

        try:
            # Use platform-specific commands to open the file
            if sys.platform == "win32":
                os.startfile(full_image_path) # Windows specific command
            elif sys.platform == "darwin":
                subprocess.run(["open", full_image_path]) # macOS specific command
            else: # Linux and other POSIX-like systems
                subprocess.run(["xdg-open", full_image_path]) # General Linux command
            print(f"DEBUG: GUI: Successfully initiated opening of {full_image_path}", flush=True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error Opening Image", f"Could not open image:\n{full_image_path}\nError: {e}")
            print(f"ERROR: GUI: Failed to open image {full_image_path}: {e}", flush=True)

    # NEW helper method: Direct search for history button (to avoid re-prompting)
    def _search_log_by_name_direct(self, person_name):
        """
        Performs a log search directly by a provided person's name (without prompting the user).
        This method is typically called when a 'View History' button in the log table is clicked.
        It then displays the filtered log entries in a new dialog.
        """
        print(f"DEBUG: GUI: Direct log search initiated for: '{person_name}'.", flush=True)
        # Read all log lines from the logging_manager module
        all_lines = logging_manager.read_log_file()
        if not all_lines:
            QtWidgets.QMessageBox.warning(self, "No Log Data", f"Log file '{config.LOG_FILE_PATH}' is empty or not found for history.")
            print(f"ERROR: GUI: Log file '{config.LOG_FILE_PATH}' is empty or not found for history.", flush=True)
            return

        filtered_content = []
        data_lines_start_index = 0

        # Check for and include the header and separator lines
        if len(all_lines) >= 2 and all_lines[0].strip().startswith('NAME') and 'GENDER' in all_lines[0]:
            filtered_content.append(all_lines[0]) # Header line
            filtered_content.append(all_lines[1]) # Separator line
            data_lines_start_index = 2

        # Filter log lines by the provided person_name (case-insensitive search)
        found_data_lines = [
            line for line in all_lines[data_lines_start_index:]
            if person_name.lower() in line.lower() # Case-insensitive search
        ]

        if found_data_lines:
            final_display_lines = filtered_content + found_data_lines
            # Reuse the common display dialog to show the filtered history
            self._display_log_content_dialog(f"History for {person_name.capitalize()}", final_display_lines)
        else:
            QtWidgets.QMessageBox.information(self, "No History", f"No log entries found for '{person_name}'.")
            print(f"DEBUG: GUI: No log entries found for '{person_name}'.", flush=True)

    # --- Renamed/Re-purposed Log Display Methods ---

    # This method is the new functionality for "View Full Log" menu item.
    def _view_full_log(self):
        """
        Displays the entire content of the log file in a QTableWidget dialog.
        This is connected to the "View Full Log" menu action.
        """
        print("DEBUG: FaceRecognitionApp: Attempting to view full log.", flush=True)

        # Read all log lines from the logging_manager module
        all_lines = logging_manager.read_log_file()
        if not all_lines:
            QtWidgets.QMessageBox.warning(self, "No Log Data", f"Log file '{config.LOG_FILE_PATH}' is empty or not found.")
            print(f"ERROR: View Full Log: Log file '{config.LOG_FILE_PATH}' is empty or not found.", flush=True)
            return
        print(f"DEBUG: View Full Log: Read {len(all_lines)} lines from log file.", flush=True)

        # Display all content directly using the QTableWidget helper
        self._display_log_content_dialog("Full System Log", all_lines)


    # This method is the new functionality for "Search Log by Name" menu item.
    def _search_log_by_name(self):
        """
        Prompts the user for a name, then filters and displays log entries
        related to that name in a QTableWidget dialog.
        This is connected to the "Search Log by Name" menu action.
        """
        print("DEBUG: FaceRecognitionApp: Attempting to search log by name (from menu).", flush=True)
        name_to_search, ok = QtWidgets.QInputDialog.getText(self, "Search Log", "Enter name to search in log:")
        if not ok or not name_to_search:
            print("DEBUG: Search Log by Name (menu): Name input cancelled or empty.", flush=True)
            return
        name_to_search = name_to_search.strip().lower() # Convert to lowercase for case-insensitive search
        print(f"DEBUG: Search Log by Name (menu): Name to search: '{name_to_search}'.", flush=True)

        # Read all log lines from the logging_manager module
        all_lines = logging_manager.read_log_file()
        if not all_lines:
            QtWidgets.QMessageBox.warning(self, "No Log Data", f"Log file '{config.LOG_FILE_PATH}' is empty or not found.")
            print(f"ERROR: Search Log by Name (menu): Log file '{config.LOG_FILE_PATH}' is empty or not found.", flush=True)
            return
        print(f"DEBUG: Search Log by Name (menu): Read {len(all_lines)} lines from log file.", flush=True)

        filtered_content = []
        data_lines_start_index = 0

        # Check for and include the header and separator lines
        if len(all_lines) >= 2 and all_lines[0].strip().startswith('NAME') and 'GENDER' in all_lines[0]:
            filtered_content.append(all_lines[0]) # Header line
            filtered_content.append(all_lines[1]) # Separator line
            data_lines_start_index = 2

        # Filter log lines where the search name is found (case-insensitive)
        found_data_lines = [
            line for line in all_lines[data_lines_start_index:]
            if name_to_search in line.lower()
        ]
        print(f"DEBUG: Search Log by Name (menu): Found {len(found_data_lines)} matching data lines.", flush=True)

        if found_data_lines:
            final_display_lines = filtered_content + found_data_lines
            self._display_log_content_dialog(f"Search Results for {name_to_search.capitalize()}", final_display_lines)
        else:
            QtWidgets.QMessageBox.information(self, "No Entries", f"No logs found for '{name_to_search}'.")
            print(f"DEBUG: Search Log by Name (menu): No entries found for '{name_to_search}'.", flush=True)

    # This method is the new functionality for "View Person History" menu item.
    def _view_person_history(self):
        """
        Acts as a shortcut to search the log by a person's name, prompting the user for input.
        This is connected to the "View Person History" menu action.
        """
        print("DEBUG: FaceRecognitionApp: Attempting to view person history (from menu).", flush=True)
        # Simply call the _search_log_by_name method, as that provides the desired functionality.
        self._search_log_by_name()
        print("DEBUG: View Person History (menu): Called _search_log_by_name.", flush=True)
        
    def download_log(self):
        # ... (This method remains unchanged) ...
        print("DEBUG: FaceRecognitionApp: Attempting to download log.", flush=True)
        if not os.path.exists(config.LOG_FILE_PATH):
            QtWidgets.QMessageBox.warning(self, "No File", f"No log file '{config.LOG_FILE_PATH}' found.")
            print(f"ERROR: Download Log: Log file '{config.LOG_FILE_PATH}' not found.", flush=True)
            return

        filter_dialog = FilterDialog(self)
        print("DEBUG: Download Log: FilterDialog opened.", flush=True)
        if filter_dialog.exec_() == QtWidgets.QDialog.Accepted:
            choice, from_date_str, to_date_str = filter_dialog.get_results()
            print(f"DEBUG: Download Log: FilterDialog accepted. Choice: {choice}, From: {from_date_str}, To: {to_date_str}.", flush=True)
            self._process_and_save_log(choice, from_date_str, to_date_str)
        else:
            TemporaryMessageBox("Cancelled", "Log download cancelled.", parent=self)
            print("DEBUG: Download Log: FilterDialog cancelled.", flush=True)

    def _process_and_save_log(self, choice, from_date_str, to_date_str):
        # ... (This method remains unchanged) ...
        print(f"DEBUG: Download Log: Processing and saving log with choice '{choice}'.", flush=True)
        current_time_exact = datetime.now()
        from_date_filter = None
        to_date_filter = None

        if choice == "1":
            try:
                from_date_filter = datetime.strptime(from_date_str, "%Y-%m-%d").replace(hour=0, minute=0, second=0, microsecond=0)
                to_date_filter = datetime.strptime(to_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999)
                if from_date_filter > to_date_filter:
                    QtWidgets.QMessageBox.critical(self, "Error", "From Date must be before or same as To Date.")
                    print(f"ERROR: Download Log: Invalid date range: From ({from_date_filter}) > To ({to_date_filter}).", flush=True)
                    return
                print(f"DEBUG: Download Log: Date range filter set: {from_date_filter} to {to_date_filter}.", flush=True)
            except ValueError:
                QtWidgets.QMessageBox.critical(self, "Error", "Invalid date format. Please use YYYY-MM-DD.")
                print(f"ERROR: Download Log: Invalid date format for '{from_date_str}' or '{to_date_str}'.", flush=True)
                return
        elif choice == '2':
            to_date_filter = current_time_exact.replace(hour=23, minute=59, second=59, microsecond=999999)
            from_date_filter = (current_time_exact - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
            print(f"DEBUG: Download Log: Filter set to last 7 days: {from_date_filter} to {to_date_filter}.", flush=True)
        elif choice == '3':
            to_date_filter = current_time_exact.replace(hour=23, minute=59, second=59, microsecond=999999)
            from_date_filter = (current_time_exact - timedelta(days=29)).replace(hour=0, minute=0, second=0, microsecond=0)
            print(f"DEBUG: Download Log: Filter set to last 30 days: {from_date_filter} to {to_date_filter}.", flush=True)
        elif choice == '4':
            to_date_filter = current_time_exact.replace(hour=23, minute=59, second=59, microsecond=999999)
            from_date_filter = (current_time_exact - timedelta(days=364)).replace(hour=0, minute=0, second=0, microsecond=0)
            print(f"DEBUG: Download Log: Filter set to last 365 days: {from_date_filter} to {to_date_filter}.", flush=True)
        elif choice == '6':
            to_date_filter = current_time_exact
            from_date_filter = current_time_exact - timedelta(hours=24)
            print(f"DEBUG: Download Log: Filter set to last 24 hours: {from_date_filter} to {to_date_filter}.", flush=True)
        elif choice == '5':
            print("DEBUG: Download Log: Filter set to all records.", flush=True)

        filtered_lines = []
        try:
            all_lines = logging_manager.read_log_file()
            if not all_lines:
                QtWidgets.QMessageBox.information(self, "No Log Data", "The log file is empty or could not be read.")
                print("ERROR: Download Log: No lines read from log file.", flush=True)
                return
            print(f"DEBUG: Download Log: Read {len(all_lines)} lines from log file.", flush=True)

            data_lines_start_index = 0
            if len(all_lines) >= 2 and all_lines[0].strip().startswith('NAME') and 'GENDER' in all_lines[0]:
                filtered_lines.append(all_lines[0])
                filtered_lines.append(all_lines[1])
                data_lines_start_index = 2
                print("DEBUG: Download Log: Log header and separator included.", flush=True)

            for line in all_lines[data_lines_start_index:]:
                parts = line.split('|')
                # Check for minimum number of parts to account for new IMAGE_LINK column
                if len(parts) >= 6: # Now expect at least 6 parts: NAME | GENDER | DAY | DATE | TIME | IMAGE_LINK
                    date_part = parts[3].strip()
                    time_part = parts[4].strip()
                    try:
                        if not date_part or not time_part:
                            print(f"WARNING: Download Log: Skipping malformed line (empty date/time): {line.strip()}", flush=True)
                            continue
                        log_datetime = datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        print(f"WARNING: Download Log: Skipping line with invalid date/time format: {line.strip()}", flush=True)
                        continue 

                    if from_date_filter is None and to_date_filter is None:
                        filtered_lines.append(line)
                    elif from_date_filter <= log_datetime <= to_date_filter:
                        filtered_lines.append(line)
                else:
                    print(f"WARNING: Download Log: Skipping malformed line (too few parts): {line.strip()}", flush=True)
            print(f"DEBUG: Download Log: Filtered down to {len(filtered_lines)} lines (including header).", flush=True)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to read or process log file: {e}")
            print(f"ERROR: Download Log: Failed to read or process log file: {e}", flush=True)
            return

        if not filtered_lines or (len(filtered_lines) <= 2 and from_date_filter is not None):
            QtWidgets.QMessageBox.information(self, "No Data", "No logs found for the selected date range.")
            print("DEBUG: Download Log: No data found for selected range.", flush=True)
            return

        default_filename = "filtered_log_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Filtered Log As", default_filename, "Text files (*.txt);;All files (*.*)"
        )
        if save_path:
            try:
                with open(save_path, "w", encoding='utf-8') as f_out:
                    f_out.writelines(filtered_lines)
                TemporaryMessageBox("Saved", f"Filtered log saved to:\n{save_path}", parent=self)
                print(f"DEBUG: Download Log: Filtered log saved to '{save_path}'.", flush=True)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save filtered log file: {e}")
                print(f"ERROR: Download Log: Failed to save filtered log file: {e}", flush=True)

    def closeEvent(self, event):
        # ... (This method remains unchanged) ...
        print("DEBUG: FaceRecognitionApp: closeEvent triggered. Terminating child processes.", flush=True)
        if self.recognition_thread and self.recognition_thread.isRunning():
            print("DEBUG: closeEvent: Signaling recognition QThread to stop...", flush=True)
            self.recognition_stop_event.set()
            
            for cam_input, timer in list(self.camera_display_timers.items()):
                if timer.isActive():
                    timer.stop()
                print(f"DEBUG: closeEvent: Stopped display timer for camera {cam_input}.", flush=True)
            self.camera_display_timers = {}

            if not self.recognition_thread.wait(5000):
                print("WARNING: closeEvent: Recognition QThread did not terminate gracefully. Forcibly terminating.", flush=True)
                self.recognition_thread.terminate()
            else:
                print("DEBUG: closeEvent: Recognition QThread terminated successfully.", flush=True)
            self.recognition_thread = None
            self.recognition_stop_event.clear()
            self.camera_display_queues = {}

        if self.encoding_update_thread and self.encoding_update_thread.isRunning():
            print("DEBUG: closeEvent: Encoding update QThread still running. Waiting for it to finish...", flush=True)
            if not self.encoding_update_thread.wait(2000):
                print("WARNING: closeEvent: Encoding update QThread did not finish gracefully within timeout.", flush=True)

        if self.delete_person_thread and self.delete_person_thread.isRunning():
            print("DEBUG: closeEvent: Delete person QThread still running. Waiting for it to finish...", flush=True)
            if not self.delete_person_thread.wait(2000):
                print("WARNING: closeEvent: Delete person QThread did not finish gracefully within timeout.", flush=True)

        self._release_capture_dialog_camera()
        cv2.destroyAllWindows()
        
        print("DEBUG: closeEvent: All child processes handled. Accepting close event.", flush=True)
        event.accept()


# --- QThread Subclasses for Background Tasks ---
class RecognitionSystemThread(QtCore.QThread):
    def __init__(self, stop_event, camera_display_queues, parent=None):
        super().__init__(parent)
        self.stop_event = stop_event
        self.camera_display_queues = camera_display_queues

    def run(self):
        print("DEBUG: RecognitionSystemThread: Starting run method.", flush=True)
        main_recognition_logic.start_live_face_recognition(self.stop_event, self.camera_display_queues)
        print("DEBUG: RecognitionSystemThread: Finished run method.", flush=True)


class EncodingUpdateThread(QtCore.QThread):
    finished_signal = QtCore.pyqtSignal(str)
    error_signal = QtCore.pyqtSignal(str)

    def run(self):
        print("DEBUG: EncodingUpdateThread: Starting run method.", flush=True)
        try:
            result_message = face_data_manager.update_encodings_from_dataset()
            self.finished_signal.emit(result_message)
            print("DEBUG: EncodingUpdateThread: Finished successfully.", flush=True)
        except Exception as e:
            error_msg = f"Error updating encodings: {e}"
            self.error_signal.emit(error_msg)
            print(f"ERROR: EncodingUpdateThread: {error_msg}", flush=True)


class DeletePersonThread(QtCore.QThread):
    finished_signal = QtCore.pyqtSignal(str)
    error_signal = QtCore.pyqtSignal(str)
    
    def __init__(self, person_name, parent=None):
        super().__init__(parent)
        self.person_name = person_name

    def run(self):
        print(f"DEBUG: DeletePersonThread: Starting run method for '{self.person_name}'.", flush=True)
        try:
            result_message = face_data_manager.delete_person_from_encodings(self.person_name)
            self.finished_signal.emit(result_message)
            print(f"DEBUG: DeletePersonThread: Finished successfully for '{self.person_name}'.", flush=True)
        except ValueError as e:
            error_msg = f"Deletion failed: {e}. Please ensure the name is correct."
            self.error_signal.emit(error_msg)
            print(f"ERROR: DeletePersonThread: {error_msg}", flush=True)
        except Exception as e:
            error_msg = f"An unexpected error occurred during deletion: {e}"
            self.error_signal.emit(error_msg)
            print(f"ERROR: DeletePersonThread: {error_msg}", flush=True)


# --- Filter Dialog for Download Log (remains largely same as original but now part of gui.py) ---
class FilterDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        print("DEBUG: FilterDialog: Initializing filter dialog.", flush=True)
        super().__init__(parent)
        self.setWindowTitle("Select Log Filter")
        self.setFixedSize(350, 300)
        self.setModal(True)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(QtWidgets.QLabel("Choose filter option:", 
                                               font=QtGui.QFont("Helvetica", 10, QtGui.QFont.Bold)))

        self.radio_group = QtWidgets.QButtonGroup(self)
        self.options = [
            ("Day Range (YYYY-MM-DD)", "1"),
            ("Last 7 Days", "2"),
            ("Last 30 Days", "3"),
            ("Last 365 Days", "4"),
            ("Today (Last 24 Hours)", "6"),
            ("All Records", "5")
        ]

        self.radio_buttons = []
        for text, val in self.options:
            radio_btn = QtWidgets.QRadioButton(text)
            radio_btn.setChecked(val == "1")
            radio_btn.clicked.connect(self.toggle_date_entries)
            self.layout.addWidget(radio_btn)
            self.radio_group.addButton(radio_btn, int(val))
            self.radio_buttons.append(radio_btn)
            print(f"DEBUG: FilterDialog: Added radio button for '{text}'.", flush=True)


        self.date_frame = QtWidgets.QFrame(self)
        self.date_layout = QtWidgets.QFormLayout(self.date_frame)

        self.entry_from = QtWidgets.QLineEdit()
        self.entry_from.setPlaceholderText(datetime.now().strftime("%Y-%m-%d"))
        self.date_layout.addRow("From Date:", self.entry_from)
        print("DEBUG: FilterDialog: 'From Date' input added.", flush=True)

        self.entry_to = QtWidgets.QLineEdit()
        self.entry_to.setPlaceholderText(datetime.now().strftime("%Y-%m-%d"))
        self.date_layout.addRow("To Date:", self.entry_to)
        self.layout.addWidget(self.date_frame)
        print("DEBUG: FilterDialog: 'To Date' input added.", flush=True)

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        self.layout.addWidget(btn_box)
        print("DEBUG: FilterDialog: OK/Cancel buttons added.", flush=True)

        self.toggle_date_entries()

        if parent:
            parent_rect = parent.geometry()
            x = parent_rect.x() + (parent_rect.width() - self.width()) // 2
            y = parent_rect.y() + (parent_rect.height() - self.height()) // 2
            self.move(x, y)
            print(f"DEBUG: FilterDialog: Centered relative to parent at ({x},{y}).", flush=True)

        print("DEBUG: FilterDialog: Initialization complete.", flush=True)


    def toggle_date_entries(self):
        """
        Toggles the visibility of the date input fields based on the currently selected radio button.
        Date fields are only visible for the "Day Range" option.
        """
        current_choice = self.radio_group.checkedId()
        if current_choice == 1: # If "Day Range" is selected (ID 1)
            self.date_frame.show()
            # print("DEBUG: FilterDialog: Date entries frame shown.", flush=True) # Removed verbose print
        else:
            self.date_frame.hide()
            # print("DEBUG: FilterDialog: Date entries frame hidden.", flush=True) # Removed verbose print


    def get_results(self):
        """
        Returns the selected filter choice (as a string ID) and the entered "From" and "To" date strings.
        """
        choice = str(self.radio_group.checkedId())
        from_date = self.entry_from.text()
        to_date = self.entry_to.text()
        print(f"DEBUG: FilterDialog: Returning results: Choice={choice}, From={from_date}, To={to_date}.", flush=True)
        return choice, from_date, to_date

# --- Main Application Execution ---
if __name__ == '__main__':
    print("DEBUG: Main GUI: Starting QApplication.", flush=True)
    app = QtWidgets.QApplication(sys.argv)
    main_window = FaceRecognitionApp()
    main_window.show()
    print("DEBUG: Main GUI: Showing main window. Entering event loop.", flush=True)
    sys.exit(app.exec_())
    print("DEBUG: Main GUI: Exited event loop. Application terminating.", flush=True)