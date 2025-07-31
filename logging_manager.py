# logging_manager.py
import os
import datetime
import threading

import config # Import the centralized configuration

# Global lock for thread-safe log file access.
# This ensures that multiple threads (e.g., camera streams) don't try to write to the log file simultaneously,
# which could corrupt the file.
logging_lock = threading.Lock()

def write_log_header_if_needed():
    """
    Writes the header row to the log file if the file does not exist or is empty.
    The header now includes an 'IMAGE_LINK' column for intruder snapshots.
    The 'HISTORY' column is managed purely in the GUI display and is NOT part of the data.
    Uses a lock to ensure thread-safe writing.
    """
    with logging_lock: # Acquire lock before checking/writing to file
        log_file_path = config.LOG_FILE_PATH # Get path from config
        if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0:
            try:
                # Updated header format with only the 'IMAGE_LINK' new column, and proper padding
                log_header = (
                    f"{'NAME'.ljust(25)} | {'GENDER'.ljust(8)} | {'DAY'.ljust(6)} | {'DATE'.ljust(12)} | {'TIME'.ljust(8)} | {'IMAGE_LINK'}\n"
                )
                # Separator length adjusted to match the headers + padding
                log_separator = (
                    f"{'-'*25}-+-{'-'*8}-+-{'-'*6}-+-{'-'*12}-+-{'-'*8}-+-{'-'*len('IMAGE_LINK')}\n"
                )
                with open(log_file_path, "w", encoding='utf-8') as log_file:
                    log_file.write(log_header)
                    log_file.write(log_separator)
                print(f"DEBUG: logging_manager: Log header written to {log_file_path}.", flush=True)
            except Exception as e:
                print(f"ERROR: logging_manager: Failed to write log header to {log_file_path}: {e}", flush=True)

def write_log_entry(name, gender, image_link=None):
    """
    Appends a new entry to the face detection log file with current timestamp.
    Accepts an optional image_link for intruder snapshots.
    The 'HISTORY' display feature is handled by GUI interactivity, not by data storage.
    Uses a lock to ensure thread-safe writing.

    Args:
        name (str): Name of the person/intruder.
        gender (str): Gender of the person ('male', 'female', 'unknown', 'intruder').
        image_link (str, optional): Path to the intruder snapshot. Defaults to None.
                                    If None or empty, "N/A" will be logged in the image_link column.
    """
    current_time_dt = datetime.datetime.now()
    day_str = current_time_dt.strftime("%a") # e.g., Mon, Tue
    date_str = current_time_dt.strftime("%Y-%m-%d") # e.g., 2023-10-27
    time_str = current_time_dt.strftime("%H:%M:%S") # e.g., 14:30:55

    # Format image_link: Use provided path or "N/A" if None/empty string
    display_image_link = image_link if image_link else "N/A"
    
    # Prepare the formatted entry string with 6 columns.
    # Note: No padding for image_link here, as its length is variable. GUI will handle alignment.
    formatted_entry = (
        f"{name.ljust(25)} | {gender.ljust(8)} | {day_str.ljust(6)} | {date_str.ljust(12)} | {time_str.ljust(8)} | "
        f"{display_image_link}\n"
    )
    
    with logging_lock: # Acquire lock before writing to file
        log_file_path = config.LOG_FILE_PATH # Get path from config
        try:
            with open(log_file_path, "a", encoding='utf-8') as log_file:
                log_file.write(formatted_entry)
                log_file.flush()  # Ensure immediate write to disk
            print(f"DEBUG: logging_manager: Logged: {name} ({gender}) with image_link '{display_image_link}'.", flush=True)
        except Exception as e:
            print(f"CRITICAL ERROR: logging_manager: Failed to write log entry for '{name}' ('{gender}') to {log_file_path}: {e}", flush=True)

def read_log_file():
    """
    Reads the entire log file and returns its content as a list of lines.
    Uses a lock to ensure thread-safe reading.
    """
    with logging_lock: # Acquire lock before reading from file
        log_file_path = config.LOG_FILE_PATH # Get path from config
        if not os.path.exists(log_file_path):
            print(f"WARNING: logging_manager: Log file '{log_file_path}' not found for reading.", flush=True)
            return []
        try:
            with open(log_file_path, "r", encoding='utf-8') as f:
                lines = f.readlines()
            print(f"DEBUG: logging_manager: Read {len(lines)} lines from log file '{log_file_path}'.", flush=True)
            return lines
        except Exception as e:
            print(f"ERROR: logging_manager: Failed to read log file '{log_file_path}': {e}", flush=True)
            return []

# Optional: Standalone test for this module
if __name__ == "__main__":
    print("\n--- Testing logging_manager.py (Standalone) with IMAGE_LINK column only ---", flush=True)
    
    # Ensure a clean log file for testing
    if os.path.exists(config.LOG_FILE_PATH):
        os.remove(config.LOG_FILE_PATH)
        print(f"DEBUG: logging_manager: Cleaned up existing log file: {config.LOG_FILE_PATH}", flush=True)

    # Test writing header (new format)
    write_log_header_if_needed()
    
    # Test writing entries (with and without image_link)
    write_log_entry("John_Doe", "male", image_link=None) # Known person, no image_link
    write_log_entry("Jane_Smith", "female", image_link="") # Also known person, empty string
    write_log_entry("Intruder_1", "Intruder", image_link="intruders/intruder_20250730_100000.jpg")
    write_log_entry("Intruder_2", "Intruder", image_link="intruders/intruder_20250730_100005.jpg")

    # Test reading entries
    print("\n--- Reading log file ---", flush=True)
    log_content = read_log_file()
    for line in log_content:
        print(line.strip(), flush=True)

    # Test concurrent writes (simple simulation)
    print("\n--- Testing concurrent writes (simulated) ---", flush=True)
    def concurrent_writer(name, gender, delay, img_link=None):
        time.sleep(delay)
        write_log_entry(name, gender, img_link)

    import time
    threads = []
    threads.append(threading.Thread(target=concurrent_writer, args=("Alice", "female", 0.1, None)))
    threads.append(threading.Thread(target=concurrent_writer, args=("Bob", "male", 0.05, "")))
    threads.append(threading.Thread(target=concurrent_writer, args=("Intruder_3", "Intruder", 0.15, "intruders/intruder_20250730_100010.jpg")))

    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    print("\n--- Reading log file after concurrent writes ---", flush=True)
    log_content_after_concurrent = read_log_file()
    for line in log_content_after_concurrent:
        print(line.strip(), flush=True)

    print("\n--- logging_manager.py Standalone Test Complete ---", flush=True)