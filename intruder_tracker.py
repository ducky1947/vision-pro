import pickle
import face_recognition
import numpy as np
# datetime and os are not strictly needed in intruder_tracker anymore since it's just ID management.
# They can be removed if you want, but keeping them doesn't hurt.
import datetime
import os

import config # Import the centralized configuration

# The intruder_encodings.pkl will store (list_of_encodings, list_of_intruder_ids).
# All other metadata (first_seen, last_seen, count, image_path) is managed externally by camera_stream.

def load_intruder_db():
    """
    Loads intruder encodings and their corresponding IDs from the pickle file.
    Returns a tuple: (list_of_encodings, list_of_intruder_ids).
    Handles FileNotFoundError and EOFError for new/empty files.
    """
    try:
        print(f"DEBUG: intruder_tracker: Attempting to load from {config.INTRUDER_ENCODINGS_FILE}", flush=True)
        with open(config.INTRUDER_ENCODINGS_FILE, "rb") as f:
            encodings, ids = pickle.load(f)
        print(f"DEBUG: intruder_tracker: Loaded {len(encodings)} intruder encodings and {len(ids)} IDs.", flush=True)
        return encodings, ids
    except (FileNotFoundError, EOFError):
        print(f"DEBUG: intruder_tracker: {config.INTRUDER_ENCODINGS_FILE} not found or empty. Starting with no known intruders.", flush=True)
        return [], [] # Return empty lists for encodings and IDs
    except Exception as e:
        print(f"ERROR: intruder_tracker: Error loading {config.INTRUDER_ENCODINGS_FILE}: {e}", flush=True)
        return [], []

def save_intruder_db(encodings, ids):
    """
    Saves intruder encodings and their corresponding IDs to the pickle file.
    """
    print(f"DEBUG: intruder_tracker: Attempting to save {len(encodings)} encodings and {len(ids)} IDs to {config.INTRUDER_ENCODINGS_FILE}", flush=True)
    with open(config.INTRUDER_ENCODINGS_FILE, "wb") as f:
        pickle.dump((encodings, ids), f)
    print(f"DEBUG: intruder_tracker: Saved intruder data successfully.", flush=True)

def match_or_add_intruder(new_encoding, threshold=config.RECOGNITION_TOLERANCE):
    """
    Matches a new face encoding against known intruders or assigns it a new intruder ID.
    This function's sole responsibility is ID management (matching or creating a new ID).
    It does NOT update timestamps, counts, or image paths; those are handled by camera_stream.py.

    Args:
        new_encoding (np.array): The face encoding of the detected face.
        threshold (float, optional): The facial recognition tolerance. Defaults to config.RECOGNITION_TOLERANCE.

    Returns:
        str: The ID of the matched or newly assigned intruder (e.g., "Intruder_1").
    """
    encodings, ids = load_intruder_db()
    
    if encodings: # Only proceed if there are existing intruder encodings to compare against
        matches = face_recognition.compare_faces(encodings, new_encoding, tolerance=threshold)
        # face_distances are needed to find the BEST match if multiple are True, or for confidence.
        # However, for simple match/no-match based on 'matches', it's not strictly necessary for this logic flow.
        # Let's keep it to ensure consistency with how face_recognition is typically used for best_match_index.
        face_distances = face_recognition.face_distance(encodings, new_encoding)
        
        if True in matches: # If any match is found within the tolerance
            best_match_index = np.argmin(face_distances) # Find the index of the closest match
            matched_id = ids[best_match_index] # Retrieve the existing ID for this best match
            print(f"DEBUG: intruder_tracker: Matched existing intruder: {matched_id}", flush=True)
            
            return matched_id # Return the matched intruder's ID
    
    # If no match was found (or if encodings list was empty), then this is a new intruder.
    new_intruder_id = f"Intruder_{len(ids) + 1}" # Assign a new sequential ID
    encodings.append(new_encoding) # Add the new face encoding to the list
    ids.append(new_intruder_id) # Add the new ID to the list
    
    save_intruder_db(encodings, ids) # Save the updated intruder database to disk
    print(f"DEBUG: intruder_tracker: Added new intruder: {new_intruder_id}", flush=True)
    
    return new_intruder_id # Return the newly assigned intruder ID