# face_data_manager.py
import os
import pickle
import face_recognition
import shutil # For deleting person's image folders
import datetime # For saving new face images with timestamp
import cv2 # For saving new face images (specifically for cv2.imwrite)

import config # Import the centralized configuration

def load_known_face_encodings():
    """
    Loads known face encodings, names, and extracts genders from the name format "Name__Gender".
    If the encodings file is not found or empty, it returns empty lists.
    Returns a tuple: (known_face_encodings, known_face_names, known_face_genders).
    """
    known_face_encodings = []
    known_face_names = []
    known_face_genders = []
    try:
        # Use the path from config.py to load the encodings file
        print(f"DEBUG: face_data_manager: Attempting to load known faces from {config.ENCODINGS_FILE}.", flush=True)
        with open(config.ENCODINGS_FILE, "rb") as file:
            known_face_encodings, known_face_names = pickle.load(file)
        print(f"DEBUG: face_data_manager: Loaded {len(known_face_encodings)} existing face encodings.", flush=True)

        # Extract genders from names based on the assumed format "Name__Gender"
        # This logic ensures that gender information is parsed if available.
        for name_with_gender in known_face_names:
            if '__' in name_with_gender:
                try:
                    _, gender = name_with_gender.split('__', 1) # Split only on the first '__'
                    known_face_genders.append(gender.lower())
                except ValueError:
                    # Fallback for malformed names (e.g., "Name__")
                    known_face_genders.append("unknown") 
            else:
                # If no '__' separator, gender is considered unknown for this entry
                known_face_genders.append("unknown") 

    except (FileNotFoundError, EOFError):
        # Handle cases where the encodings file doesn't exist or is empty
        print(f"DEBUG: face_data_manager: {config.ENCODINGS_FILE} not found or empty. Starting with no known faces.", flush=True)
    except Exception as e:
        # Catch any other unexpected errors during loading
        print(f"ERROR: face_data_manager: An unexpected error occurred while loading {config.ENCODINGS_FILE}: {e}", flush=True)

    return known_face_encodings, known_face_names, known_face_genders

def save_known_face_encodings(encodings, names):
    """
    Saves the current known face encodings and their corresponding names to the pickle file.
    """
    try:
        print(f"DEBUG: face_data_manager: Attempting to save {len(encodings)} encodings to {config.ENCODINGS_FILE}.", flush=True)
        with open(config.ENCODINGS_FILE, "wb") as f:
            pickle.dump((encodings, names), f)
        print(f"DEBUG: face_data_manager: Saved encodings successfully.", flush=True)
    except Exception as e:
        # Re-raise the exception to allow the calling function (e.g., in GUI) to handle it
        print(f"ERROR: face_data_manager: Failed to save encodings: {e}", flush=True)
        raise 

def update_encodings_from_dataset():
    """
    Scans the configured dataset path for new person image folders or new images in existing folders,
    encodes any new faces found, and updates the main encoding file.
    Returns a descriptive message string indicating the outcome.
    Raises FileNotFoundError if the dataset base path is not found.
    """
    # Load existing data first so we only process new additions
    known_face_encodings, known_face_names, _ = load_known_face_encodings() 
    
    # Create a set of already processed people (using only the name part before '__')
    # This helps avoid re-processing individuals who are already in the encodings file.
    existing_people_names_only = set(name.split('__')[0] for name in known_face_names if '__' in name)
    # Also add names without gender if they exist in old format
    existing_people_names_only.update(name for name in known_face_names if '__' not in name)


    newly_added_person_count = 0
    total_images_scanned = 0
    faces_encoded_in_this_run = 0

    if not os.path.exists(config.DATASET_BASE_PATH):
        raise FileNotFoundError(f"Dataset base path not found: {config.DATASET_BASE_PATH}. Please check config.py.")

    print(f"DEBUG: face_data_manager: Scanning dataset at {config.DATASET_BASE_PATH} for new faces...", flush=True)
    
    # Iterate through gender subfolders (e.g., 'male', 'female', 'other')
    for gender_folder_name in os.listdir(config.DATASET_BASE_PATH):
        gender_folder_path = os.path.join(config.DATASET_BASE_PATH, gender_folder_name)
        if not os.path.isdir(gender_folder_path):
            continue # Skip if it's not a directory

        # Iterate through person-specific subfolders within each gender folder
        for person_name_in_folder in os.listdir(gender_folder_path):
            person_folder_path = os.path.join(gender_folder_path, person_name_in_folder)
            
            # Skip if it's not a directory or if this person has already been encoded.
            if not os.path.isdir(person_folder_path) or person_name_in_folder in existing_people_names_only:
                print(f"DEBUG: face_data_manager: Skipping '{person_name_in_folder}' (already encoded or not a directory).", flush=True)
                continue

            print(f"DEBUG: face_data_manager: Processing new person: '{person_name_in_folder}' from gender '{gender_folder_name}'.", flush=True)

            person_had_new_images_encoded_this_pass = False
            for filename in os.listdir(person_folder_path):
                image_path = os.path.join(person_folder_path, filename)
                # Only process common image file types
                if not os.path.isfile(image_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue 

                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    total_images_scanned += 1

                    if encodings:
                        # Add only the first encoding found in the image.
                        known_face_encodings.append(encodings[0])
                        # Store name as "PersonName__Gender" for future gender retrieval
                        full_name_with_gender = f"{person_name_in_folder}__{gender_folder_name.lower()}" # Ensure gender is lowercase
                        known_face_names.append(full_name_with_gender)
                        faces_encoded_in_this_run += 1
                        person_had_new_images_encoded_this_pass = True
                        print(f"DEBUG: face_data_manager:    Image '{filename}' encoded successfully.", flush=True)
                    else:
                        print(f"WARNING: face_data_manager:    No face found in '{filename}' for '{person_name_in_folder}'.", flush=True)
                except Exception as img_e:
                    print(f"ERROR: face_data_manager: Failed to process image '{image_path}': {img_e}", flush=True)

            if person_had_new_images_encoded_this_pass:
                newly_added_person_count += 1
                # Add this person's base name to the set of existing people for this run
                existing_people_names_only.add(person_name_in_folder)

    # Save updated data back to the pickle file if any changes occurred
    if newly_added_person_count > 0 or faces_encoded_in_this_run > 0:
        save_known_face_encodings(known_face_encodings, known_face_names)

    # Construct and return a descriptive summary message
    if newly_added_person_count == 0 and faces_encoded_in_this_run == 0:
        return "No new faces found or no new images processed in the dataset."
    elif newly_added_person_count > 0:
        return (f"Successfully encoded {faces_encoded_in_this_run} faces "
                f"for {newly_added_person_count} new people. Total images scanned: {total_images_scanned}.")
    else:
        # This case handles when new images are added to *existing* person folders
        return (f"Processed {total_images_scanned} images. "
                f"No new people added, but {faces_encoded_in_this_run} new faces were encoded "
                f"for existing known people (if more images were added to existing folders).")


def delete_person_from_encodings(person_to_remove):
    """
    Removes a person's data (their face encodings from the pickle file and
    their corresponding image folder from the dataset).
    Expects `person_to_remove` as the full name (e.g., 'John_Doe__male') or just the base name ('John_Doe').
    Raises ValueError if the person is not found in the encodings.
    """
    known_face_encodings, known_face_names, _ = load_known_face_encodings() # Get existing data

    # Determine which full names (e.g., "John_Doe__male") from the list need to be deleted.
    # This handles cases where input might be "John_Doe" but stored as "John_Doe__male".
    names_in_encodings_to_filter_out = []
    
    # Extract the base name (e.g., "John_Doe" from "John_Doe__male") for folder deletion.
    person_base_name = person_to_remove.split('__')[0] 

    found_any_matching_encoding = False
    for name_in_list in known_face_names:
        # Check if the full `person_to_remove` string matches an entry
        # OR if the base name part of the list entry matches the base name of the input.
        if name_in_list == person_to_remove or name_in_list.split('__')[0] == person_base_name:
            names_in_encodings_to_filter_out.append(name_in_list)
            found_any_matching_encoding = True

    if not found_any_matching_encoding:
        raise ValueError(f"No person named '{person_to_remove}' found in encodings to delete.")

    # Filter out the encodings and names for the person(s) to remove
    filtered_encodings = []
    filtered_names = []
    removed_encoding_count = 0

    for encoding, name in zip(known_face_encodings, known_face_names):
        if name not in names_in_encodings_to_filter_out:
            filtered_encodings.append(encoding)
            filtered_names.append(name)
        else:
            removed_encoding_count += 1

    # Save the updated (filtered) data back to the pickle file
    save_known_face_encodings(filtered_encodings, filtered_names)
    
    # --- Delete corresponding image folder(s) from the dataset ---
    # The folder structure is assumed to be DATASET_BASE_PATH/gender_folder/person_name_folder
    
    deleted_folder_paths = []
    
    # If the input `person_to_remove` contains gender (e.g., 'John_Doe__male'),
    # we can try to find the specific gender subfolder directly.
    if '__' in person_to_remove: 
        name_part, gender_part = person_to_remove.split('__', 1) # Split only once
        folder_to_delete = os.path.join(config.DATASET_BASE_PATH, gender_part, name_part)
        if os.path.exists(folder_to_delete):
            try:
                shutil.rmtree(folder_to_delete) # Delete the directory and its contents
                deleted_folder_paths.append(folder_to_delete)
                print(f"DEBUG: face_data_manager: Removed image folder: {folder_to_delete}", flush=True)
            except Exception as e:
                # Log the error but don't re-raise, as encoding deletion was successful
                print(f"ERROR: face_data_manager: Failed to delete folder '{folder_to_delete}': {e}", flush=True)
        else:
            print(f"WARNING: face_data_manager: Image folder for '{person_to_remove}' not found at '{folder_to_delete}'.", flush=True)
    else: 
        # If only the base name was provided (e.g., 'John_Doe'), search all gender subfolders.
        print(f"DEBUG: face_data_manager: Searching for image folders for '{person_base_name}' across all genders.", flush=True)
        # Iterate through actual gender directories found in the dataset base path
        for gender_dir in os.listdir(config.DATASET_BASE_PATH): 
            potential_folder = os.path.join(config.DATASET_BASE_PATH, gender_dir, person_base_name)
            if os.path.isdir(potential_folder): # Check if the folder exists
                try:
                    shutil.rmtree(potential_folder)
                    deleted_folder_paths.append(potential_folder)
                    print(f"DEBUG: face_data_manager: Removed image folder: {potential_folder}", flush=True)
                except Exception as e:
                    print(f"ERROR: face_data_manager: Failed to delete folder {potential_folder}: {e}", flush=True)
                
        if not deleted_folder_paths:
            print(f"WARNING: face_data_manager: No image folders found for '{person_to_remove}'.", flush=True)

    # Return a summary message about the deletion
    return (f"Successfully removed {removed_encoding_count} entries for '{person_to_remove}' from encodings. "
            f"Image folders deleted: {', '.join(deleted_folder_paths) or 'None'}.")


def save_new_face_image(frame, person_name, gender):
    """
    Saves a captured face image (OpenCV frame) to the appropriate person's folder
    within the dataset based on provided name and gender.
    The folder structure is DATASET_BASE_PATH/gender/person_name.
    Returns the absolute path where the image was saved.
    """
    # Construct the full path to the person's folder
    person_folder_path = os.path.join(config.DATASET_BASE_PATH, gender.lower(), person_name)
    print(f"DEBUG: face_data_manager: Creating person folder if not exists: {person_folder_path}", flush=True)
    os.makedirs(person_folder_path, exist_ok=True) # Ensure the directory exists

    # Generate a unique filename using timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"{person_name}_{timestamp}.jpg"
    image_path = os.path.join(person_folder_path, image_filename)
    
    print(f"DEBUG: face_data_manager: Saving captured image to: {image_path}", flush=True)
    cv2.imwrite(image_path, frame) # Save the image using OpenCV
    
    return image_path # Return the path for logging or confirmation

# Example of how this module could be tested standalone (optional)
if __name__ == "__main__":
    print("\n--- Testing face_data_manager.py (Standalone) ---", flush=True)

    # Test 1: Load known faces
    print("\n--- Test 1: Loading known faces ---", flush=True)
    encs, names, genders = load_known_face_encodings() # This call should now work correctly
    print(f"Initial load: {len(encs)} encodings found. Names: {names}", flush=True)
    print(f"Genders: {genders}", flush=True)


    # Test 2: Update encodings from dataset
    # For a meaningful test, ensure you have a new person's folder with images
    # in your DATASET_BASE_PATH that's not already in encodings1.pkl.
    # Example: E:\sem 6\face recg wts\compact feed\images\male\NewPerson\new_face.jpg
    print("\n--- Test 2: Running update_encodings_from_dataset ---", flush=True)
    try:
        update_message = update_encodings_from_dataset()
        print(f"Update result: {update_message}", flush=True)
        encs_after_update, names_after_update, genders_after_update = load_known_face_encodings()
        print(f"After update: {len(encs_after_update)} encodings total. Names: {names_after_update}", flush=True)
        print(f"Genders after update: {genders_after_update}", flush=True)
    except FileNotFoundError as e:
        print(f"ERROR: Update failed: {e}", flush=True)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during update: {e}", flush=True)


    # Test 3: Delete a person (BE CAREFUL, this deletes files!)
    # Choose a name that actually exists in your encodings1.pkl (e.g., 'kewal__male' or 'Intruder_1').
    print("\n--- Test 3: Deleting a person ---", flush=True)
    person_to_delete_test = input("\nEnter name to delete (e.g., 'John_Doe__male' or 'Intruder_1'), or leave empty to skip: ").strip()
    if person_to_delete_test:
        confirm = input(f"Are you SURE you want to delete '{person_to_delete_test}' and its image folder(s)? (yes/no): ").lower()
        if confirm == 'yes':
            print(f"\n--- Attempting to delete '{person_to_delete_test}' ---", flush=True)
            try:
                delete_message = delete_person_from_encodings(person_to_delete_test)
                print(f"Deletion result: {delete_message}", flush=True)
                encs_after_delete, names_after_delete, genders_after_delete = load_known_face_encodings()
                print(f"After deletion: {len(encs_after_delete)} encodings total. Names: {names_after_delete}", flush=True)
                print(f"Genders after deletion: {genders_after_delete}", flush=True)
            except ValueError as e:
                print(f"ERROR: Deletion failed: {e}", flush=True)
            except Exception as e:
                print(f"ERROR: An unexpected error occurred during deletion: {e}", flush=True)
        else:
            print("Deletion cancelled by user.", flush=True)
    else:
        print("Skipping person deletion test.", flush=True)

    print("\n--- face_data_manager.py Standalone Test Complete ---", flush=True)