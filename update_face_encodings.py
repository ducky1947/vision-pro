import os
import face_recognition
import pickle

# Paths
dataset_path = r"E:\sem 6\face recg wts\final software\images"
pickle_file = "encodings1.pkl"

# Load existing encodings
if os.path.exists(pickle_file):
    with open(pickle_file, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
    print(f" Loaded {len(known_face_encodings)} existing face encodings.")
else:
    known_face_encodings = []
    known_face_names = []
    print(" No existing encodings found. Creating a new one.")

# Get list of already processed people
existing_people = set(known_face_names)

for gender in os.listdir(dataset_path):
    gender_folder = os.path.join(dataset_path, gender)
    if not os.path.isdir(gender_folder):
        continue

    for person_name in os.listdir(gender_folder):
        person_folder = os.path.join(gender_folder, person_name)
        # Skip if not a folder or already processed
        if not os.path.isdir(person_folder) or any(person_name in name for name in existing_people):
            continue

        print(f" Encoding new person: {person_name}")

        for filename in os.listdir(person_folder):
            image_path = os.path.join(person_folder, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_face_encodings.append(encodings[0])
                full_name = f"{person_name}__{(gender)}"  # Include gender in the name
                known_face_names.append(full_name)
                print(f"    Image '{filename}' encoded.")
            else:
                print(f"    No face found in '{filename}'.")

# Save updated encodings back to pickle file
with open("encodings1.pkl", "wb") as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print(f"\n Updated encodings saved. Total people: {len(set(known_face_names))}")
