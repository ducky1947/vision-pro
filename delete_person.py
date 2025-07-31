import sys
import pickle
import os

if len(sys.argv) < 2:
    print(" No name provided to delete.")
    exit()

person_to_remove = sys.argv[1]  # ✅ Get name from argument
encoding_file = "encodings1.pkl"

# Check if encoding file exists
if not os.path.exists(encoding_file):
    print(" Encoding file not found.")
    exit()

# Load the existing data
with open(encoding_file, "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)

if person_to_remove not in known_face_names:
    print(f"No person named '{person_to_remove}' found in encodings.")
    exit()


# Filter out the person to remove
filtered_encodings = []
filtered_names = []

for encoding, name in zip(known_face_encodings, known_face_names):
    if name != person_to_remove:
        filtered_encodings.append(encoding)
        filtered_names.append(name)

# Save updated data
with open(encoding_file, "wb") as f:
    pickle.dump((filtered_encodings, filtered_names), f)

print(f" Removed '{person_to_remove}' from encodings.")
#print(f" {len(filtered_names)} people remaining.")
