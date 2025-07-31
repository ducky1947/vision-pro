import pickle
import os

# Load existing encodings and names
pickle_file = "intruderg_encodings.pkl"

if os.path.exists(pickle_file):
    with open(pickle_file, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)

    # Print the list of names
    print("List of stored names in the pickle file:")
    for name in known_face_names:
        print(name)
else:
    print("Pickle file not found.")
