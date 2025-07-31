import sys
import pickle
import os

encoding_file = "intruder_encodings.pkl"

# Check if encoding file exists
if not os.path.exists(encoding_file):
    print("❌ Encoding file not found.")
    sys.exit()

# Load the existing data
with open(encoding_file, "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)

# If the list is empty
if not known_face_names:
    print("⚠️ No registered names found in the encodings file.")
    sys.exit()

# Show list of people
print("✅ Registered names:")
for i, name in enumerate(known_face_names, start=1):
    print(f"  {i}. {name}")

# Ask for name to delete
name_to_delete = input("\nEnter the exact name of the person to delete: ").strip()

if name_to_delete not in known_face_names:
    print(f"❌ No person named '{name_to_delete}' found in encodings.")
    sys.exit()

# Filter out the selected person
filtered_encodings = []
filtered_names = []

for encoding, name in zip(known_face_encodings, known_face_names):
    if name != name_to_delete:
        filtered_encodings.append(encoding)
        filtered_names.append(name)

# Save the updated encodings
with open(encoding_file, "wb") as f:
    pickle.dump((filtered_encodings, filtered_names), f)

print(f"✅ Removed '{name_to_delete}' from encodings.")
print(f"👥 {len(filtered_names)} people remaining.")
