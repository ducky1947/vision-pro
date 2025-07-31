import random
from datetime import datetime, timedelta

# Output file name
output_file = "cctv_log.txt"

# Fixed names and genders
known_people = [
    {"name": "kewal", "gender": "male"},
    {"name": "chandrakant", "gender": "male"},
    {"name": "chaitali", "gender": "female"},
]

# Start time
base_time = datetime(2025, 7, 31, 14, 27, 40)

# Header row
header = "NAME                      | GENDER   | DAY    | DATE         | TIME     | IMAGE_LINK\n"
separator = "-" * 26 + "+" + "-" * 10 + "+" + "-" * 8 + "+" + "-" * 14 + "+" + "-" * 10 + "+" + "-" * 50 + "\n"

# Store lines
lines = [header, separator]

# Generate 100 rows
for _ in range(100):
    base_time += timedelta(seconds=random.randint(3, 15))
    day = base_time.strftime("%a")
    date = base_time.strftime("%Y-%m-%d")
    time_str = base_time.strftime("%H:%M:%S")

    if random.random() < 0.3:
        person = random.choice(known_people)
        name = person["name"]
        gender = person["gender"]
        image_link = "N/A"
    else:
        intruder_id = random.randint(111, 1110)
        name = f"Intruder_{intruder_id}"
        gender = "Intruder"
        rand_num = random.randint(100000, 999999)
        image_link = f"intruders\\intruder_{base_time.strftime('%Y%m%d_%H%M%S')}_{rand_num}.jpg"

    # Format line
    line = f"{name:<26}| {gender:<9}| {day:<6}| {date:<12}| {time_str:<9}| {image_link}"
    lines.append(line)

# Write to file
with open(output_file, "w") as f:
    f.write("\n".join(lines))

print(f"✅ Text log saved to {output_file}")
