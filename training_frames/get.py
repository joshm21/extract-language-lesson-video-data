import shutil
from pathlib import Path

# Setup paths
src_dir = Path('.')  # Search starting from current directory
dest_dir = Path('training_frames')
dest_dir.mkdir(exist_ok=True)

# Target pattern
pattern = "*_01_extract.at_current_timestamp.jpg"

print(f"Searching for files matching: {pattern}")

for file_path in src_dir.rglob(pattern):
    # grandparent/parent/filename
    video_id = file_path.parent.parent.parent.name

    # Split the filename to get the part before "01_extract..."
    prefix = file_path.name.split("01_extract")[0]

    # Construct new name: {video_id}-{prefix}.jpg
    # We strip trailing underscores from the prefix if they exist
    new_name = f"{video_id}~~{prefix.rstrip('_')}.jpg"

    # Perform the copy
    shutil.copy2(file_path, dest_dir / new_name)
    print(f"Copied: {file_path.name} -> {new_name}")

print("Done!")
