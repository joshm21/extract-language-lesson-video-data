# COLAB:
# https://colab.research.google.com/drive/1Mm286EG-YHJZpJyJrKg4HG5dKHYax-2T?authuser=1#scrollTo=Xa7WsCNt8e9K

# BLOCK 1
from google.colab import drive, auth
import os

# Mount Drive to access files like a local filesystem
drive.mount('/content/drive')

# Install whisper
!pip install git+https://github.com/linto-ai/whisper-timestamped

# BLOCK 2
import torch
import whisper_timestamped as whisper
# Check if model is already loaded to avoid redundant work
if 'model' not in globals():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = whisper.load_model("large-v3", device=device)
    print("Model loaded and ready!")
else:
    print("Model already in memory. Skipping reload.")

# BLOCK 3
import io
import json
import torch
import whisper_timestamped as whisper
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# --- CONFIGURATION ---
# List your File IDs here (or load them from a text file on Drive)
# Instead of FILE_IDS = ['ID1', 'ID2'...]
ID_LIST_PATH = '/content/drive/MyDrive/Transcripts/file_ids.txt'

with open(ID_LIST_PATH, 'r') as f:
    # Read lines and strip out whitespace/newlines
    FILE_IDS = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(FILE_IDS)} files to process.")

SAVE_FOLDER = '/content/drive/MyDrive/Transcripts/' # Where JSONs will live
# ---------------------

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Authenticate for Drive API (to download files)
auth.authenticate_user()
drive_service = build('drive', 'v3')

def download_file(file_id, output_path):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(output_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    print(f"Downloaded: {output_path}")

def process_files():
    for index, fid in enumerate(FILE_IDS):
        output_filename = f"transcript_{fid}.json"
        full_output_path = os.path.join(SAVE_FOLDER, output_filename)
        temp_video_path = f"temp_{fid}.mp4"

        # 1. RESUME LOGIC: Check if transcript already exists
        if os.path.exists(full_output_path):
            print(f"Skipping {fid} - Transcript already exists.")
            continue

        try:
            print(f"Starting process for: {fid} ({index+1}/{len(FILE_IDS)})")

            # 2. Download
            download_file(fid, temp_video_path)

            # 3. Transcribe
            result = whisper.transcribe(model, temp_video_path, language="ka")

            # 4. Save to Drive
            with open(full_output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"Successfully saved: {output_filename}")

        except Exception as e:
            print(f"Error processing {fid}: {e}")
            continue # Move to the next file if one fails

        finally:
            # 5. CLEANUP: Delete video to save Colab disk space
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

process_files()
