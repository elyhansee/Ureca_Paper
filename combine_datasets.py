#!/usr/bin/env python3
"""
combine_datasets.py
===================
Consolidates synthetic tool-use speech waveforms and labels into your main 
evaluation dataset directory, merging metadata maps into a single labels.json.
"""

import json
import shutil
from pathlib import Path

# Configure paths
DATASET_DIR = Path("./dataset")
TOOL_DIR = Path("./tool_test_audio")

def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load existing labels from the main dataset folder (if any exist)
    main_labels = {}
    main_labels_path = DATASET_DIR / "labels.json"
    if main_labels_path.exists():
        try:
            with open(main_labels_path, "r") as f:
                main_labels = json.load(f)
            print(f"Loaded {len(main_labels)} existing labels from {main_labels_path}")
        except Exception as e:
            print(f"[WARN] Failed to read existing labels.json: {e}")
    else:
        print("No existing labels.json found in ./dataset. Starting fresh.")

    # 2. Load the newly generated tool labels
    tool_labels_path = TOOL_DIR / "tool_labels.json"
    if tool_labels_path.exists():
        try:
            with open(tool_labels_path, "r") as f:
                tool_labels = json.load(f)
            print(f"Loaded {len(tool_labels)} tool labels from {tool_labels_path}")
            
            # Merge the dictionaries (tool labels will update or append to main labels)
            main_labels.update(tool_labels)
        except Exception as e:
            print(f"[ERROR] Failed to read tool_labels.json: {e}")
            return
    else:
        print(f"[ERROR] Could not find {tool_labels_path}. Run generate_test_audio.py first.")
        return

    # 3. Save the consolidated labels dictionary back to dataset/labels.json
    try:
        with open(main_labels_path, "w") as f:
            json.dump(main_labels, f, indent=2)
        print(f"Consolidated labels ({len(main_labels)} total) written to {main_labels_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save merged labels.json: {e}")
        return

    # 4. Copy the .wav audio assets into the main dataset directory
    copied_count = 0
    for wav_file in TOOL_DIR.glob("*.wav"):
        dest_file = DATASET_DIR / wav_file.name
        try:
            shutil.copy(wav_file, dest_file)
            copied_count += 1
        except Exception as e:
            print(f"[WARN] Failed to copy {wav_file.name}: {e}")
            
    print(f"Successfully copied {copied_count} wave files to {DATASET_DIR}")
    print("\nDataset consolidation complete!")

if __name__ == "__main__":
    main()
