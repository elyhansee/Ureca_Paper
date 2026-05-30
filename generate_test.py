#!/usr/bin/env python3
"""
generate_test_audio.py
======================
Generates synthetic telecom customer speech queries (audio waveforms) and 
associated ground-truth labels using offline Text-to-Speech (TTS) synthesis.

Core functionality:
  * Automatically initializes the target dataset directory if not present locally.
  * Normalizes voice synthesis properties (e.g., speech rate) using the pyttsx3 
    multi-platform text-to-speech engine.
  * Expands customer demographic targets and voice intents (20 templates × 5 profiles) 
    using a Cartesian product to generate 100 distinct dialogue contexts.
  * Queues synthetic voice renders asynchronously and batches wave file generation 
    via pyttsx3's queue loop to prevent engine blockages.
  * Generates an aligned, structured metadata label file mapping each synthesized 
    audio asset to its respective intent, phone number, customer ID, and text.

Prerequisites (Inputs):
  - A Python environment with pyttsx3 installed (`pip install pyttsx3`)
  - Platform-native TTS libraries installed on the host OS:
    * Windows: SAPI5 (Standard out-of-the-box)
    * macOS: NSSpeechSynthesizer (Standard out-of-the-box)
    * Linux: espeak-ng (`sudo apt-get install espeak-ng`)

Outputs:
  - ./tool_test_audio/                     – Target workspace for synthesized files
  - ./tool_test_audio/tool_mass_*.wav      – 100 synthesized audio query clips
  - ./tool_test_audio/tool_labels.json     – Ground-truth evaluation metadata maps

Usage:
    python generate_test_audio.py
"""

import json
import pyttsx3
import itertools
from pathlib import Path

# 1. Changed to a local folder name
DATASET_DIR = Path("./tool_test_audio")

CUSTOMERS = [
    {"id": "TEST_001", "phone": "+12345678901"},
    {"id": "TEST_002", "phone": "+19876543210"},
    {"id": "TEST_003", "phone": "+11223344550"},
    {"id": "TEST_004", "phone": "+14155550199"},
    {"id": "TEST_005", "phone": "+6591234567"},
]

TEMPLATES = [
    {"intent": "card_arrival", "text": "Hi, I ordered a new card but it hasn't arrived yet. Could you check the status for {phone}?"},
    {"intent": "card_not_working", "text": "My card was just declined at the store. The number on my account is {phone}. What is going on?"},
    {"intent": "change_pin", "text": "I forgot my PIN and need to reset it. My phone number is {phone}."},
    {"intent": "compromised_card", "text": "Someone made a fraudulent charge on my account! Please freeze the card for {phone} immediately!"},
    {"intent": "contactless_not_working", "text": "The tap to pay feature on my card stopped working. Can you look up {phone} and send a replacement?"},
    {"intent": "edit_personal_details", "text": "I need to update my email address. My account is under the number {phone}."},
    {"intent": "declined_transfer", "text": "I tried to send money but the transfer was declined. Can you check {phone} to see why?"},
    {"intent": "pending_transfer", "text": "My rent transfer has been pending for three days. The account phone number is {phone}."},
    {"intent": "lost_or_stolen_card", "text": "I lost my wallet at the park. Please cancel the card connected to {phone}."},
    {"intent": "order_physical_card", "text": "I'd like to upgrade from a virtual card to a physical one. My number is {phone}."},
    {"intent": "refund_not_showing_up", "text": "A merchant issued a refund yesterday but I don't see it. Check the account for {phone}."},
    {"intent": "terminate_account", "text": "I want to close my account completely. The phone number is {phone}."},
    {"intent": "top_up_failed", "text": "I tried to top up my account but it failed. Could you look into {phone}?"},
    {"intent": "transaction_charged_twice", "text": "I got double charged for my coffee this morning! The number is {phone}."},
    {"intent": "verify_my_identity", "text": "The app is asking me to verify my identity again. My phone number is {phone}."},
    {"intent": "wrong_amount_of_cash_received", "text": "The ATM shortchanged me! Please look up my account at {phone}."},
    {"intent": "supported_cards_and_currencies", "text": "I'm traveling next week. Can you check if the plan for {phone} supports foreign transactions?"},
    {"intent": "passcode_forgotten", "text": "I got locked out of the app because I forgot my passcode. Number is {phone}."},
    {"intent": "getting_virtual_card", "text": "How do I generate a disposable virtual card? The account is under {phone}."},
    {"intent": "request_refund", "text": "I was charged an ATM fee that I want refunded. My phone number is {phone}."}
]

def main():
    # 2. Tell Python to automatically create this folder on your Windows computer
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    engine = pyttsx3.init()
    engine.setProperty("rate", 155) 
    
    # 3. Rename to tool_labels.json to avoid overwriting your cluster file
    labels_path = DATASET_DIR / "tool_labels.json"
    
    if labels_path.exists():
        with open(labels_path, "r") as f:
            labels = json.load(f)
    else:
        labels = {}

    print(f"Queueing 100 custom test audio files for {DATASET_DIR}...")
    
    counter = 1
    for customer, template in itertools.product(CUSTOMERS, TEMPLATES):
        
        final_text = template["text"].format(phone=customer["phone"])
        stem = f"tool_mass_{counter:03d}"
        wav_path = DATASET_DIR / f"{stem}.wav"
        
        if not wav_path.exists():
            # Queue the file, but DO NOT call runAndWait() here!
            engine.save_to_file(final_text, str(wav_path))
            print(f"  [{counter:>3}/100] Queued  {stem}.wav | {template['intent']}")
        
        labels[stem] = {
            "requires_tool": True,
            "phone": customer["phone"],
            "customer_id": customer["id"],
            "intent": template["intent"],
            "original_text": final_text
        }
        
        counter += 1

    # 4. Tell the engine to process the entire queue of 100 files at once!
    print("\nProcessing all audio files now (this will take a few seconds)...")
    engine.runAndWait() 

    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)
        
    print(f"\nDone! Folder '{DATASET_DIR}' is ready to be zipped and transferred.")

if __name__ == "__main__":
    main()
