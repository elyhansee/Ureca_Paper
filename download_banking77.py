#!/usr/bin/env python3
"""
download_banking77.py
=====================
Downloads the PolyAI/banking77 dataset from Hugging Face and converts
every text query to a .wav audio file using gTTS (Google Text-to-Speech).

Banking77 has 13,083 customer service queries across 77 banking intents.
ALL of them are general questions (no phone-number lookups), so every output
file is named  general_<intent>_<id>.wav  — which tells the benchmark pipeline
these are direct-answer samples (no tool call required).

Usage:
    # Full dataset (10,003 train + 3,080 test = 13,083 files)
    python download_banking77.py --out ./audio_banking77

    # Quick smoke test — only first 50 samples
    python download_banking77.py --out ./audio_banking77 --max 50

    # Only the test split (3,080 samples — good default for benchmarking)
    python download_banking77.py --out ./audio_banking77 --split test

    # Slower but higher quality TTS (uses pyttsx3 offline engine)
    python download_banking77.py --out ./audio_banking77 --tts pyttsx3

Requirements:
    pip install datasets gTTS pydub soundfile
    # For pyttsx3 backend:
    pip install pyttsx3
    # On Linux you also need: sudo apt-get install espeak

Outputs:
    ./audio_banking77/
        general_activate_my_card_00001.wav
        general_card_arrival_00002.wav
        general_exchange_rate_00003.wav
        ...
        banking77_manifest.csv        ← maps filename → original text + intent
        labels.json                   ← ground-truth file for infer_label()
"""

import os
import re
import json
import csv
import time
import argparse
import tempfile
from pathlib import Path
from typing import Optional

# ── Intent label map (from dataset card) ─────────────────────────────────────
INTENT_NAMES = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working",
    "card_payment_fee_charged", "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
    "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge",
    "exchange_rate", "exchange_via_app", "extra_charge_on_statement", "failed_transfer",
    "fiat_currency_support", "get_disposable_virtual_card", "get_physical_card",
    "getting_spare_card", "getting_virtual_card", "lost_or_stolen_card",
    "lost_or_stolen_phone", "order_physical_card", "passcode_forgotten",
    "pending_card_payment", "pending_cash_withdrawal", "pending_top_up",
    "pending_transfer", "pin_blocked", "receiving_money", "refund_not_showing_up",
    "request_refund", "reverted_card_payment", "supported_cards_and_currencies",
    "terminate_account", "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
    "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
    "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
    "verify_top_up", "virtual_card_not_working", "visa_or_mastercard",
    "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]


# ── TTS Backends ──────────────────────────────────────────────────────────────
def tts_gtts(text: str, out_path: Path):
    """Google TTS — requires internet, sounds natural, free."""
    from gtts import gTTS
    tts = gTTS(text=text, lang="en", slow=False)
    # gTTS saves as mp3, convert to wav via pydub
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    tts.save(tmp_path)
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(tmp_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(str(out_path), format="wav")
    finally:
        os.unlink(tmp_path)


def tts_pyttsx3(text: str, out_path: Path):
    """Offline TTS — no internet needed, robotic voice, always works."""
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)   # words per minute
    engine.setProperty("volume", 0.9)
    engine.save_to_file(text, str(out_path))
    engine.runAndWait()


TTS_BACKENDS = {
    "gtts":    tts_gtts,
    "pyttsx3": tts_pyttsx3,
}


# ── Filename sanitiser ────────────────────────────────────────────────────────
def safe_stem(intent: str, idx: int) -> str:
    """
    Returns the filename stem WITHOUT extension.
    Format: general_<intent>_<zero-padded-id>
    The 'general_' prefix tells the benchmark pipeline no tool call is needed.
    """
    clean_intent = re.sub(r"[^a-z0-9_]", "", intent.lower().replace(" ", "_"))
    return f"general_{clean_intent}_{idx:05d}"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Download Banking77 → WAV files")
    parser.add_argument("--out",   default="./audio_banking77",
                        help="Output directory for WAV files")
    parser.add_argument("--split", default="test",
                        choices=["train", "test", "all"],
                        help="Which split to download (default: test = 3,080 samples)")
    parser.add_argument("--max",   type=int, default=0,
                        help="Max samples to convert (0 = all)")
    parser.add_argument("--tts",   default="gtts",
                        choices=["gtts", "pyttsx3"],
                        help="TTS backend (default: gtts)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds to wait between gTTS calls (avoid rate limiting)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Download dataset ──────────────────────────────────────────────────────
    print("Downloading PolyAI/banking77 from Hugging Face...")
    from datasets import load_dataset
    if args.split == "all":
        ds_train = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True)
        ds_test  = load_dataset("PolyAI/banking77", split="test",  trust_remote_code=True)
        # Merge — label is an integer, text is a string
        samples = [{"text": r["text"], "label": r["label"], "source": "train"}
                   for r in ds_train]
        samples += [{"text": r["text"], "label": r["label"], "source": "test"}
                    for r in ds_test]
    else:
        ds = load_dataset("PolyAI/banking77", split=args.split, trust_remote_code=True)
        samples = [{"text": r["text"], "label": r["label"], "source": args.split}
                   for r in ds]

    if args.max > 0:
        samples = samples[:args.max]

    print(f"  {len(samples)} samples to convert (split={args.split}, tts={args.tts})")

    tts_fn = TTS_BACKENDS[args.tts]

    # ── Convert to WAV ────────────────────────────────────────────────────────
    manifest_rows = []
    labels_json   = {}
    failed        = []

    for idx, sample in enumerate(samples, 1):
        intent_name = INTENT_NAMES[sample["label"]]
        stem        = safe_stem(intent_name, idx)
        wav_path    = out_dir / f"{stem}.wav"

        # Skip if already done (resume support)
        if wav_path.exists():
            print(f"  [{idx:>5}/{len(samples)}] SKIP (exists) {stem}.wav")
        else:
            try:
                tts_fn(sample["text"], wav_path)
                print(f"  [{idx:>5}/{len(samples)}] OK  {stem}.wav  |  {sample['text'][:60]}")
                if args.tts == "gtts":
                    time.sleep(args.delay)   # be polite to Google's servers
            except Exception as e:
                print(f"  [{idx:>5}/{len(samples)}] FAIL {stem}: {e}")
                failed.append({"stem": stem, "error": str(e)})
                continue

        # Record in manifest and labels
        manifest_rows.append({
            "filename":  f"{stem}.wav",
            "stem":      stem,
            "intent":    intent_name,
            "label_id":  sample["label"],
            "source":    sample["source"],
            "text":      sample["text"],
        })
        labels_json[stem] = {
            "requires_tool": False,   # banking77 = all general questions
            "phone":         None,
            "customer_id":   None,
            "intent":        intent_name,
            "original_text": sample["text"],
        }

    # ── Write manifest CSV ────────────────────────────────────────────────────
    manifest_path = out_dir / "banking77_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"\nManifest → {manifest_path}  ({len(manifest_rows)} rows)")

    # ── Write labels.json ─────────────────────────────────────────────────────
    labels_path = out_dir / "labels.json"
    labels_path.write_text(json.dumps(labels_json, indent=2))
    print(f"Labels   → {labels_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Done.  {len(manifest_rows)} WAV files in {out_dir}")
    if failed:
        print(f"Failed: {len(failed)} files — check your internet / TTS setup")
    print(f"\nNext step — run the benchmark:")
    print(f"  python benchmark_runner.py \\")
    print(f"      --db ./customer_db \\")
    print(f"      --dataset {out_dir} \\")
    print(f"      --models qwen3omni,nemotron,cascade \\")
    print(f"      --judge bertscore")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
