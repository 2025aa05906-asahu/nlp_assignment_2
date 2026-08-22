"""
Downloads the trained checkpoint (models/nmt_model.pt) from Hugging Face Hub.

This mirrors run_pipeline.py's role for the data pipeline: the checkpoint is
too large (54 MB) to include directly in Group112_Code.zip (10 MB upload
limit), so this script fetches the exact checkpoint used for this submission
and verifies its integrity, rather than requiring a full retrain to reproduce
a working application (per the assignment's General Instructions).

Usage:
    python download_model.py
    python download_model.py --repo_id your-username/group112-nmt-en-hi
"""

import argparse
import hashlib
import os
import shutil

from huggingface_hub import hf_hub_download

REPO_ID = "buvika/group112-nmt-en-hi"   # <-- set to the actual HF repo id after upload
FILENAME = "nmt_model.pt"
DEST_PATH = os.path.join("models", "nmt_model.pt")
EXPECTED_SHA256 = "2e40843c3ddaf9a3875a0bb19620e99bb56e745037824aea22db1efe82875b6"
EXPECTED_SIZE_BYTES = 56139874  # ~53.5 MB


def parse_args():
    p = argparse.ArgumentParser(description="Download the trained NMT checkpoint from Hugging Face Hub")
    p.add_argument("--repo_id", type=str, default=REPO_ID,
                    help="Hugging Face Hub repo id holding nmt_model.pt.")
    p.add_argument("--skip_verify", action="store_true",
                    help="Skip the SHA-256/size check (not recommended).")
    return p.parse_args()


def sha256_of(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def already_present_and_valid(path):
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) != EXPECTED_SIZE_BYTES:
        return False
    return sha256_of(path) == EXPECTED_SHA256


def main():
    """Fetches models/nmt_model.pt from Hugging Face Hub and verifies it
    matches the checkpoint used for this submission's report and eval_report.json.
    """
    args = parse_args()
    os.makedirs("models", exist_ok=True)

    print("[1/3] Checking for an existing local checkpoint...")
    if already_present_and_valid(DEST_PATH):
        print(f"{DEST_PATH} is already present and verified. Nothing to download.")
        return

    print(f"[2/3] Downloading {FILENAME} from Hugging Face repo '{args.repo_id}'...")
    cached_path = hf_hub_download(repo_id=args.repo_id, filename=FILENAME)
    shutil.copy(cached_path, DEST_PATH)

    print("[3/3] Verifying checksum...")
    if args.skip_verify:
        print("Skipped verification (--skip_verify was passed).")
    else:
        size = os.path.getsize(DEST_PATH)
        digest = sha256_of(DEST_PATH)
        if size != EXPECTED_SIZE_BYTES or digest != EXPECTED_SHA256:
            raise SystemExit(
                f"Verification FAILED.\n"
                f"  size:   got {size}, expected {EXPECTED_SIZE_BYTES}\n"
                f"  sha256: got {digest}, expected {EXPECTED_SHA256}\n"
                f"The download may be corrupt or the wrong file was uploaded to the repo. "
                f"Try re-running this script."
            )

    print("\n--- Download Summary ---")
    print(f"Checkpoint saved to: {DEST_PATH}")
    print(f"Size: {os.path.getsize(DEST_PATH)} bytes")
    print("Verified: OK" if not args.skip_verify else "Verified: skipped")
    print("Ready to run: streamlit run app.py | python infer.py \"...\" | python evaluate.py")


if __name__ == "__main__":
    main()