"""Publish the public demo to Hugging Face Spaces.

Run from the orbital_eye_ai/ folder (e.g. in Codespaces):

    pip install huggingface_hub
    python deploy/push_to_hf.py --space YOUR_HF_USERNAME/orbital-eye-ai

The script
  1. asks for a Hugging Face access token (write permission) - typed input is hidden,
  2. creates the Space (Docker) if it does not exist,
  3. stores MODEL_URL (your private Google Drive link to retinal_unet.pt) and a random
     SECRET_KEY as Space *secrets* - they are never written to any file or to GitHub,
  4. uploads the app code (no database, uploads, datasets or model weights).

Run it again any time to publish the latest code. Add --skip-secrets to keep existing secrets.
"""
from __future__ import annotations

import argparse
import getpass
import secrets
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
IGNORE = [
    ".venv/*", "venv/*", "**/__pycache__/*", "*.pyc", ".pytest_cache/*", ".env", "*.db",
    "uploads/*", "models/*.pt", "models/*.pth", "data/datasets/*", "training/*", "tests/*",
    "deploy/*", "README.md",
]


def main():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("Install the client first:  pip install huggingface_hub")

    p = argparse.ArgumentParser()
    p.add_argument("--space", required=True, help="e.g. svetli3020/orbital-eye-ai")
    p.add_argument("--skip-secrets", action="store_true", help="keep the Space secrets as they are")
    p.add_argument("--private", action="store_true", help="create the Space as private")
    args = p.parse_args()

    token = getpass.getpass("Hugging Face token (write access, input hidden): ").strip()
    api = HfApi(token=token)
    user = api.whoami()["name"]
    print(f"Logged in to Hugging Face as {user}")

    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    print(f"Space ready: https://huggingface.co/spaces/{args.space}")

    if not args.skip_secrets:
        model_url = getpass.getpass(
            "Google Drive share link of retinal_unet.pt (input hidden; press Enter to skip = heuristic only): ").strip()
        if model_url:
            api.add_space_secret(args.space, "MODEL_URL", model_url)
            print("Saved MODEL_URL as a Space secret.")
        api.add_space_secret(args.space, "SECRET_KEY", secrets.token_hex(32))
        print("Saved a random SECRET_KEY as a Space secret.")

    api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(APP_DIR),
                      ignore_patterns=IGNORE, commit_message="Publish ORBITAL EYE AI demo")
    api.upload_file(repo_id=args.space, repo_type="space",
                    path_or_fileobj=str(APP_DIR / "deploy" / "space_README.md"), path_in_repo="README.md",
                    commit_message="Space card")
    print("\nUploaded. The Space is now building (about 5-10 minutes the first time).")
    print(f"Watch it here: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
