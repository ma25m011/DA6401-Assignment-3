"""
Upload the trained checkpoint and vocab.json to Hugging Face Hub.

Setup (once):
    pip install huggingface_hub

Run (set token via env var or pass --token):
    set HF_TOKEN=hf_xxx                              # Windows
    python upload_to_hf.py
    # or
    python upload_to_hf.py --token hf_xxx
"""

import argparse
import os
from huggingface_hub import HfApi, create_repo

REPO_NAME = "da6401-a3-transformer"
CHECKPOINT_PATH = "checkpoints/checkpoint_epoch19.pt"
VOCAB_PATH = "vocab.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="HF write token (or set env HF_TOKEN)")
    parser.add_argument("--user", default=None,
                        help="HF username (auto-detected from token if omitted)")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("No token provided. Set HF_TOKEN env var or pass --token.")

    api = HfApi(token=args.token)
    username = args.user or api.whoami(token=args.token)["name"]
    repo_id = f"{username}/{REPO_NAME}"
    print(f"Using repo: {repo_id}")

    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=args.token)

    for local_path, repo_path in [
        (CHECKPOINT_PATH, "checkpoint_epoch19.pt"),
        (VOCAB_PATH, "vocab.json"),
    ]:
        if not os.path.exists(local_path):
            print(f"SKIP {local_path} — file not found")
            continue
        print(f"Uploading {local_path} ({os.path.getsize(local_path) / 1e6:.1f} MB) -> {repo_id}/{repo_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
        )

    print(f"\nDone. Files available at:")
    print(f"  https://huggingface.co/{repo_id}/resolve/main/checkpoint_epoch19.pt")
    print(f"  https://huggingface.co/{repo_id}/resolve/main/vocab.json")


if __name__ == "__main__":
    main()
