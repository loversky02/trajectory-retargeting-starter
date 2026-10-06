"""Pinned MuJoCo Menagerie robot models: one-time fetch and offline verification."""
from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COMMIT = "f054586a8e90465d49ee5be15335c4a0c7f57caf"
REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie"
DIRECTORIES = ("franka_emika_panda", "universal_robots_ur5e", "robotiq_2f85")
XML = {
    "panda": "franka_emika_panda/panda.xml",
    "ur5e": "universal_robots_ur5e/ur5e.xml",
    "2f85": "robotiq_2f85/2f85.xml",
}
RECEIPT = "receipt.json"


def _git_blob_sha(content: bytes) -> str:
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def _curl(url: str) -> bytes:
    completed = subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error", "--connect-timeout", "15",
         "--max-time", "120", "--retry", "4", url],
        check=True, capture_output=True)
    return completed.stdout


def fetch_models(cache: Path) -> dict:
    """Download the three robot folders at COMMIT and write a SHA-256 receipt.

    Files already present are kept only if their git blob SHA matches upstream.
    """
    cache = Path(cache)
    tree = json.loads(_curl(f"https://api.github.com/repos/google-deepmind/mujoco_menagerie/git/trees/{COMMIT}?recursive=1"))
    if tree.get("truncated") or "tree" not in tree:
        raise RuntimeError("upstream Menagerie tree listing is incomplete")
    blobs = {item["path"]: item["sha"] for item in tree["tree"] if item["type"] == "blob"
             and item["path"].split("/")[0] in DIRECTORIES
             and not item["path"].endswith((".png", ".gif", ".jpg"))}
    missing = [path for path in XML.values() if path not in blobs]
    if missing:
        raise RuntimeError(f"upstream is missing expected model files: {missing}")

    def fetch_one(relative: str) -> tuple[str, str]:
        destination = cache / relative
        if destination.exists() and _git_blob_sha(destination.read_bytes()) == blobs[relative]:
            content = destination.read_bytes()
        else:
            content = _curl(f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{COMMIT}/{relative}")
            if _git_blob_sha(content) != blobs[relative]:
                raise ValueError(f"downloaded file differs from upstream blob: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        return relative, hashlib.sha256(content).hexdigest()

    with ThreadPoolExecutor(8) as pool:
        hashes = dict(pool.map(fetch_one, sorted(blobs)))
    receipt = {"repository": REPOSITORY, "commit": COMMIT, "files": hashes}
    (cache / RECEIPT).write_text(json.dumps(receipt, indent=1, sort_keys=True))
    return receipt


def verify_models(cache: Path) -> str:
    """Check every receipt file offline; return the receipt's own SHA-256."""
    cache = Path(cache)
    receipt_path = cache / RECEIPT
    if not receipt_path.exists():
        raise FileNotFoundError(
            f"no model receipt in {cache}; run `python -m physics_replay fetch-models --cache {cache}` once")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("commit") != COMMIT:
        raise ValueError(f"model cache commit {receipt.get('commit')} != pinned {COMMIT}")
    for relative, digest in receipt["files"].items():
        path = cache / relative
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"model file missing or modified: {relative}")
    return hashlib.sha256(receipt_path.read_bytes()).hexdigest()
