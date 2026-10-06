"""Immutable upstream MuJoCo Menagerie model cache and file receipts."""
from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


COMMIT = "f054586a8e90465d49ee5be15335c4a0c7f57caf"
REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie"
MODELS = {"panda": ("franka_emika_panda", "panda.xml"), "ur5e": ("universal_robots_ur5e", "ur5e.xml")}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download(url: str) -> bytes:
    completed = subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                                "--connect-timeout", "10", "--max-time", "45", "--retry", "3", url],
                               check=True, capture_output=True)
    return completed.stdout


def fetch_models(cache: Path) -> dict:
    if (cache / "receipt.json").exists():
        raise FileExistsError(f"model cache already complete: {cache}")
    api = f"https://api.github.com/repos/google-deepmind/mujoco_menagerie/git/trees/{COMMIT}?recursive=1"
    tree = json.loads(_download(api))
    if tree.get("truncated") or "tree" not in tree:
        raise RuntimeError("upstream tree is incomplete")
    wanted = set()
    upstream_blobs = {}
    for directory, xml in MODELS.values():
        wanted.update((f"{directory}/{xml}", f"{directory}/LICENSE"))
    for item in tree["tree"]:
        path = item["path"]
        if item["type"] == "blob":
            upstream_blobs[path] = item["sha"]
        if item["type"] == "blob" and any(path.startswith(f"{directory}/assets/") for directory, _ in MODELS.values()):
            wanted.add(path)
    cache.mkdir(parents=True, exist_ok=True)
    hashes = {}
    try:
        def fetch_one(relative: str) -> tuple[str, str]:
            destination = cache / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{COMMIT}/{relative}"
            if destination.exists():
                content = destination.read_bytes()
                git_sha = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
                if git_sha != upstream_blobs[relative]:
                    raise ValueError(f"incomplete cache file differs from upstream: {relative}")
            else:
                content = _download(url)
                git_sha = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
                if git_sha != upstream_blobs[relative]:
                    raise ValueError(f"downloaded file differs from upstream: {relative}")
                destination.write_bytes(content)
            return relative, sha256(destination)

        with ThreadPoolExecutor(max_workers=8) as executor:
            for relative, digest in executor.map(fetch_one, sorted(wanted)):
                hashes[relative] = digest
        receipt = {"repository": REPOSITORY, "commit": COMMIT, "files_sha256": hashes,
                   "license_paths": [f"{directory}/LICENSE" for directory, _ in MODELS.values()]}
        (cache / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    except Exception:
        # Keep incomplete cache visible; a human can inspect it. No silent re-fetch/overwrite.
        raise


def verify_models(cache: Path) -> dict:
    receipt_path = cache / "receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"model cache missing receipt; run fetch-models --cache {cache}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("commit") != COMMIT or receipt.get("repository") != REPOSITORY:
        raise ValueError("model source revision mismatch")
    files = receipt.get("files_sha256", {})
    required = {f"{directory}/{name}" for directory, xml in MODELS.values() for name in (xml, "LICENSE")}
    if not required.issubset(files):
        raise ValueError("model cache is incomplete")
    for relative, expected in files.items():
        path = cache / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"model hash mismatch: {relative}")
    return receipt
