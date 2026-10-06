"""Command line: ``python -m physics_replay {fetch-models,demo,replay,benchmark}``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import runner
from .models import fetch_models, verify_models
from .retarget import METHODS as RETARGET_METHODS

DEFAULT_CONFIG = runner.REPO_ROOT / "config" / "physics-replay" / "pick-place-v1.json"
DEFAULT_CACHE = runner.REPO_ROOT / "data" / "physics-replay" / "models"


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m physics_replay", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("fetch-models", "download the pinned Menagerie Panda, UR5e and Robotiq 2F-85 once"),
                       ("demo", "source demo + three target runs in one new scene, with videos and plots"),
                       ("replay", "retarget a schema-valid episode JSON onto the UR5e in the demo target scene"),
                       ("benchmark", "seeded paired benchmark over sampled target layouts")):
        command = commands.add_parser(name, help=text, description=text)
        command.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="model cache directory")
        if name == "fetch-models":
            continue
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--out", type=Path, help="run directory (default: data/physics-replay/runs/<kind>-<time>)")
        if name in ("demo", "replay"):
            command.add_argument("--no-video", action="store_true", help="skip rendering")
        if name == "replay":
            command.add_argument("--episode", type=Path, required=True)
            command.add_argument("--method", choices=RETARGET_METHODS, default="object_relative")
        if name == "benchmark":
            command.add_argument("--configs", type=int, help="number of sampled layouts (default from config)")
            command.add_argument("--seed", type=int, help="layout sampling seed (default from config)")
            command.add_argument("--methods", nargs="+", choices=runner.METHODS, default=list(runner.METHODS))
    args = parser.parse_args()

    if args.command == "fetch-models":
        if (args.cache / "receipt.json").exists():
            print(f"model cache already complete, receipt sha256 {verify_models(args.cache)}")
            return
        receipt = fetch_models(args.cache)
        print(f"fetched {len(receipt['files'])} files, receipt sha256 {verify_models(args.cache)}")
        return
    out = args.out or runner.default_out(args.command)
    if args.command == "demo":
        manifest = runner.demo(args.config, args.cache, out, not args.no_video, runner.argv())
        result = {name: {k: m[k] for k in ("success", "failure_reason", "final_xy_error_m")}
                  for name, m in manifest["metrics"].items()}
    elif args.command == "replay":
        manifest = runner.replay(args.episode, args.config, args.cache, out, args.method, not args.no_video,
                                 runner.argv())
        result = {k: manifest["metrics"][k] for k in ("success", "failure_reason", "final_xy_error_m")}
    else:
        manifest = runner.benchmark(args.config, args.cache, out, args.configs, args.seed, args.methods,
                                    runner.argv())
        print((out / "summary.md").read_text())
        result = None
    if result is not None:
        print(json.dumps(result, indent=1))
    print(f"outputs: {out}")


if __name__ == "__main__":
    main()
