#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys

from task2_mac.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task2_mac.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    for style_id in cfg["styles"].keys():
        cmd = [
            sys.executable,
            "scripts/train_style_lora.py",
            "--config",
            args.config,
            "--style-id",
            str(style_id),
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
