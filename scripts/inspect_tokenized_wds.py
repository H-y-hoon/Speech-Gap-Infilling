from __future__ import annotations

import argparse
import io
import json
import tarfile

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a VoiceCraft-X tokenized WebDataset shard.")
    parser.add_argument("shard")
    parser.add_argument("--max-samples", type=int, default=3)
    args = parser.parse_args()

    grouped = {}
    with tarfile.open(args.shard, mode="r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            key, ext = member.name.split(".", 1)
            grouped.setdefault(key, {})[ext] = tar.extractfile(member).read()
            if len(grouped) >= args.max_samples and all("json" in value for value in grouped.values()):
                break

    for key, files in list(grouped.items())[: args.max_samples]:
        meta = json.loads(files["json"].decode("utf-8"))
        tokens = torch.load(io.BytesIO(files["tokens.pt"]), map_location="cpu")
        print(key, tuple(tokens.shape), meta["duration_sec"], meta["speaker_id"])


if __name__ == "__main__":
    main()
