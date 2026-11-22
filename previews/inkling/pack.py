"""Make the pinned Inkling raw expert sidecar from a caller-owned checkpoint."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

from preview import HERE, LOCK, verify

sys.path.insert(0, str(HERE.parents[1] / "tools"))
from gguf_layout import read_layout

ALIGNMENT = 4096
LAYERS = 24
EXPERTS = 128


def expert_plan(layout):
    metadata = layout["metadata"]
    if metadata["general.architecture"] != "inkling2_loop":
        raise ValueError("Expected an Inkling 2 loop checkpoint")
    tensors = {tensor["name"]: tensor for tensor in layout["tensors"]}
    rows = []
    for layer in range(LAYERS):
        offsets = []
        for kind in ("gate_up", "down"):
            tensor = tensors[f"blk.{layer}.ffn_{kind}_exps.weight"]
            if len(tensor["shape"]) != 3 or tensor["shape"][-1] != EXPERTS:
                raise ValueError(f"Invalid expert tensor: {tensor['name']}")
            tile, tail = divmod(tensor["nbytes"], EXPERTS)
            if not tile or tail:
                raise ValueError(f"Invalid expert tile length: {tensor['name']}")
            offsets.append((tensor["offset"], tile))
        rows.append(offsets)
    return rows


def write_records(source, output, rows, experts=EXPERTS):
    for row in rows:
        payload = sum(tile for _, tile in row)
        padding = bytes(-payload % ALIGNMENT)
        for expert in range(experts):
            for start, tile in row:
                source.seek(start + expert * tile)
                data = source.read(tile)
                if len(data) != tile:
                    raise ValueError("Truncated checkpoint expert tile")
                output.write(data)
            output.write(padding)


def create(model, destination):
    identity = LOCK["model"]
    if destination.exists():
        raise FileExistsError(destination)
    if model.stat().st_size != identity["bytes"]:
        raise ValueError("Checkpoint size differs from the pinned file")
    verify(model, identity["sha256"])
    rows = expert_plan(read_layout(model, architecture_prefixes=("inkling2_loop.",)))
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".",
                                     suffix=".tmp", delete=False) as output:
        staged = Path(output.name)
    try:
        with model.open("rb") as source, staged.open("wb") as output:
            write_records(source, output, rows)
        if staged.stat().st_size != identity["packed_bytes"]:
            raise ValueError("Packed sidecar length differs from the pinned layout")
        verify(staged, identity["packed_sha256"])
        if os.name == "nt":
            staged.rename(destination)
        else:
            os.link(staged, destination)
    finally:
        staged.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    create(args.model, args.output)
    print(f"Verified packed sidecar: {args.output}")


if __name__ == "__main__":
    main()
