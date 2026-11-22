"""Inspect GGUF descriptors and tokenizer metadata without mapping model weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
QUANTS = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
          6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34),
          10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
          13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
          20: ("IQ4_NL", 32, 18), 30: ("BF16", 1, 2)}
CHUNK = 1024 * 1024


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb", buffering=0) as source:
        while block := source.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def read_layout(path, architecture_prefixes=("gemma4.",)):
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb", buffering=CHUNK) as source:
        def read(length):
            if length < 0 or source.tell() + length > size:
                raise ValueError("GGUF field extends outside file")
            data = source.read(length)
            if len(data) != length:
                raise ValueError("truncated GGUF")
            return data

        def number(fmt):
            return struct.unpack("<" + fmt, read(struct.calcsize("<" + fmt)))[0]

        def skip(length):
            if length < 0 or source.tell() + length > size:
                raise ValueError("GGUF field extends outside file")
            source.seek(length, 1)

        def string(keep=True):
            length = number("Q")
            if keep:
                if length > 16 * CHUNK:
                    raise ValueError("oversized GGUF metadata string")
                return read(length).decode("utf-8")
            skip(length)

        def value(kind, keep):
            if kind == 8:
                return string(keep)
            if kind == 9:
                element, count = number("I"), number("Q")
                if count > size or element == 9:
                    raise ValueError("unsupported or oversized GGUF array")
                if keep and count <= 16384:
                    return [value(element, True) for _ in range(count)]
                if element == 8:
                    for _ in range(count):
                        string(False)
                elif element in FORMATS:
                    skip(struct.calcsize("<" + FORMATS[element]) * count)
                else:
                    raise ValueError("unknown GGUF array element type")
                return None
            if kind not in FORMATS:
                raise ValueError("unknown GGUF value type")
            result = number(FORMATS[kind])
            return result if keep else None

        if read(4) != b"GGUF":
            raise ValueError("not a little-endian GGUF file")
        version, tensor_count, kv_count = number("I"), number("Q"), number("Q")
        if version not in (2, 3) or tensor_count > 100000 or kv_count > 100000:
            raise ValueError("unsupported GGUF version or descriptor counts")
        metadata, tokenizer_hashes, seen = {}, {}, set()
        for _ in range(kv_count):
            key = string()
            if key in seen:
                raise ValueError("duplicate GGUF metadata key")
            seen.add(key)
            start = source.tell()
            kind = number("I")
            keep = key.startswith((*architecture_prefixes, "general.", "quantize."))
            item = value(kind, keep)
            end = source.tell()
            if keep:
                metadata[key] = item
            if key.startswith("tokenizer."):
                source.seek(start)
                digest = hashlib.sha256()
                remaining = end - start
                while remaining:
                    block = read(min(CHUNK, remaining))
                    digest.update(block)
                    remaining -= len(block)
                tokenizer_hashes[key] = digest.hexdigest()
        tensors, names = [], set()
        for _ in range(tensor_count):
            name, dimensions = string(), number("I")
            if name in names or dimensions < 1 or dimensions > 4:
                raise ValueError("duplicate tensor or unsupported dimensions")
            names.add(name)
            shape = [number("Q") for _ in range(dimensions)]
            kind, relative = number("I"), number("Q")
            if kind not in QUANTS or any(d < 1 for d in shape):
                raise ValueError("unsupported tensor type or shape")
            label, block, width = QUANTS[kind]
            if shape[0] % block:
                raise ValueError("tensor row incompatible with quantization block")
            nbytes = math.prod(shape) // block * width
            tensors.append({"name": name, "type": label, "type_id": kind, "shape": shape,
                            "relative_offset": relative, "nbytes": nbytes})
        alignment = metadata.get("general.alignment", 32)
        if not isinstance(alignment, int) or alignment < 1 or alignment > CHUNK or alignment & (alignment - 1):
            raise ValueError("invalid GGUF alignment")
        data_offset = (source.tell() + alignment - 1) // alignment * alignment
        for tensor in tensors:
            tensor["offset"] = data_offset + tensor["relative_offset"]
            if tensor["offset"] > size or tensor["nbytes"] > size - tensor["offset"]:
                raise ValueError("tensor payload extends outside file")
    return {"path": str(path.resolve()), "file_bytes": size, "version": version,
            "data_offset": data_offset, "metadata": metadata,
            "tokenizer_value_sha256": tokenizer_hashes, "tensors": tensors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = json.dumps(read_layout(args.model), indent=2)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
