"""Model- and GPU-free checks for the source preview."""

import io
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pack import ALIGNMENT, write_records
from preview import HERE, LOCK, verify
from run import MAX_PROMPT_BYTES, configuration, register, registered_inputs, validate_prompt


class PreviewTest(unittest.TestCase):
    def test_source_receipts(self):
        for name, sha in LOCK["preview_source"].items():
            verify(HERE / name, sha)
        for patch in LOCK["patches"]:
            verify(HERE / patch["name"], patch["sha256"])

    def test_prompt_read_is_bounded_before_decode(self):
        class DecodeForbidden(bytes):
            def decode(self, *args, **kwargs):
                raise AssertionError("oversized prompt was decoded")

        source = io.BytesIO()
        source.read = Mock(return_value=DecodeForbidden(b"x" * (MAX_PROMPT_BYTES + 1)))
        with patch.object(Path, "open", return_value=source):
            with self.assertRaisesRegex(ValueError, "at most 1 MiB"):
                validate_prompt(Path("prompt.txt"))
        source.read.assert_called_once_with(MAX_PROMPT_BYTES + 1)

    def test_prompt_exact_limit_and_invalid_content(self):
        for data in (b"", b" \n", b"\xff", b"x" * MAX_PROMPT_BYTES):
            with self.subTest(length=len(data)), patch.object(Path, "open", return_value=io.BytesIO(data)):
                if len(data) == MAX_PROMPT_BYTES:
                    validate_prompt(Path("prompt.txt"))
                else:
                    with self.assertRaisesRegex(ValueError, "nonempty UTF-8"):
                        validate_prompt(Path("prompt.txt"))

    def test_raw_record_order_and_padding(self):
        output = io.BytesIO()
        write_records(io.BytesIO(b"abcdWXYZ"), output, [[(0, 2), (4, 2)]], experts=2)
        records = output.getvalue()
        self.assertEqual(records, b"abWX" + bytes(ALIGNMENT - 4) + b"cdYZ" + bytes(ALIGNMENT - 4))

    def test_truncated_tile_fails(self):
        with self.assertRaisesRegex(ValueError, "Truncated checkpoint expert tile"):
            write_records(io.BytesIO(b"short"), io.BytesIO(), [[(0, 6)]], experts=1)

    def test_published_configuration(self):
        opts = SimpleNamespace(
            bin=Path("bin"), model=Path("model.gguf"), pack=Path("experts.pack"),
            prompt=Path("prompt.txt"), metrics=Path("metrics.json"), tokens=160,
        )
        cmd, env = configuration(opts)
        for flag, value in (("--context", "2048"), ("--cache-mib", "2048"), ("--prompt-batch", "176")):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertIn("--no-op-offload", cmd)
        self.assertEqual(env["BOVARY_STREAM_JOINT_DRAFT"], "48")
        self.assertEqual(env["BOVARY_STREAM_LOOKUP_DEPTH"], "16")
        self.assertEqual(env["BOVARY_STREAM_PACKED"], str(opts.pack))

    def test_registered_inputs_are_not_rehashed_and_stale_inputs_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            model, pack, receipt = (directory / name for name in ("model.gguf", "pack.bin", "receipt.json"))
            model.write_bytes(b"model")
            pack.write_bytes(b"pack")
            pinned = {"bytes": 5, "packed_bytes": 4,
                      "sha256": hashlib.sha256(b"model").hexdigest(),
                      "packed_sha256": hashlib.sha256(b"pack").hexdigest()}
            with patch.dict(LOCK["model"], pinned):
                register(model, pack, receipt)
                with patch("run.hashlib.file_digest", side_effect=AssertionError("unexpected rehash")):
                    self.assertEqual(registered_inputs(receipt), {"model": model, "pack": pack})
                old = model.stat()
                os.utime(model, ns=(old.st_atime_ns, old.st_mtime_ns - 2_000_000_000))
                with self.assertRaisesRegex(ValueError, "changed since registration"):
                    registered_inputs(receipt)
                os.utime(model, ns=(old.st_atime_ns, old.st_mtime_ns))
                new_receipt = directory / "new-receipt.json"
                register(model, pack, new_receipt)
                before_replacement = pack.stat()
                replacement = directory / "replacement"
                replacement.write_bytes(b"pack")
                os.utime(replacement, ns=(before_replacement.st_atime_ns, before_replacement.st_mtime_ns))
                replacement.replace(pack)
                with self.assertRaisesRegex(ValueError, "changed since registration"):
                    registered_inputs(new_receipt)


if __name__ == "__main__":
    unittest.main()
