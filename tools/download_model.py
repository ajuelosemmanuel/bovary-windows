"""One-copy, resumable, pinned checkpoint download (standard library only)."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import time
import urllib.request

if __package__:
    from .verify_model import REPO, REV, NAME, SIZE, existing_manifest, expected_manifest, path_identity, publish_no_replace, verify_file, write_manifest
else:
    from verify_model import REPO, REV, NAME, SIZE, existing_manifest, expected_manifest, path_identity, publish_no_replace, verify_file, write_manifest


class DownloadProtocolError(RuntimeError):
    """A response that cannot safely extend the pinned partial file."""


def validate_response(response, have):
    """Check declared byte identity and size before opening the partial file."""
    if have and response.status != 206:
        raise DownloadProtocolError("Server ignored resume Range; preserved partial file")
    if response.status == 206:
        value = response.headers.get("Content-Range", "")
        matched = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)", value, re.IGNORECASE)
        if (matched is None or any(len(group) > 20 for group in matched.groups())
                or tuple(map(int, matched.groups())) != (have, SIZE - 1, SIZE)):
            raise DownloadProtocolError("Server Content-Range differs from the pinned remaining bytes; preserved partial file")
    elif response.status != 200:
        raise DownloadProtocolError("Unexpected download response status; preserved partial file")
    length = response.headers.get("Content-Length")
    if length is not None and (len(length) > 20 or re.fullmatch(r"[0-9]+", length) is None or int(length) != SIZE - have):
        raise DownloadProtocolError("Server Content-Length differs from the pinned remaining bytes; preserved partial file")
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise DownloadProtocolError("Encoded response cannot extend pinned model bytes; preserved partial file")


class DownloadLock:
    """Exclusive process lock; its stable file is deliberately never removed."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory/".bovary-download.lock"
        self.file = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise RuntimeError("Download lock must not be a symbolic link")
        stream = self.path.open("a+b")
        try:
            if os.fstat(stream.fileno()).st_nlink > 1:
                raise RuntimeError("Download lock must not be a shared hard link")
            if os.name == "nt":
                import msvcrt
                stream.seek(0, os.SEEK_END)
                if stream.tell() == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException as error:
            stream.close()
            if isinstance(error, OSError):
                raise RuntimeError("Another downloader is active for this model directory; wait for it to finish") from error
            raise
        self.file = stream
        return self

    def __exit__(self, *args):
        stream, self.file = self.file, None
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--directory", default="work/models")
    p.add_argument("--manifest", type=Path, help="New proof path; defaults to model-provenance.json in the download directory")
    a = p.parse_args(argv)
    directory = Path(a.directory)
    with DownloadLock(directory):
        return _download_locked(directory, a.manifest)


def _download_locked(directory, manifest=None):
    """Read/write model state only while main owns this directory's lock."""
    dest = directory / NAME
    partial = dest.with_suffix(".gguf.partial")
    manifest = manifest if manifest is not None else directory / "model-provenance.json"
    existing_manifest(manifest, expected_manifest(NAME))
    if not dest.exists():
        if partial.is_symlink() or (partial.exists() and partial.stat().st_nlink > 1):
            raise RuntimeError("Partial file is a link; preserved without downloading into it")
        have = partial.stat().st_size if partial.exists() else 0
        if have > SIZE:
            raise RuntimeError("Partial file exceeds pinned size; preserved for review")
        if shutil.disk_usage(directory).free < SIZE - have + 3 * 1024**3:
            raise RuntimeError("Insufficient disk space including 3 GiB reserve")
        for attempt in range(12):
            have = partial.stat().st_size if partial.exists() else 0
            if have == SIZE:
                break
            request = urllib.request.Request(
                f"https://huggingface.co/{REPO}/resolve/{REV}/{NAME}",
                headers={"Range": f"bytes={have}-", "User-Agent": "bovary-prototype/1"})
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    validate_response(response, have)
                    with partial.open("ab" if have else "wb") as output:
                        last = time.monotonic()
                        while block := response.read(min(8 * 1024**2, SIZE - have + 1)):
                            if len(block) > SIZE - have:
                                raise DownloadProtocolError("Server sent more than pinned size; excess bytes were not written")
                            output.write(block)
                            have += len(block)
                            if time.monotonic() - last > 20:
                                print(json.dumps({"downloaded": have, "total": SIZE}), flush=True)
                                last = time.monotonic()
                if have != SIZE:
                    raise RuntimeError(f"Incomplete: {have}/{SIZE}")
            except DownloadProtocolError:
                raise  # Retrying a mismatched response must not poison resumable bytes.
            except Exception as e:
                print(json.dumps({"attempt": attempt, "error": str(e)}), flush=True)
                if attempt == 11:
                    raise
                time.sleep(min(2**attempt, 30))
        candidate = partial
    else:
        candidate = dest
    metadata = verify_file(candidate)
    metadata["filename"] = NAME
    if candidate == partial:
        try:
            publish_no_replace(partial, dest)
        except FileExistsError:
            metadata = verify_file(dest)
            partial.unlink(missing_ok=True)
        else:
            partial.unlink(missing_ok=True)
            if path_identity(dest) != metadata["file_identity"]:
                # Publishing and unlinking can change the final file identity.
                metadata = verify_file(dest)
    metadata = write_manifest(manifest, metadata)
    print(json.dumps(metadata), flush=True)

if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Download failed: {error}")
