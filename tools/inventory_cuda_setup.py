"""Fetch a minimal, hash-verified portable CUDA 13.3.1 build tree.

No installer, registry, driver, PATH, or global environment changes. NVIDIA archive
licenses are retained. Reuses archives and outputs an installation manifest.
"""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DEPS=ROOT/"work/deps"
VERSION="13.3.1"
BASE="https://developer.download.nvidia.com/compute/cuda/redist/"
COMPONENTS=("cuda_nvcc","cuda_crt","cuda_cudart","libcublas","cccl","libnvvm")

def main():
    DEPS.mkdir(parents=True,exist_ok=True)
    manifest=DEPS/f"cuda-redist-{VERSION}.json"
    if not manifest.exists(): urllib.request.urlretrieve(BASE+f"redistrib_{VERSION}.json",manifest)
    data=json.loads(manifest.read_text())
    target=DEPS/"cuda-13.3"; target.mkdir(exist_ok=True)
    records=[]
    for name in COMPONENTS:
        info=data[name]["windows-x86_64"]; archive=DEPS/Path(info["relative_path"]).name
        if not archive.exists() or archive.stat().st_size != int(info["size"]):
            temporary=archive.with_suffix(".download")
            urllib.request.urlretrieve(BASE+info["relative_path"],temporary)
            temporary.replace(archive)
        with archive.open("rb") as source:
            actual=hashlib.file_digest(source,"sha256").hexdigest()
        if actual != info["sha256"]: raise ValueError(f"SHA256 mismatch: {name}")
        print(f"Verified {name}: {archive.stat().st_size:,} bytes",flush=True)
        # Merge component root directories directly, avoiding a second extracted copy.
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                parts=Path(entry.filename).parts[1:]
                if not parts: continue
                dst=target.joinpath(*parts)
                if not dst.resolve().is_relative_to(target.resolve()): raise ValueError("Unsafe archive path")
                if entry.is_dir(): dst.mkdir(parents=True,exist_ok=True)
                else:
                    dst.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(entry) as src, dst.open("wb") as output: shutil.copyfileobj(src,output)
        records.append({"component":name,"source":BASE+info["relative_path"],"sha256":actual,"download_bytes":archive.stat().st_size})
    (target/"portable-install.json").write_text(json.dumps({"cuda_release":VERSION,"components":records},indent=2))
    print(f"Portable CUDA ready: {target}")

if __name__=="__main__": main()
