"""Construct matched serial/pinned experiments from current core templates."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("smoke", "matrix"), default="smoke")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--build-manifest", type=Path, required=True)
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="bovary-template-") as temporary:
        template = Path(temporary)/"template.json"
        subprocess.run([sys.executable, str(ROOT/"tools/make_config.py"), "--mode", args.mode,
                        "--output", str(template), "--build-manifest", str(args.build_manifest)], check=True)
        config = json.loads(template.read_text(encoding="utf-8"))
    configs = []
    if args.mode == "smoke":
        original = next(x for x in config["configurations"] if x["id"] == "E1-direct512-gpuembed")
        for mode in ("serial", "pinned2"):
            c = copy.deepcopy(original)
            c["id"] = "v2-" + mode + "512"
            c["env"]["BOVARY_STREAM_POOL"] = mode
            configs.append(c)
    else:
        for mode, budgets in (("serial", (512, 1536)), ("pinned2", (32, 512, 1024, 1536))):
            for budget in budgets:
                source_budget = budget if budget != 1024 else 512
                c = copy.deepcopy(next(x for x in config["configurations"] if x["id"] == f"E2-direct{source_budget}"))
                c["command"][c["command"].index("--cache-mib") + 1] = str(budget)
                c["id"] = f"v2-{mode}{budget}"
                c["env"]["BOVARY_STREAM_POOL"] = mode
                configs.append(c)
    config["configurations"] = configs
    args.output.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(args.output)

if __name__ == "__main__":
    main()
