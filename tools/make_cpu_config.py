"""Generate a CPU-only measurement configuration that never queries a GPU."""
import argparse
import json
from build_provenance import load_build, verified_model
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'work/cpu-only.json')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument("--build-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    executable, build_metadata = load_build(args.build_manifest, 'cpu', ROOT)
    model, model_metadata = verified_model(ROOT)
    config = {
        'configurations': [{
            'id': 'CPU-only-Q4-batch32',
            'command': [str(executable), '--require-cpu-only', '--model', str(model),
                '--prompt-file', '{prompt_file}', '--metrics', '{metrics_file}',
                '--logits', '{logits_file}', '--predict', '{n_predict}', '--context', '{ctx}',
                '--gpu-layers', '0', '--cache-mib', '0', '--load-mode', 'mmap',
                '--threads', str(args.threads), '--prompt-batch', '32'],
            'env': {'BOVARY_STREAM': '0', 'GGML_BACKEND_PATH': ''},
            'gpu_telemetry': False,
            'stdout_is_generation': True,
            'metadata': {**build_metadata, **model_metadata, 'threads': args.threads,
                'prompt_batch': 32, 'ubatch': 32,
                'device_policy': 'CPU static backend only; no GPU enumeration, libraries, telemetry or PDH GPU counters',
                'full_logits_dump': 'enabled equally across repetitions',
                'power_policy': 'unchanged; no GPU-heavy applications need to close for GPU contention'}
        }],
        'workloads': ['short'], 'repetitions': 3, 'warmups': 0,
        'context': 4096, 'n_predict': 32, 'timeout_seconds': 1800,
        'sample_interval_seconds': .2, 'pdh': False, 'gpu_telemetry': False,
        'monitor_disk_path': str(ROOT)
    }
    args.output.write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
