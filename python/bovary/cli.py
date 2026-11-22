"""Bovary inference workbench. Repeated prompts and JSONL sessions reuse one worker."""
import argparse
import io
import json
import math
from pathlib import Path
import sys
import threading
import time

from .contract import ExecutionError, Request
from .runtime import create_workbench, project_home, load_validator
from . import __version__


def configure_output():
    """Use lossless UTF-8 for real text outputs, including redirected files.

    Python's Windows console writer handles UTF-8 as Unicode. Embedded streams
    retain their own write contract: StringIO and custom streams need neither
    a buffer nor reconfigure(). Never substitute characters in model output.
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding='utf-8', errors='strict')


def configure_input():
    if sys.stdin is sys.__stdin__ and isinstance(sys.stdin, io.TextIOWrapper) and not sys.stdin.isatty():
        sys.stdin.reconfigure(encoding='utf-8-sig', errors='strict')


def output_limit(value):
    if value.lower() == 'unlimited':
        return None
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('Use a positive token count or unlimited') from error
    if count < 1:
        raise argparse.ArgumentTypeError('Use a positive token count or unlimited')
    return count


def interactive_inputs():
    print('Local session. Each prompt starts fresh. Use /unload to release the model or /quit to exit.',
          file=sys.stderr, flush=True)
    while True:
        if sys.stdin.isatty():
            print('bovary> ', end='', file=sys.stderr, flush=True)
        line = sys.stdin.readline()
        if not line or line.strip() == '/quit':
            return
        if line.strip() == '/unload':
            yield {'unload': True}
        elif line.strip():
            yield {'prompt': line.rstrip('\r\n')}


def main(argv=None):
    configure_output()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'mini':
        from .mini import main as mini_main
        return mini_main(argv[1:])
    if argv and argv[0] == 'doctor':
        from .doctor import main as doctor_main
        return doctor_main(argv[1:])
    parser = argparse.ArgumentParser(prog='bovary', description=__doc__,
        epilog='Resource-bounded local tasks: bovary mini --help. Offline setup diagnostics: bovary doctor --help.')
    parser.add_argument('--version', action='version', version=f'Bovary {__version__}')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt', action='append', help='Repeat for independent warm requests')
    source.add_argument('--prompt-file', type=Path)
    source.add_argument('--interactive', action='store_true', help='Enter independent local prompts one line at a time with a warm model')
    source.add_argument('--session', action='store_true', help='Read JSON objects with prompt from stdin; {"unload": true} releases weights')
    parser.add_argument('--mode', choices=('local', 'commercial', 'suggest', 'auto'), default='local')
    parser.add_argument('--preset', choices=('cpu-only', 'balanced', 'gpu-fast', 'gpu-more-vram',
                                             'gpu-cache-2g', 'gpu-prefill-2g'), default='cpu-only')
    parser.add_argument('--task', choices=('general', 'extract', 'classify', 'format', 'summarize', 'translate', 'code', 'reason'), default='general')
    parser.add_argument('--workload', default='default')
    parser.add_argument('--context', type=int, default=4096)
    output = parser.add_mutually_exclusive_group()
    output.add_argument('--tokens', type=output_limit, default=128, help='Positive output cap or unlimited (local only; context and deadline still apply)')
    output.add_argument('--until-finished', dest='tokens', action='store_const', const=None, help='Local generation until model end, context limit, deadline, or cancellation')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--host-cache-mib', type=int, default=0,
                        help='Additional expert cache in host RAM for local streaming CUDA presets, in MiB (default 0)')
    parser.add_argument('--cpu-prompt-batch', type=int, metavar='1..128',
                        help='CPU input tokens processed together during prompt evaluation (default 32)')
    parser.add_argument('--deadline-s', type=float, default=300)
    parser.add_argument('--allow-cloud', action='store_true', help='Allow destination; live execution also requires approved provider config')
    parser.add_argument('--offline', action='store_true', help='Overrides cloud flags')
    parser.add_argument('--private', action='store_true', help='Forbid prompt upload')
    parser.add_argument('--data-class', default='', help='Explicitly classify permitted data for approved provider configuration')
    parser.add_argument('--max-cost-usd', type=float, default=0)
    parser.add_argument('--provider-config', type=Path)
    parser.add_argument('--validator', type=Path, help='Versioned task validator JSON')
    parser.add_argument('--allow-escalation', action='store_true')
    parser.add_argument('--home', type=Path, default=project_home())
    parser.add_argument('--json', action='store_true', help='Emit results as JSONL, stream events to stderr')
    args = parser.parse_args(argv)
    if args.interactive and (args.mode != 'local' or args.allow_cloud or args.allow_escalation):
        parser.error('--interactive requires local mode with cloud and escalation disabled')
    if not math.isfinite(args.deadline_s) or args.deadline_s <= 0:
        parser.error('deadline-s must be positive and finite')
    workbench = None
    cancel = threading.Event()
    exit_code = 0
    try:
        validator = load_validator(args.validator)
        workbench = create_workbench(args.home, args.provider_config)
        if args.session or args.interactive:
            # Redirected Windows stdin can use the active ANSI code page.
            # JSONL on the wire is UTF-8. Keep an embedded caller's text
            # stream and interactive console on their own decoding contract.
            configure_input()
            inputs = interactive_inputs() if args.interactive else (json.loads(line) for line in sys.stdin if line.strip())
        else:
            prompts = args.prompt if args.prompt else [args.prompt_file.read_text(encoding='utf-8')]
            inputs = ({'prompt': prompt} for prompt in prompts)
        for payload in inputs:
            if not isinstance(payload, dict):
                raise ExecutionError('invalid_request', 'Session lines must be JSON objects')
            if payload == {'unload': True}:
                workbench.unload()
                print('Model unloaded.' if args.interactive and not args.json else json.dumps({'status': 'unloaded'}), flush=True)
                continue
            defaults = dict(mode=args.mode, local_profile=args.preset, allow_gpu=args.preset != 'cpu-only',
                task=args.task, workload=args.workload, context_tokens=args.context, output_tokens=args.tokens,
                threads=args.threads, cpu_prompt_batch=args.cpu_prompt_batch, host_cache_mib=args.host_cache_mib,
                deadline=time.monotonic()+args.deadline_s,
                offline=args.offline or not args.allow_cloud, private=args.private or not args.allow_cloud,
                allowed_destinations=('local', 'commercial') if args.allow_cloud else ('local',),
                max_cost_usd=args.max_cost_usd, allow_escalation=args.allow_escalation,
                validator=validator, identity={'data_class': args.data_class})
            # Session requests share the same contract, but cannot grant cloud/GPU permissions absent CLI flags.
            request = Request(**{**defaults, **payload})
            if not args.allow_cloud:
                request.offline = request.private = True
                request.allowed_destinations = ('local',)
            if args.offline:
                request.offline = True
            if args.private:
                request.private = True
            if args.preset == 'cpu-only':
                request.allow_gpu = False
            request.max_cost_usd = min(request.max_cost_usd, args.max_cost_usd)
            request.deadline = min(request.deadline, defaults['deadline']) if request.deadline is not None else defaults['deadline']
            json_output = args.json or args.session
            output_errors = []
            def event(item):
                if output_errors:
                    return
                try:
                    if item.kind == 'text':
                        print(item.text, end='', file=sys.stderr if json_output else sys.stdout, flush=True)
                    elif item.kind == 'attempt' and not json_output and not args.interactive:
                        print(f"\n[Attempt {item.data.get('number')}: {item.data.get('destination')}]", flush=True)
                except (OSError, UnicodeError) as error:
                    # Let the adapter cancel and reap its worker, then preserve
                    # the output failure instead of relabelling it as inference.
                    output_errors.append(error)
                    cancel.set()
            try:
                result = workbench.execute(request, on_event=event, cancel=cancel)
            except Exception:
                if output_errors:
                    raise output_errors[0]
                raise
            if output_errors:
                raise output_errors[0]
            serialized = result.to_dict()
            if json_output:
                # JSONL remains lossless even when Windows redirects stdout
                # through a legacy code page. Consumers decode Unicode escapes.
                print(json.dumps(serialized, ensure_ascii=True, allow_nan=False), flush=True)
            elif args.interactive:
                print(flush=True)
                if result.status != 'complete':
                    error = result.error or {}
                    detail = error.get('message') or result.validation.get('reason') or result.status
                    if error.get('code') == 'context_overflow':
                        detail = 'Prompt and output allowance exceed the context window. Try a shorter prompt.'
                    elif result.status == 'truncated':
                        detail = result.usage.get('finish_reason') or detail
                    print(f'[{result.status}] {detail}', file=sys.stderr, flush=True)
            else:
                if result.status == 'suggested':
                    print(json.dumps(serialized, ensure_ascii=False, indent=2))
                else:
                    print()
                    if len(result.attempts) > 1:
                        print('[Final selected answer]')
                        print(result.text)
                    print(json.dumps({k: v for k, v in serialized.items() if k != 'text'}, ensure_ascii=False), file=sys.stderr)
            if result.status not in ('complete', 'suggested'):
                exit_code = 2
    except KeyboardInterrupt:
        cancel.set()
        print('Cancelled; unloading worker.', file=sys.stderr)
        exit_code = 130
    except BrokenPipeError:
        raise
    except (OSError, ValueError, TypeError, ExecutionError) as error:
        print(json.dumps({'status': 'failed', 'error': {'code': getattr(error, 'code', 'configuration'), 'message': str(error)}}), file=sys.stderr)
        exit_code = 2
    finally:
        if workbench is not None:
            workbench.unload()
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
