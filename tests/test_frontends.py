"""Frontend execution gates, without model/network/desktop interaction."""
import contextlib
import io
import json
from pathlib import Path
import sys
import time
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from bovary import cli
from bovary.contract import Event, ExecutionError, Request, Result
from bovary.routing import Workbench


class Adapter:
    def __init__(self):
        self.requests = []
        self.unloads = 0

    def execute(self, request, on_event=None, cancel=None):
        self.requests.append(request)
        if on_event:
            on_event(Event('text', request.request_id, text='ok'))
        return Result(request.request_id, request.mode, text='ok')

    def unload(self):
        self.unloads += 1


class FrontendTests(unittest.TestCase):
    def run_cli(self, args, workbench, stdin=''):
        output, error = io.StringIO(), io.StringIO()
        with mock.patch.object(cli, 'create_workbench', return_value=workbench), mock.patch.object(sys, 'stdin', io.StringIO(stdin)), contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = cli.main(args)
        return code, output.getvalue(), error.getvalue()

    def test_repeated_prompts_share_adapter_and_unload(self):
        local = Adapter()
        code, output, _ = self.run_cli(['--prompt', 'a', '--prompt', 'b', '--json'], Workbench(local=local))
        self.assertEqual(code, 0)
        self.assertEqual([r.prompt for r in local.requests], ['a', 'b'])
        self.assertTrue(all(r.local_profile == 'cpu-only' and not r.allow_gpu and r.offline and r.private for r in local.requests))
        self.assertEqual(len([json.loads(x) for x in output.splitlines()]), 2)
        self.assertEqual(local.unloads, 1)

    def test_output_limit_options_and_session_null_reach_the_shared_contract(self):
        for args in ([], ['--tokens', '7'], ['--tokens', 'unlimited'], ['--until-finished']):
            with self.subTest(args=args):
                local = Adapter()
                code, _, _ = self.run_cli(['--prompt', 'answer', '--json', *args], Workbench(local=local))
                self.assertEqual(code, 0)
                self.assertEqual(local.requests[0].output_tokens, 128 if not args else 7 if args == ['--tokens', '7'] else None)
                self.assertIsNotNone(local.requests[0].deadline)
        local = Adapter()
        code, _, _ = self.run_cli(['--session'], Workbench(local=local), '{"prompt":"answer","output_tokens":null}\n')
        self.assertEqual(code, 0)
        self.assertIsNone(local.requests[0].output_tokens)

    def test_gpu_prefill_cli_explicitly_grants_local_gpu_permission(self):
        local = Adapter()
        code, _, _ = self.run_cli(['--prompt', 'answer', '--preset', 'gpu-prefill-2g'], Workbench(local=local))
        self.assertEqual(code, 0)
        request, = local.requests
        self.assertEqual(request.local_profile, 'gpu-prefill-2g')
        self.assertTrue(request.allow_gpu and request.offline and request.private)

    def test_host_cache_reaches_single_and_session_requests(self):
        for source in (['--prompt', 'answer'], ['--session']):
            with self.subTest(source=source):
                local = Adapter()
                code, _, _ = self.run_cli([*source, '--preset', 'gpu-prefill-2g', '--host-cache-mib', '4096'],
                                          Workbench(local=local), '{"prompt":"answer"}\n')
                self.assertEqual(code, 0)
                request, = local.requests
                self.assertEqual(request.host_cache_mib, 4096)
                self.assertTrue(request.allow_gpu)

    def test_until_finished_cannot_reach_commercial_or_escalation(self):
        for args in (['--mode', 'commercial', '--allow-cloud'], ['--mode', 'suggest'], ['--allow-escalation']):
            with self.subTest(args=args):
                local, cloud = Adapter(), Adapter()
                code, output, _ = self.run_cli(['--prompt', 'answer', '--until-finished', '--json', *args], Workbench(local=local, commercial=cloud))
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output)['error']['code'], 'invalid_request')
                self.assertEqual(local.requests + cloud.requests, [])

    def test_until_finished_requires_deadline_and_numeric_limits_remain_strict(self):
        with self.assertRaises(ExecutionError) as caught:
            Request('answer', output_tokens=None).check()
        self.assertEqual(caught.exception.code, 'deadline_required')
        Request('answer', output_tokens=None, deadline=time.monotonic()+10).check()
        for value in (0, -1, True, 'unlimited', 4096):
            with self.subTest(value=value), self.assertRaises(ExecutionError):
                Request('answer', output_tokens=value).check()

    def test_cli_rejects_invalid_or_conflicting_limits(self):
        for args in (['--tokens', '0'], ['--tokens', '-1'], ['--tokens', 'forever'], ['--tokens', '2', '--until-finished']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                cli.main(['--prompt', 'answer', *args])
            self.assertEqual(caught.exception.code, 2)

    def test_json_export_preserves_unicode_on_legacy_redirected_stdout(self):
        answer = '\u05e9\u05dc\u05d5\u05dd caf\u00e9 \U0001f9ea'
        class UnicodeAdapter(Adapter):
            def execute(self, request, on_event=None, cancel=None):
                self.requests.append(request)
                if on_event:
                    on_event(Event('text', request.request_id, text=answer))
                return Result(request.request_id, 'local', text=answer, model='\u6a21\u578b')
        data = io.BytesIO()
        output = io.TextIOWrapper(data, encoding='cp1252')
        with mock.patch.object(cli, 'create_workbench', return_value=Workbench(local=UnicodeAdapter())), \
                mock.patch.object(sys, 'stdout', output), contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(['--prompt', 'greet', '--json'])
            output.flush()
        self.assertEqual(code, 0)
        result = json.loads(data.getvalue().decode('ascii'))
        self.assertEqual(result['text'], answer)
        self.assertEqual(result['model'], '\u6a21\u578b')

    def test_suggest_and_auto_never_execute_adapters(self):
        local, cloud = Adapter(), Adapter()
        wb = Workbench(local=local, commercial=cloud)
        code, output, _ = self.run_cli(['--prompt', 'extract name', '--task', 'extract', '--mode', 'suggest', '--json'], wb)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)['status'], 'suggested')
        code, output, _ = self.run_cli(['--prompt', 'extract name', '--task', 'extract', '--mode', 'auto', '--json'], wb)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output)['error']['code'], 'auto_disabled')
        self.assertEqual(local.requests + cloud.requests, [])

    def test_session_cannot_grant_cloud_gpu_budget_or_longer_deadline(self):
        local, cloud = Adapter(), Adapter()
        payloads = [dict(prompt='ok', deadline=time.monotonic()+99999, max_cost_usd=100),
                    dict(prompt='blocked cloud', mode='commercial', offline=False, private=False, allowed_destinations=['commercial']),
                    dict(prompt='blocked gpu', local_profile='gpu-fast', allow_gpu=True)]
        code, output, _ = self.run_cli(['--session', '--deadline-s', '2'], Workbench(local=local, commercial=cloud), '\n'.join(json.dumps(p) for p in payloads))
        self.assertEqual(code, 2)
        self.assertEqual(len(local.requests), 1)
        self.assertEqual(cloud.requests, [])
        self.assertEqual(local.requests[0].max_cost_usd, 0)
        self.assertLessEqual(local.requests[0].deadline, time.monotonic()+2)
        results = [json.loads(x) for x in output.splitlines()]
        self.assertEqual(results[1]['error']['code'], 'destination_forbidden')
        self.assertEqual(results[2]['error']['code'], 'gpu_forbidden')

    def test_invalid_contract_values_are_typed_errors(self):
        for change in ({'context_tokens': '4096'}, {'private': 'false'}, {'offline': 0}, {'allowed_destinations': 'local'}, {'output_tokens': True}, {'max_cost_usd': float('nan')}, {'deadline': 'later'}, {'validator': []}):
            with self.subTest(change=change), self.assertRaises(ExecutionError):
                Request('hello', **change).check()

    def test_invalid_session_bounds_are_rejected_without_losing_later_requests(self):
        local = Adapter()
        invalid_bounds = [
            {'deadline': True}, {'deadline': float('inf')}, {'deadline': 'later'},
            {'max_cost_usd': True}, {'max_cost_usd': float('inf')}, {'max_cost_usd': '0'},
        ]
        lines = [json.dumps(dict(prompt='invalid', **bounds)) for bounds in invalid_bounds]
        lines.append(json.dumps({'prompt': 'valid'}))
        code, output, _ = self.run_cli(['--session'], Workbench(local=local), '\n'.join(lines))
        results = [json.loads(line) for line in output.splitlines()]
        self.assertEqual(code, 2)
        self.assertEqual([result['error']['code'] for result in results[:-1]],
                         ['invalid_request'] * len(invalid_bounds))
        self.assertEqual(results[-1]['status'], 'complete')
        self.assertEqual([request.prompt for request in local.requests], ['valid'])

    def test_invalid_unicode_and_huge_numbers_fail_before_any_backend(self):
        local, cloud = Adapter(), Adapter()
        workbench = Workbench(local=local, commercial=cloud)
        for mode in ('local', 'suggest'):
            for change in ({'prompt': chr(0xd800)}, {'prompt': chr(0xdfff)},
                           {'max_cost_usd': 10**1000}, {'deadline': 10**1000}):
                with self.subTest(mode=mode, fields=list(change)):
                    result = workbench.execute(Request(**{'prompt':'hello', 'mode':mode, **change}))
                    self.assertEqual(result.status, 'failed')
                    self.assertEqual(result.error['code'], 'invalid_request')
                    self.assertEqual(result.attempts, [])
        self.assertEqual(local.requests + cloud.requests, [])
        Request('\u05e9\u05dc\u05d5\u05dd caf\u00e9 \U0001f9ea').check()

    def test_session_rejects_escaped_surrogate_and_continues_to_next_request(self):
        local, cloud = Adapter(), Adapter()
        code, output, _ = self.run_cli(['--session', '--mode', 'suggest'],
            Workbench(local=local, commercial=cloud), '{"prompt":"\\ud800"}\n{"prompt":"valid"}\n')
        self.assertEqual(code, 2)
        results = [json.loads(line) for line in output.splitlines()]
        self.assertEqual(results[0]['error']['code'], 'invalid_request')
        self.assertEqual(results[0]['attempts'], [])
        self.assertEqual(results[1]['status'], 'suggested')
        self.assertEqual(local.requests + cloud.requests, [])

    def test_existing_home_policy_is_shared_by_runtime_and_legacy_planner(self):
        from bovary.runtime import create_workbench
        from bovary.policy import default_policy_path
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundled = default_policy_path(root)
            policy = json.loads(bundled.read_text())
            policy['minimum_cases'] = 777
            editable = root/'benchmarks/decision_policy.json'
            editable.parent.mkdir()
            editable.write_text(json.dumps(policy), encoding='utf-8')
            self.assertEqual(default_policy_path(root), editable)
            self.assertEqual(create_workbench(root=root).policy['minimum_cases'], 777)

    def test_gui_unload_keeps_commercial_budget_ledger(self):
        from bovary.gui import SettingsWindow
        local, commercial = Adapter(), SimpleNamespace(committed_usd=1.5)
        workbench = Workbench(local=local, commercial=commercial)
        window = SettingsWindow.__new__(SettingsWindow)
        window.process = None
        window.workbench = workbench
        window.status = SimpleNamespace(set=lambda value: None)
        window.unload()
        self.assertIs(window.workbench, workbench)
        self.assertIs(window.workbench.commercial, commercial)
        self.assertEqual(window.workbench.commercial.committed_usd, 1.5)
        self.assertEqual(local.unloads, 1)

    def test_provider_ledger_identity_survives_config_path_aliases_and_copies(self):
        from bovary.runtime import load_provider_config, provider_identity
        from bovary.gui import SettingsWindow
        with tempfile.TemporaryDirectory() as directory:
            first, copy = Path(directory)/'first.json', Path(directory)/'copy.json'
            first.write_text('{"enabled":false,"session_cap_usd":0}', encoding='utf-8')
            copy.write_text('{ "session_cap_usd": 0, "enabled": false }', encoding='utf-8')
            snapshot = load_provider_config(first)
            self.assertEqual(provider_identity(snapshot), provider_identity(load_provider_config(copy)))
            self.assertEqual(provider_identity(snapshot),
                             provider_identity(load_provider_config(first.parent/'.'/first.name)))
            window = SettingsWindow.__new__(SettingsWindow)
            window.process, window.workbench = None, None
            window.workbench_cache, window.approved_provider_key = {}, None
            window.status = SimpleNamespace(set=lambda value: None)
            approved = Workbench(local=Adapter(), commercial=SimpleNamespace(committed_usd=1.5))
            with mock.patch('bovary.gui.create_workbench', return_value=approved) as factory:
                window.select_workbench(first)
                window.unload()
                window.select_workbench(copy)
                self.assertIs(window.workbench, approved)
                factory.assert_called_once_with(mock.ANY, snapshot)
                self.assertEqual(window.workbench.commercial.committed_usd, 1.5)
                copy.write_text('{"enabled":true,"session_cap_usd":99}', encoding='utf-8')
                with self.assertRaises(ExecutionError) as caught:
                    window.select_workbench(copy)
                self.assertEqual(caught.exception.code, 'provider_session_locked')


if __name__ == '__main__':
    unittest.main()
