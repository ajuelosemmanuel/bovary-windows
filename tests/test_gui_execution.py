"""Exercise GUI request construction through code, without desktop input."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from bovary.contract import Event, ExecutionError, Request, Result, STREAMING_GPU_PROFILES
from bovary.gui import DEFAULTS, SettingsWindow, build_request, save_settings
from bovary.routing import Workbench


class RecordingAdapter:
    def __init__(self, text):
        self.text, self.requests = text, []

    def execute(self, request, on_event=None, cancel=None):
        self.requests.append(request)
        if on_event:
            on_event(Event('text', request.request_id, text=self.text))
        return Result(request.request_id, request.mode, text=self.text)

    def unload(self):
        pass


class GuiExecutionTests(unittest.TestCase):
    def window(self, settings_path, prompt='Answer'):
        """Use the real controller with in-memory widgets, never a Tk root."""
        class Text:
            def __init__(self, text=''):
                self.text = text
            def get(self, *args):
                return self.text
            def configure(self, **kwargs):
                pass
            def delete(self, *args):
                self.text = ''
            def insert(self, position, text):
                self.text += text
            def see(self, *args):
                pass
        class Variable:
            def __init__(self, value):
                self.value = value
            def get(self):
                return self.value
            def set(self, value):
                self.value = value
        import queue
        window = SettingsWindow.__new__(SettingsWindow)
        window.root = SimpleNamespace(after=lambda *args: None, destroy=mock.Mock(),
            clipboard_clear=mock.Mock(), clipboard_append=mock.Mock())
        window.process, window.workbench, window.approved_provider_key = None, None, None
        window.workbench_cache = {}
        window.events, window.cancel = queue.Queue(), threading.Event()
        window.closing = False
        window.settings_path = settings_path
        window.settings = lambda: DEFAULTS.copy()
        window.prompt, window.output, window.details = Text(prompt), Text(), Text()
        values = dict(route='Local', task='classify', offline=True, private=True,
            device='CPU', profile='CPU only', host_cache_mib='0',
            threads='8', context='4096', tokens='128', description='',
            provider_config='', data_class='', budget='0', timeout='30',
            validator_path='', allow_escalation=False, until_finished=False, status='Ready')
        for name, value in values.items():
            setattr(window, name, Variable(value))
        for name in ('device_box', 'profile_box', 'context_box', 'route_box', 'task_box',
                     'thread_box', 'token_box', 'host_cache_box', 'until_finished_box',
                     'save_button', 'run_button', 'stop_button', 'unload_button'):
            setattr(window, name, mock.Mock())
        window.request_controls = (mock.Mock(), mock.Mock())
        return window

    def pump_until_done(self, window):
        deadline = time.monotonic()+3
        while window.process is not None and time.monotonic() < deadline:
            window.pump()
            time.sleep(.005)
        self.assertIsNone(window.process)

    def test_validator_and_escalation_are_explicit_request_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'validator.json'
            spec = {'kind': 'exact', 'expected': 'yes'}
            path.write_text(json.dumps(spec), encoding='utf-8')
            request = build_request(DEFAULTS, 'Answer', task='classify', validator_path=path,
                allow_escalation=True, private=False, offline=False, budget='1', deadline_s='30')
            path.write_text('{"kind":"exact","expected":"changed"}', encoding='utf-8')
            self.assertEqual(request.validator, spec)
            self.assertTrue(request.allow_escalation)
            self.assertFalse(request.allow_gpu)
            local, cloud = RecordingAdapter('wrong'), RecordingAdapter('yes')
            result = Workbench(local=local, commercial=cloud).execute(request)
            self.assertEqual(result.text, 'yes')
            self.assertEqual(result.status, 'complete')
            self.assertEqual(len(result.attempts), 2)
            self.assertEqual((len(local.requests), len(cloud.requests)), (1, 1))

    def test_private_offline_still_override_escalation_checkbox(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'validator.json'
            path.write_text('{"kind":"exact","expected":"yes"}', encoding='utf-8')
            for private, offline in ((True, False), (False, True), (True, True)):
                request = build_request(DEFAULTS, 'Answer', task='classify', validator_path=path,
                    allow_escalation=True, private=private, offline=offline, budget='1')
                local, cloud = RecordingAdapter('wrong'), RecordingAdapter('yes')
                result = Workbench(local=local, commercial=cloud).execute(request)
                self.assertEqual(result.status, 'needs_review')
                self.assertFalse(cloud.requests)

    def test_defaults_remain_manual_cpu_private_offline_without_validator(self):
        request = build_request(DEFAULTS, 'Answer')
        self.assertEqual(request.mode, 'local')
        self.assertEqual(request.local_profile, 'cpu-only')
        self.assertTrue(request.private and request.offline)
        self.assertFalse(request.allow_gpu or request.allow_escalation)
        self.assertIsNone(request.validator)
        self.assertEqual(request.max_cost_usd, 0)
        self.assertEqual(request.host_cache_mib, 0)

    def test_host_cache_control_only_applies_to_eligible_idle_local_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.window(Path(directory)/'settings.json')
            window.refresh_profile()
            window.host_cache_box.configure.assert_called_with(state='disabled')
            window.device.set('GPU')
            window.change_device()
            window.host_cache_box.configure.assert_called_with(state='disabled')
            for preset in STREAMING_GPU_PROFILES:
                with self.subTest(preset=preset):
                    window.configure_profiles(preset)
                    window.host_cache_box.configure.assert_called_with(state='normal')
            window.host_cache_mib.set('768')
            self.assertEqual(SettingsWindow.settings(window)['host_cache_mib'], 768)
            window.busy(True)
            window.host_cache_box.configure.assert_called_with(state='disabled')
            self.assertEqual(window.host_cache_mib.get(), '768')
            window.busy(False)
            window.host_cache_box.configure.assert_called_with(state='normal')
            for route in ('Commercial', 'Suggest', 'Auto'):
                with self.subTest(route=route):
                    window.route.set(route)
                    window.host_cache_mib.set('invalid')
                    self.assertEqual(SettingsWindow.settings(window)['host_cache_mib'], 0)
                    window.refresh_host_cache()
                    self.assertEqual(window.host_cache_mib.get(), '0')
                    window.host_cache_box.configure.assert_called_with(state='disabled')
            window.route.set('Local')
            window.refresh_host_cache()
            self.assertEqual(window.host_cache_mib.get(), '0')
            window.host_cache_box.configure.assert_called_with(state='normal')
            window.host_cache_mib.set('768')
            window.configure_profiles('gpu-fast')
            self.assertEqual(window.host_cache_mib.get(), '0')
            window.host_cache_box.configure.assert_called_with(state='disabled')

    def test_positive_host_cache_requires_gpu_and_local_mode(self):
        request = build_request({**DEFAULTS, 'device':'GPU', 'preset':'balanced', 'host_cache_mib':768}, 'Answer')
        for changes in ({'allow_gpu':False}, {'mode':'commercial'}, {'mode':'suggest'}, {'mode':'auto'}):
            with self.subTest(changes=changes), self.assertRaises(ExecutionError):
                replace(request, **changes).check()

    def test_until_finished_is_local_with_a_positive_deadline(self):
        started = time.monotonic()
        request = build_request({**DEFAULTS, 'tokens':None}, 'Answer', deadline_s='30')
        self.assertIsNone(request.output_tokens)
        self.assertGreater(request.deadline, started)
        self.assertLessEqual(request.deadline, time.monotonic()+30)
        self.assertTrue(request.private and request.offline)
        self.assertEqual(request.allowed_destinations, ('local',))
        self.assertEqual(request.max_cost_usd, 0)
        for deadline in (None, 0, -1, float('inf'), float('nan'), True):
            with self.subTest(deadline=deadline), self.assertRaises(ExecutionError):
                build_request({**DEFAULTS, 'tokens':None}, 'Answer', deadline_s=deadline)

    def test_until_finished_rejects_other_modes_and_commercial_escalation(self):
        for change in ({'mode':'commercial'}, {'mode':'suggest'}, {'mode':'auto'}, {'allow_escalation':True}):
            with self.subTest(change=change), self.assertRaisesRegex(ExecutionError, 'Until finished requires Local'):
                build_request({**DEFAULTS, 'tokens':None}, 'Answer', **change)

    def test_until_finished_checkbox_ignores_disabled_spin_value_and_restores_it(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.window(Path(directory)/'settings.json')
            variable = lambda value: SimpleNamespace(get=lambda: value)
            window.device, window.profile = variable('CPU'), variable('CPU only')
            window.threads, window.context, window.tokens = variable('8'), variable('4096'), variable('invalid')
            window.until_finished.set(True)
            self.assertIsNone(SettingsWindow.settings(window)['tokens'])
            window.busy(True)
            window.until_finished_box.configure.assert_called_with(state='disabled')
            window.busy(False)
            window.until_finished_box.configure.assert_called_with(state='normal')
            window.token_box.configure.assert_called_with(state='disabled')
            window.until_finished.set(False)
            window.tokens = variable('256')
            window.refresh_answer_limit()
            window.token_box.configure.assert_called_with(state='normal')
            self.assertEqual(SettingsWindow.settings(window)['tokens'], 256)

    def test_until_finished_controller_runs_local_and_persists_null(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.window(Path(directory)/'settings.json')
            window.until_finished.set(True)
            window.settings = lambda: {**DEFAULTS, 'tokens':None}
            local = RecordingAdapter('finished')
            with mock.patch('bovary.gui.create_workbench', return_value=Workbench(local=local)), mock.patch('bovary.gui.messagebox.showerror') as error:
                window.run()
                self.pump_until_done(window)
            error.assert_not_called()
            self.assertEqual(len(local.requests), 1)
            self.assertIsNone(local.requests[0].output_tokens)
            self.assertIsNone(json.loads(window.settings_path.read_text())['tokens'])
            window.token_box.configure.assert_called_with(state='disabled')
            self.assertEqual(window.output.text, 'finished')

    def test_until_finished_other_modes_fail_before_backend_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ('Commercial', 'Suggest', 'Auto'):
                window = self.window(Path(directory)/'settings.json')
                window.settings = lambda: {**DEFAULTS, 'tokens':None}
                window.route.set(mode)
                with self.subTest(mode=mode), mock.patch('bovary.gui.create_workbench') as factory, mock.patch('bovary.gui.messagebox.showerror') as error:
                    window.run()
                factory.assert_not_called()
                error.assert_called_once()
                self.assertIn('Until finished requires Local', error.call_args.args[1])
                self.assertFalse(window.settings_path.exists())

    def test_incomplete_finish_reason_remains_visible_in_run_details(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.window(Path(directory)/'settings.json')
            window.events.put(('result', {'text':'partial', 'status':'incomplete', 'finish_reason':'context_limit'}))
            window.pump()
            self.assertEqual(window.output.text, 'partial')
            self.assertIn('context_limit', window.details.text)
            self.assertTrue(window.status.get().startswith('incomplete.'))

    def test_invalid_validator_files_and_blind_escalation_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'validator.json'
            for content in ('null', '[]', '{"kind":"unknown"}', '{"kind":"exact"}', '{', ' '*65537, '['*2000+']'*2000):
                path.write_text(content, encoding='utf-8')
                with self.subTest(content=content[:30]), self.assertRaises(ExecutionError):
                    build_request(DEFAULTS, 'Answer', validator_path=path)
            with self.assertRaises(ExecutionError):
                build_request(DEFAULTS, 'Answer', allow_escalation=True)

    def test_cli_rejects_invalid_validator_before_constructing_any_backend(self):
        import contextlib
        import io
        from unittest import mock
        from bovary import cli
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'invalid.json'
            path.write_text('{"kind":"exact"}', encoding='utf-8')
            with mock.patch.object(cli, 'create_workbench') as factory, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(['--prompt', 'Hello', '--validator', str(path)]), 2)
            factory.assert_not_called()

    def test_api_cannot_request_escalation_without_a_validator(self):
        local, cloud = RecordingAdapter('wrong'), RecordingAdapter('yes')
        result = Workbench(local=local, commercial=cloud).execute(Request('Answer', allow_escalation=True))
        self.assertEqual(result.error['code'], 'invalid_validator')
        self.assertEqual((local.requests, cloud.requests), ([], []))

    def test_validator_and_cloud_controls_never_persist_in_resource_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'settings.json'
            save_settings(path, {**DEFAULTS, 'validator': {'kind': 'exact', 'expected': 'private'},
                'provider_config': 'private-path', 'allow_escalation': True, 'api_key': 'do-not-save'})
            self.assertEqual(json.loads(path.read_text()), DEFAULTS)

    def test_gui_controller_wires_validator_and_copies_only_selected_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            validator = directory/'validator.json'
            validator.write_text('{"kind":"exact","expected":"yes"}', encoding='utf-8')
            window = self.window(directory/'settings.json')
            window.validator_path.set(str(validator))
            window.allow_escalation.set(True)
            window.offline.set(False)
            window.private.set(False)
            window.budget.set('1')
            local, cloud = RecordingAdapter('wrong'), RecordingAdapter('yes')
            workbench = Workbench(local=local, commercial=cloud)
            with mock.patch('bovary.gui.create_workbench', return_value=workbench), mock.patch('bovary.gui.messagebox.showerror') as error:
                window.run()
                self.pump_until_done(window)
            error.assert_not_called()
            self.assertEqual(len(local.requests), 1)
            self.assertEqual(len(cloud.requests), 1)
            self.assertEqual(window.output.text, 'yes')
            window.copy_answer()
            window.root.clipboard_append.assert_called_once_with('yes')
            self.assertEqual(json.loads(window.settings_path.read_text()), DEFAULTS)
            window.unload()

    def test_gui_stop_and_close_use_cancellation_without_desktop_input(self):
        entered = threading.Event()
        class CancellableAdapter(RecordingAdapter):
            def execute(self, request, on_event=None, cancel=None):
                entered.set()
                if not cancel.wait(2):
                    raise AssertionError('GUI never forwarded cancellation')
                raise ExecutionError('cancelled', 'fake cancelled')
        with tempfile.TemporaryDirectory() as directory:
            window = self.window(Path(directory)/'settings.json')
            workbench = Workbench(local=CancellableAdapter(''))
            with mock.patch('bovary.gui.create_workbench', return_value=workbench), mock.patch('bovary.gui.messagebox.showerror') as error:
                window.run()
                self.assertTrue(entered.wait(2))
                window.close()
                self.pump_until_done(window)
            error.assert_not_called()
            self.assertTrue(window.cancel.is_set())
            window.root.destroy.assert_called_once()

    def test_until_finished_interruption_keeps_partial_answer_and_copy(self):
        for code, status in (('deadline', 'failed'), ('cancelled', 'cancelled')):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                class InterruptedAdapter(RecordingAdapter):
                    def execute(self, request, on_event=None, cancel=None):
                        self.requests.append(request)
                        on_event(Event('text', request.request_id, text=self.text))
                        attempt = Result(request.request_id, 'local', text=self.text, status=status).to_dict()
                        raise ExecutionError(code, 'fixture interruption', attempt=attempt)
                window = self.window(Path(directory)/'settings.json')
                window.until_finished.set(True)
                window.settings = lambda: {**DEFAULTS, 'tokens':None}
                local = InterruptedAdapter('Useful partial answer')
                with mock.patch('bovary.gui.create_workbench', return_value=Workbench(local=local)), mock.patch('bovary.gui.messagebox.showerror') as error:
                    window.run()
                    self.pump_until_done(window)
                error.assert_not_called()
                self.assertEqual(len(local.requests), 1)
                self.assertIsNone(local.requests[0].output_tokens)
                self.assertEqual(window.output.text, 'Useful partial answer')
                self.assertTrue(window.status.get().startswith(status+'.'))
                self.assertIn('"code": "'+code+'"', window.details.text)
                window.copy_answer()
                window.root.clipboard_append.assert_called_once_with('Useful partial answer')


if __name__ == '__main__':
    unittest.main()
