"""Bovary settings window; no model/device initialization until Run."""
import argparse
import codecs
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import uuid

from .runtime import project_home, create_workbench, provider_identity, load_validator
from .contract import ExecutionError, MAX_HOST_CACHE_MIB, Request, STREAMING_GPU_PROFILES
from .validators import validate_specification
import time

ROOT = project_home()
SETTINGS = ROOT/'work/bovary-settings.json'
GPU_PROFILES = {'Fast — more RAM': 'gpu-fast', 'Balanced — low RAM': 'balanced',
                'More GPU memory': 'gpu-more-vram', '2 GiB GPU expert cache': 'gpu-cache-2g',
                'Batched prompts — 2 GiB cache': 'gpu-prefill-2g'}
DEFAULTS = {'device': 'CPU', 'preset': 'cpu-only', 'threads': 8, 'context': 4096, 'tokens': 128,
            'host_cache_mib': 0}
DESCRIPTIONS = {
    'cpu-only': 'Leaves the GPU unused. Uses RAM and SSD. About 9.4 GiB resident process RAM in the short CPU test; longer prompts can use more.',
    'gpu-fast': 'Uses more system RAM for faster prompt processing. Prior 4K tests used about 12–13 GiB process RAM and 4.4 GiB GPU allocation. Leave room for the desktop and other apps.',
    'balanced': 'Streams expert weights from SSD to keep RAM low. Prior 4K tests used about 0.5 GiB process RAM and 3.6 GiB GPU allocation. Long prompts are slower to process.',
    'gpu-more-vram': 'Keeps more expert weights on the GPU to reduce SSD reads. Prior cache tests used about 4.1 GiB GPU allocation. This leaves less GPU memory for other apps.',
    'gpu-cache-2g': 'Streams weights with a 2 GiB GPU expert cache. Uses about 1 GiB more GPU memory than Balanced to reduce repeated reads.',
    'gpu-prefill-2g': 'Processes prompt tokens in batches while streaming weights with a 2 GiB GPU expert cache.',
}


def build_request(settings, prompt, *, mode='local', task='general', offline=True,
                  private=True, data_class='', budget=0, deadline_s=300,
                  validator_path=None, allow_escalation=False):
    """Build a detached request snapshot without a window, model or network."""
    settings = validate_settings(settings)
    if isinstance(deadline_s, bool) or isinstance(budget, bool):
        raise ExecutionError('invalid_request', 'Deadline and budget must be numbers')
    try:
        deadline_s, budget = float(deadline_s), float(budget)
    except (TypeError, ValueError) as error:
        raise ExecutionError('invalid_request', 'Deadline and budget must be numbers') from error
    if not math.isfinite(deadline_s) or not 0 < deadline_s <= 86400:
        raise ExecutionError('invalid_request', 'Deadline must be positive and at most 86400 seconds')
    if settings['tokens'] is None and (mode != 'local' or allow_escalation):
        raise ExecutionError('invalid_request', 'Until finished requires Local mode with commercial escalation off. Choose an answer limit for other modes.')
    validator = load_validator(validator_path)
    validate_specification(validator)
    if allow_escalation and validator is None:
        raise ExecutionError('invalid_validator', 'Choose a task validator before enabling escalation')
    request = Request(prompt=prompt, mode=mode, task=task,
        local_profile=settings['preset'], allow_gpu=settings['device'] == 'GPU',
        threads=settings['threads'], context_tokens=settings['context'], output_tokens=settings['tokens'],
        host_cache_mib=settings['host_cache_mib'] if mode == 'local' else 0,
        offline=offline, private=private,
        allowed_destinations=('local', 'commercial') if not offline and not private else ('local',),
        max_cost_usd=budget, deadline=time.monotonic()+deadline_s,
        identity={'data_class': data_class}, validator=validator, allow_escalation=allow_escalation)
    request.check()
    return request


def validate_settings(value):
    if not isinstance(value, dict):
        raise ValueError('Settings must be an object.')
    result = {key: value.get(key, default) for key, default in DEFAULTS.items()}
    if result['device'] not in ('CPU', 'GPU'):
        raise ValueError('Choose CPU or GPU.')
    valid = {'cpu-only'} if result['device'] == 'CPU' else set(GPU_PROFILES.values())
    if result['preset'] not in valid:
        raise ValueError('The selected profile does not match the device.')
    if result['preset'] not in STREAMING_GPU_PROFILES:
        result['host_cache_mib'] = 0
    elif type(result['host_cache_mib']) is not int or not 0 <= result['host_cache_mib'] <= MAX_HOST_CACHE_MIB:
        raise ValueError(f'Host expert cache must be a whole number from 0 to {MAX_HOST_CACHE_MIB} MiB.')
    for key in ('threads', 'context'):
        if type(result[key]) is not int:
            raise ValueError(f'{key} must be a whole number.')
    if not 1 <= result['threads'] <= 256:
        raise ValueError('CPU threads must be between 1 and 256.')
    if result['context'] not in (2048, 4096):
        raise ValueError('Choose a 2K or 4K context.')
    if result['tokens'] is not None and (type(result['tokens']) is not int or not 1 <= result['tokens'] < result['context']):
        raise ValueError('Answer limit must be positive and smaller than the context.')
    paths = value.get('paths')
    if paths is not None:
        if not isinstance(paths, dict) or any(k not in ('model', 'manifest', 'cpu_executable', 'cuda_executable') or not isinstance(v, str) or '\x00' in v for k, v in paths.items()):
            raise ValueError('Model paths must be text settings.')
        result['paths'] = dict(paths)
    return result


def effective_paths(saved=None, environ=None):
    environ = os.environ if environ is None else environ
    names = {'model': 'BOVARY_MODEL', 'manifest': 'BOVARY_MODEL_MANIFEST',
             'cpu_executable': 'BOVARY_CPU_EXE', 'cuda_executable': 'BOVARY_CUDA_EXE'}
    return {key: environ.get(env) or (saved or {}).get(key, '') for key, env in names.items()}


def resolve_setup_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def verification_command(paths):
    resolved = resolved_gui_paths(paths)
    verifier = ROOT / 'tools/verify_model.py'
    if not verifier.is_file():
        return ('Extract the Bovary source distribution, then run its tools/verify_model.py with '
                '--model <selected model> --manifest <new proof path>. Check the exit code before inference. '
                'This installed runtime does not contain the source verifier.')
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    return '& ' + quote(sys.executable) + ' ' + quote(verifier) + ' --model ' + quote(resolved['model']) + ' --manifest ' + quote(resolved['manifest'])


def resolved_gui_paths(paths, device='CPU'):
    from .doctor import resolve_paths
    key = 'cpu_executable' if device == 'CPU' else 'cuda_executable'
    executable = paths.get(key) or ('work/build/bovary-cpu/bovary-infer.exe' if device == 'CPU' else 'work/build/bovary-cuda/bovary-infer.exe')
    return resolve_paths(ROOT, model_path=paths.get('model') or None,
                         manifest_path=paths.get('manifest') or None,
                         executable=executable, use_environment=False)



def save_settings(path, settings):
    settings = validate_settings(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(settings, indent=2), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class SettingsWindow:
    def __init__(self, root, settings_path=SETTINGS):
        self.root, self.settings_path = root, settings_path
        self.process = None
        self.workbench = None
        self.workbench_cache = {}
        self.approved_provider_key = None
        self.cancel = threading.Event()
        self.closing = False
        self.events = queue.Queue()
        self.stop_file = None
        self.control = None
        self.job = None
        warning = None
        try:
            settings = validate_settings(json.loads(settings_path.read_text(encoding='utf-8')))
        except FileNotFoundError:
            settings = DEFAULTS.copy()
        except (OSError, UnicodeError, ValueError, TypeError) as error:
            settings = DEFAULTS.copy()
            warning = f'Saved settings could not be loaded; using CPU defaults. {error}'
        self.model_paths = effective_paths(settings.get('paths'))
        self.model_state = tk.StringVar(value='unloaded')
        self.setup_dialog = None
        self.device = tk.StringVar(value=settings['device'])
        self.profile = tk.StringVar()
        self.threads = tk.StringVar(value=str(settings['threads']))
        self.context = tk.StringVar(value=str(settings['context']))
        self.tokens = tk.StringVar(value=str(DEFAULTS['tokens'] if settings['tokens'] is None else settings['tokens']))
        self.host_cache_mib = tk.StringVar(value=str(settings['host_cache_mib']))
        self.until_finished = tk.BooleanVar(value=settings['tokens'] is None)
        self.route = tk.StringVar(value='Local')
        self.task = tk.StringVar(value='general')
        self.offline = tk.BooleanVar(value=True)
        self.private = tk.BooleanVar(value=True)
        self.provider_config = tk.StringVar()
        self.data_class = tk.StringVar(value='')
        self.validator_path = tk.StringVar(value='')
        self.allow_escalation = tk.BooleanVar(value=False)
        self.budget = tk.StringVar(value='0')
        self.timeout = tk.StringVar(value='300')
        self.description = tk.StringVar()
        self.status = tk.StringVar(value=warning or 'Ready. Choose a device and profile, then enter a prompt.')
        root.title('Bovary — Settings and Run')
        root.geometry('920x920')
        root.minsize(780, 680)
        root.protocol('WM_DELETE_WINDOW', self.close)
        style = ttk.Style(root)
        if 'vista' in style.theme_names():
            style.theme_use('vista')
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'))
        style.configure('TLabel', font=('Segoe UI', 10))
        outer = ttk.Frame(root, padding=16)
        outer.pack(fill='both', expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(5, weight=1)
        ttk.Label(outer, text='Bovary', style='Title.TLabel').grid(row=0, column=0, sticky='w')
        ttk.Label(outer, text='Choose how much RAM and GPU memory to trade for speed.').grid(row=1, column=0, sticky='w', pady=(4, 12))
        options = ttk.LabelFrame(outer, text='Run settings', padding=12)
        options.grid(row=2, column=0, sticky='ew')
        options.columnconfigure(3, weight=1)
        ttk.Label(options, text='Run on').grid(row=0, column=0, sticky='w')
        self.device_box = ttk.Combobox(options, textvariable=self.device, values=('CPU','GPU'), state='readonly', width=13)
        self.device_box.grid(row=0, column=1, sticky='w', padx=(8, 22))
        self.device_box.bind('<<ComboboxSelected>>', self.change_device)
        ttk.Label(options, text='Profile').grid(row=0, column=2, sticky='w')
        self.profile_box = ttk.Combobox(options, textvariable=self.profile, state='readonly', width=29)
        self.profile_box.grid(row=0, column=3, sticky='ew', padx=(8, 0))
        self.profile_box.bind('<<ComboboxSelected>>', lambda event: self.refresh_profile())
        ttk.Label(options, textvariable=self.description, wraplength=825, justify='left').grid(row=1, column=0, columnspan=4, sticky='w', pady=(12, 10))
        advanced = ttk.Frame(options)
        advanced.grid(row=2, column=0, columnspan=4, sticky='ew')
        ttk.Label(advanced, text='CPU threads').pack(side='left')
        self.thread_box = ttk.Spinbox(advanced, textvariable=self.threads, from_=1, to=max(8, os.cpu_count() or 8), width=5)
        self.thread_box.pack(side='left', padx=(6, 16))
        ttk.Label(advanced, text='Context').pack(side='left')
        self.context_box = ttk.Combobox(advanced, textvariable=self.context, values=('2048','4096'), state='readonly', width=7)
        self.context_box.pack(side='left', padx=(6, 16))
        ttk.Label(advanced, text='Answer limit').pack(side='left')
        self.token_box = ttk.Spinbox(advanced, textvariable=self.tokens, from_=1, to=4095, width=7)
        self.token_box.pack(side='left', padx=(6, 8))
        self.until_finished_box = ttk.Checkbutton(advanced, text='Until finished', variable=self.until_finished,
            command=self.refresh_answer_limit)
        self.until_finished_box.pack(side='left', padx=(0, 12))
        self.save_button = ttk.Button(advanced, text='Save settings', command=self.save)
        self.save_button.pack(side='right')
        ttk.Label(options, text='Until finished is local only. Stops when model finishes, context fills, deadline expires, or you cancel.',
                  wraplength=825).grid(row=3, column=0, columnspan=4, sticky='w', pady=(8, 0))
        ttk.Label(options, text='GPU memory figures exclude other apps and are estimates, not a live measurement. More CPU threads may not be faster.',
                  wraplength=825, foreground='#555555').grid(row=4, column=0, columnspan=4, sticky='w', pady=(10, 0))
        host_cache = ttk.Frame(options)
        host_cache.grid(row=5, column=0, columnspan=4, sticky='ew', pady=(8, 0))
        ttk.Label(host_cache, text='Host expert cache (MiB)').pack(side='left')
        self.host_cache_box = ttk.Spinbox(host_cache, textvariable=self.host_cache_mib,
            from_=0, to=MAX_HOST_CACHE_MIB, increment=256, width=6)
        self.host_cache_box.pack(side='left', padx=(6, 12))
        eligible_profiles = ', '.join(label for label, preset in GPU_PROFILES.items()
                                      if preset in STREAMING_GPU_PROFILES)
        ttk.Label(host_cache, text=f'Uses extra system RAM. 0 turns it off. Local GPU profiles: {eligible_profiles}.',
                  wraplength=560).pack(side='left', fill='x', expand=True)
        routing = ttk.LabelFrame(options, text='Destination (separate from CPU/GPU profile)', padding=6)
        routing.grid(row=6, column=0, columnspan=4, sticky='ew', pady=(8, 0))
        self.route_box = ttk.Combobox(routing, textvariable=self.route, values=('Local', 'Commercial', 'Suggest', 'Auto'), state='readonly', width=13)
        self.route_box.grid(row=0, column=0, padx=4)
        self.route_box.bind('<<ComboboxSelected>>', lambda event: self.refresh_host_cache())
        self.task_box = ttk.Combobox(routing, textvariable=self.task, values=('general', 'extract', 'classify', 'format', 'summarize', 'translate', 'reason', 'code'), state='readonly', width=12)
        self.task_box.grid(row=0, column=1, padx=4)
        self.offline_box = ttk.Checkbutton(routing, text='Offline', variable=self.offline)
        self.offline_box.grid(row=0, column=2)
        self.private_box = ttk.Checkbutton(routing, text='Private', variable=self.private)
        self.private_box.grid(row=0, column=3)
        ttk.Label(routing, text='Cap USD / deadline seconds').grid(row=1, column=0, columnspan=2, sticky='w')
        self.budget_entry = ttk.Entry(routing, textvariable=self.budget, width=9)
        self.budget_entry.grid(row=1, column=2)
        self.timeout_entry = ttk.Entry(routing, textvariable=self.timeout, width=9)
        self.timeout_entry.grid(row=1, column=3)
        ttk.Label(routing, text='Approved provider config').grid(row=2, column=0, columnspan=2, sticky='w')
        self.provider_entry = ttk.Entry(routing, textvariable=self.provider_config, width=45)
        self.provider_entry.grid(row=2, column=2, columnspan=2, sticky='ew')
        ttk.Label(routing, text='Suggest sends nothing. Auto requires qualified evidence and is disabled in this alpha. No API keys in settings.', wraplength=800).grid(row=3, column=0, columnspan=4, sticky='w')
        ttk.Label(routing, text='Authorized data class').grid(row=4, column=0, columnspan=2, sticky='w')
        self.data_entry = ttk.Entry(routing, textvariable=self.data_class, width=22)
        self.data_entry.grid(row=4, column=2, columnspan=2, sticky='ew')
        ttk.Label(routing, text='Validator JSON (optional)').grid(row=5, column=0, columnspan=2, sticky='w')
        self.validator_entry = ttk.Entry(routing, textvariable=self.validator_path, width=45)
        self.validator_entry.grid(row=5, column=2, columnspan=2, sticky='ew')
        self.escalation_box = ttk.Checkbutton(routing,
            text='Allow one commercial escalation after failed local validation', variable=self.allow_escalation)
        self.escalation_box.grid(row=6, column=0, columnspan=4, sticky='w')
        self.request_controls = (self.offline_box, self.private_box, self.budget_entry,
            self.timeout_entry, self.provider_entry, self.data_entry, self.validator_entry, self.escalation_box)
        prompt_area = ttk.LabelFrame(outer, text='Prompt', padding=8)
        prompt_area.grid(row=3, column=0, sticky='ew', pady=(12, 8))
        self.prompt = scrolledtext.ScrolledText(prompt_area, height=4, wrap='word', font=('Segoe UI', 10))
        self.prompt.pack(fill='both', expand=True)
        actions = ttk.Frame(outer)
        actions.grid(row=4, column=0, sticky='ew', pady=(0, 8))
        self.run_button = ttk.Button(actions, text='Run', command=self.run)
        self.run_button.pack(side='left')
        self.stop_button = ttk.Button(actions, text='Stop', command=self.stop, state='disabled')
        self.stop_button.pack(side='left', padx=8)
        ttk.Button(actions, text='Copy answer', command=self.copy_answer).pack(side='left')
        self.setup_button = ttk.Button(actions, text='Model setup…', command=self.open_setup)
        self.setup_button.pack(side='left', padx=8)
        ttk.Label(actions, textvariable=self.model_state).pack(side='left', padx=8)
        self.unload_button = ttk.Button(actions, text='Unload model', command=self.unload)
        self.unload_button.pack(side='right')
        answer = ttk.LabelFrame(outer, text='Answer', padding=8)
        answer.grid(row=5, column=0, sticky='nsew')
        self.output = scrolledtext.ScrolledText(answer, height=9, wrap='word', state='disabled', font=('Segoe UI', 10))
        self.output.pack(fill='both', expand=True)
        ttk.Label(outer, textvariable=self.status, wraplength=870).grid(row=6, column=0, sticky='ew', pady=(8, 6))
        details = ttk.LabelFrame(outer, text='Run details', padding=6)
        details.grid(row=7, column=0, sticky='ew')
        self.details = scrolledtext.ScrolledText(details, height=3, wrap='word', state='disabled', font=('Consolas', 9))
        self.details.pack(fill='x')
        self.configure_profiles(settings['preset'])
        self.refresh_answer_limit()
        self.root.after(50, self.pump)

    def open_setup(self):
        if self.process is not None:
            return
        if self.setup_dialog is not None and self.setup_dialog.winfo_exists():
            self.setup_dialog.lift()
            return
        dialog = self.setup_dialog = tk.Toplevel(self.root)
        dialog.title('Bovary — Model setup')
        dialog.transient(self.root)
        dialog.grab_set()
        frame = ttk.Frame(dialog, padding=16)
        frame.pack(fill='both', expand=True)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text='Select existing files. Nothing is downloaded or loaded.', wraplength=680).grid(row=0, column=0, columnspan=3, sticky='w')
        variables = {}
        labels = {'model': 'Model GGUF', 'manifest': 'Verification manifest',
                  'cpu_executable': 'CPU executable', 'cuda_executable': 'GPU executable (optional)'}
        for row, (key, label) in enumerate(labels.items(), 1):
            variables[key] = tk.StringVar(value=self.model_paths.get(key, ''))
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky='w', pady=5)
            ttk.Entry(frame, textvariable=variables[key], width=64).grid(row=row, column=1, sticky='ew', padx=8)
            def browse(k=key):
                value = filedialog.askopenfilename(parent=dialog, title=labels[k],
                    filetypes=[('Model files', '*.gguf'), ('All files', '*.*')] if k == 'model' else
                    [('JSON files', '*.json'), ('All files', '*.*')] if k == 'manifest' else
                    [('Programs', '*.exe'), ('All files', '*.*')])
                if value:
                    variables[k].set(value)
            ttk.Button(frame, text='Browse…', command=browse).grid(row=row, column=2)
        ttk.Label(frame, text='Blank fields use runtime defaults. Relative paths use Bovary home. Environment values override saved settings at launch; selections here apply now.', wraplength=710).grid(row=5, column=0, columnspan=3, sticky='w', pady=8)
        report = scrolledtext.ScrolledText(frame, width=88, height=10, wrap='word')
        report.grid(row=6, column=0, columnspan=3, sticky='ew')
        def selected():
            return {key: variable.get().strip() for key, variable in variables.items()}
        def inspect_paths():
            from .doctor import check_model
            paths = selected()
            resolved = resolved_gui_paths(paths)
            checks = check_model(resolved['model'], resolved['manifest'])
            executable = resolved['executable']
            lines = [f"{item['status']}: {item['message']}" for item in checks]
            lines.append('CPU executable: ' + ('file exists; use bovary doctor for protocol/runtime checks.' if executable.is_file() else 'missing. Build with scripts\\build.cmd cpu 4 or select the CPU executable.'))
            lines.append('Stored proof and file size are not a fresh full-file hash verification.')
            lines.append('Verify offline in PowerShell (source checkout required):\n' + verification_command(paths))
            lines.append('\nIf verification fails, do not run the model. Choose a new manifest path for conflicting proofs.')
            report.delete('1.0', 'end')
            report.insert('end', '\n\n'.join(lines))
        def inspect():
            try:
                inspect_paths()
            except (OSError, ValueError, ExecutionError) as error:
                messagebox.showerror('Model setup', 'Check the selected paths. ' + str(error), parent=dialog)
        def apply():
            if self.process is not None:
                return
            try:
                paths = selected()
                for key, value in paths.items():
                    if value and not resolve_setup_path(value).is_file():
                        raise ValueError(labels[key] + ' does not exist. Select an existing file, or leave blank to use defaults.')
                # Unload before committing the selection. Provider objects/caps survive.
                self.commit_model_paths(paths)
                self.set_model_state('unloaded')
                self.status.set('Model paths saved locally. Model unloaded. Check prerequisites before running.')
                dialog.destroy()
            except (OSError, ValueError, ExecutionError) as error:
                messagebox.showerror('Model setup', str(error), parent=dialog)
        actions = ttk.Frame(frame)
        actions.grid(row=7, column=0, columnspan=3, sticky='ew', pady=(10, 0))
        ttk.Button(actions, text='Check paths / verification command', command=inspect).pack(side='left')
        ttk.Button(actions, text='Apply and save', command=apply).pack(side='right')
        ttk.Button(actions, text='Cancel', command=dialog.destroy).pack(side='right', padx=8)
        report.insert('end', 'Check paths to inspect the stored proof and see the exact offline verification command. Full-model hashing is performed outside this window.')

    def commit_model_paths(self, paths):
        if self.process is not None:
            raise ExecutionError('busy', 'Stop the active request before changing model paths.')
        if paths.get('model') or paths.get('manifest'):
            from .doctor import check_model
            resolved = resolved_gui_paths(paths)
            failures = [c for c in check_model(resolved['model'], resolved['manifest']) if c['status'] == 'fail']
            if failures:
                raise ValueError(failures[0]['message'] + ' Use Check paths for the verification command.')
        for workbench in self.workbench_cache.values():
            workbench.unload()
        self.set_model_state('unloaded')
        old_paths = self.model_paths
        self.model_paths = dict(paths)
        try:
            save_settings(self.settings_path, self.settings())
        except BaseException:
            self.model_paths = old_paths
            raise

    def configure_profiles(self, preset):
        if self.device.get() == 'CPU':
            self.profile_box.configure(values=('CPU only',))
            self.profile.set('CPU only')
        else:
            self.profile_box.configure(values=tuple(GPU_PROFILES))
            self.profile.set(next((label for label, key in GPU_PROFILES.items() if key == preset), next(iter(GPU_PROFILES))))
        self.refresh_profile()

    def change_device(self, event=None):
        self.configure_profiles('gpu-fast' if self.device.get() == 'GPU' else 'cpu-only')

    def settings(self):
        try:
            preset = self.selected_preset()
            value = {'device': self.device.get(),
                'preset': preset,
                'threads': int(self.threads.get()), 'context': int(self.context.get()),
                'tokens': None if self.until_finished.get() else int(self.tokens.get()),
                'host_cache_mib': int(self.host_cache_mib.get()) if self.host_cache_eligible() else 0}
            if getattr(self, 'model_paths', None):
                value['paths'] = dict(self.model_paths)
            return validate_settings(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f'Check the run settings. {error}') from error

    def selected_preset(self):
        return 'cpu-only' if self.device.get() == 'CPU' else GPU_PROFILES.get(self.profile.get())

    def host_cache_eligible(self):
        return self.route.get() == 'Local' and self.selected_preset() in STREAMING_GPU_PROFILES

    def refresh_profile(self):
        self.description.set(DESCRIPTIONS[self.selected_preset()])
        self.refresh_host_cache()

    def refresh_host_cache(self, *, busy=False):
        eligible = self.host_cache_eligible()
        if not eligible:
            self.host_cache_mib.set('0')
        self.host_cache_box.configure(state='normal' if eligible and not busy and self.process is None else 'disabled')

    def refresh_answer_limit(self):
        self.token_box.configure(state='disabled' if self.process is not None or self.until_finished.get() else 'normal')

    def save(self):
        try:
            save_settings(self.settings_path, self.settings())
            self.status.set('Settings saved. Prompts and answers are not stored in the settings file.')
        except (OSError, ValueError) as error:
            messagebox.showerror('Settings', str(error), parent=self.root)

    def busy(self, value):
        if hasattr(self, 'setup_button'):
            self.setup_button.configure(state='disabled' if value else 'normal')
        for widget in (self.device_box, self.profile_box, self.context_box, self.route_box, self.task_box):
            widget.configure(state='disabled' if value else 'readonly')
        for widget in (self.thread_box, self.until_finished_box, self.save_button, self.run_button, self.prompt):
            widget.configure(state='disabled' if value else 'normal')
        for widget in self.request_controls:
            widget.configure(state='disabled' if value else 'normal')
        self.token_box.configure(state='disabled' if value or self.until_finished.get() else 'normal')
        self.refresh_host_cache(busy=value)
        self.stop_button.configure(state='normal' if value else 'disabled')

    @staticmethod
    def append(widget, text):
        widget.configure(state='normal')
        widget.insert('end', text)
        widget.see('end')
        widget.configure(state='disabled')

    def run(self):
        if self.process is not None:
            return
        try:
            settings = self.settings()
            prompt = self.prompt.get('1.0', 'end-1c')
            request = build_request(settings, prompt, mode=self.route.get().lower(), task=self.task.get(),
                offline=self.offline.get(), private=self.private.get(), budget=self.budget.get(),
                deadline_s=self.timeout.get(), data_class=self.data_class.get().strip(),
                validator_path=self.validator_path.get().strip() or None,
                allow_escalation=self.allow_escalation.get())
            config = self.provider_config.get().strip() or None
            self.select_workbench(config)
            save_settings(self.settings_path, settings)
        except (OSError, ValueError, ExecutionError) as error:
            messagebox.showerror('Cannot run', str(error), parent=self.root)
            return
        for widget in (self.output, self.details):
            widget.configure(state='normal')
            widget.delete('1.0', 'end')
            widget.configure(state='disabled')
        self.cancel.clear()
        self._local_request = request.mode == 'local'
        self.process = True
        self.busy(True)
        self.unload_button.configure(state='disabled')
        self.set_model_state('ready' if self.local_loaded() else ('loading' if request.mode == 'local' else 'unloaded'))
        self.status.set('Running. Local weights stay in memory until Unload or window close.')

        def execute():
            try:
                result = self.workbench.execute(request, on_event=lambda e: self.events.put(('attempt', e.data.get('destination')) if e.kind == 'attempt' else ('output' if e.kind == 'text' else 'details', e.text)), cancel=self.cancel)
                self.events.put(('result', result.to_dict()))
            except Exception as error:
                self.events.put(('result', {'status': 'failed', 'error': {'code': getattr(error, 'code', 'internal'), 'message': str(error)}}))
        threading.Thread(target=execute, daemon=True).start()

    def select_workbench(self, config):
        config_key = provider_identity(config)
        if config_key and self.approved_provider_key and config_key != self.approved_provider_key:
            raise ExecutionError('provider_session_locked', 'Provider authorization is fixed for this window. Reconcile the existing session before approving a new configuration and restarting.')
        if self.workbench is None or getattr(self, '_provider_key', None) != config_key:
            self.unload()
            # Aliases, copies, local mode and Unload never renew the provider cap.
            if config_key not in self.workbench_cache:
                self.workbench_cache[config_key] = create_workbench(ROOT, config)
            self.workbench = self.workbench_cache[config_key]
            self._provider_key = config_key
            if config_key:
                self.approved_provider_key = config_key

        self.apply_model_paths()

    def local_loaded(self):
        return bool(getattr(getattr(self.workbench, 'local', None), 'loaded', False))

    def set_model_state(self, state):
        if hasattr(self, 'model_state'):
            self.model_state.set(state)

    def apply_model_paths(self):
        if not hasattr(self, 'model_paths') or self.workbench is None:
            return
        local = self.workbench.local
        paths = self.model_paths
        resolved = resolved_gui_paths(paths, self.device.get())
        values = tuple(resolved[k] for k in ('model', 'manifest', 'executable'))
        if (local.model_path, local.manifest_path, local.exe) != values:
            self.workbench.unload()
            local.model_path, local.manifest_path, local.exe = values
            self.set_model_state('unloaded')

    def stop(self):
        if self.process is not None:
            self.set_model_state('stopping')
            self.cancel.set()
            self.status.set('Stopping and releasing the worker...')
            self.stop_button.configure(state='disabled')

    def unload(self):
        if self.process is not None:
            return
        if self.workbench is not None:
            self.workbench.unload()
        self.set_model_state('unloaded')
        self.status.set('Model unloaded. Retained local memory released.')

    def pump(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'attempt':
                    self._local_request = value == 'local'
                    if not self.cancel.is_set() and not self.closing:
                        self.set_model_state(('ready' if self.local_loaded() else 'loading') if self._local_request else ('ready' if self.local_loaded() else 'unloaded'))
                elif kind == 'result':
                    self.process = None
                    self.set_model_state('ready' if self.local_loaded() else 'unloaded')
                    self.busy(False)
                    self.unload_button.configure(state='normal')
                    # Results stay in memory; prompt/response logging is off by default.
                    # Streaming may contain a discarded local attempt. Copy answer
                    # must contain only the final selected result after escalation.
                    self.output.configure(state='normal')
                    self.output.delete('1.0', 'end')
                    self.output.insert('end', value.get('text', ''))
                    self.output.configure(state='disabled')
                    self.append(self.details, json.dumps({k: v for k, v in value.items() if k != 'text'}, ensure_ascii=False, indent=2))
                    self.status.set(value.get('status', 'failed') + '. Local weights may remain loaded; use Unload to release memory.')
                    if self.closing:
                        self.unload()
                        self.root.destroy()
                        return
                elif value:
                    if kind == 'output' and self.local_loaded() and getattr(self, '_local_request', False) and not self.cancel.is_set() and not self.closing:
                        self.set_model_state('generating')
                    self.append(self.output if kind == 'output' else self.details, value)
        except queue.Empty:
            pass
        self.root.after(50, self.pump)

    def copy_answer(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.output.get('1.0', 'end-1c'))

    def close(self):
        if self.process is None:
            self.unload()
            self.root.destroy()
        else:
            self.closing = True
            self.stop()


def main():
    parser = argparse.ArgumentParser(prog='bovary-settings', description=__doc__)
    parser.add_argument('--settings-file', type=Path, default=SETTINGS)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    root = tk.Tk()
    if args.smoke_test:
        root.withdraw()
    window = SettingsWindow(root, args.settings_file)
    if args.smoke_test:
        root.update_idletasks()
        safe_settings = window.settings()
        if 'paths' in safe_settings:
            safe_settings['paths'] = {key: '<redacted>' for key in safe_settings['paths']}
        print(json.dumps({'settings': safe_settings, 'title': root.title(), 'model_started': window.process is not None}))
        root.destroy()
    else:
        root.mainloop()


if __name__ == '__main__':
    main()
