import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from bovary.contract import ExecutionError, Request
from bovary.provider import OpenAIAdapter, ProviderConfig, ResponsesTransport


class FakeTransport:
    def __init__(self, events=None, error=None):
        self.calls = []
        self.sequence = events if events is not None else [
            {"type": "response.output_text.delta", "delta": "Hello"},
            {"type": "response.completed", "response": {"usage": {"input_tokens": 3, "output_tokens": 1,
             "input_tokens_details": {"cached_tokens": 1}, "output_tokens_details": {"reasoning_tokens": 0}}}}]
        self.error = error

    def events(self, payload, **kwargs):
        self.calls.append((payload, {k: v for k, v in kwargs.items() if k != "key"}))
        if self.error:
            raise self.error
        yield from self.sequence


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.config = ProviderConfig(enabled=True, account="fake-account-not-live", session_cap_usd=1,
                                     input_usd_per_million=1, output_usd_per_million=2, pricing_version="fixture-pricing-v1")
        self.request = Request("Hello", mode="commercial", offline=False, private=False,
                               allowed_destinations=("commercial",), max_cost_usd=.1,
                               deadline=time.monotonic() + 20, identity={"data_class": "public"})

    def test_all_preflight_gates_prove_zero_outbound(self):
        cases = [({"offline": True}, "privacy"), ({"private": True}, "privacy"),
                 ({"output_tokens": None}, "invalid_request"),
                 ({"allowed_destinations": ("local",)}, "privacy"), ({"max_cost_usd": 0}, "budget"),
                 ({"max_cost_usd": .00000001}, "budget"), ({"deadline": time.monotonic() - 1}, "deadline"),
                 ({"deadline": None}, "deadline_required"), ({"identity": {}}, "data_forbidden"),
                 ({"context_tokens": 256}, "context"), ({"prompt": "x" * 5000}, "context"),
                 ({"local_profile": "balanced"}, "gpu_forbidden")]
        for changes, code in cases:
            with self.subTest(changes=changes), patch.object(os, 'environ', {**os.environ, "OPENAI_API_KEY": "fake"}):
                request = copy.deepcopy(self.request)
                for key, value in changes.items():
                    setattr(request, key, value)
                transport = FakeTransport()
                adapter = OpenAIAdapter(self.config, transport)
                with self.assertRaises(ExecutionError) as raised:
                    adapter.execute(request)
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(transport.calls, [])

    def test_direct_adapter_cannot_use_local_until_finished_to_bypass_paid_cap(self):
        request = copy.deepcopy(self.request)
        request.mode, request.output_tokens = 'local', None
        transport = FakeTransport()
        adapter = OpenAIAdapter(self.config, transport)
        with self.assertRaises(ExecutionError) as raised:
            adapter.execute(request)
        self.assertEqual(raised.exception.code, 'invalid_request')
        self.assertEqual(transport.calls, [])
        self.assertEqual(adapter.committed_usd, 0)

    def test_defaults_missing_credentials_unknown_pricing_and_session_budget(self):
        for config, env, code in ((ProviderConfig(), {"OPENAI_API_KEY": "fake"}, "live_disabled"),
                                  (self.config, {}, "credentials"),
                                  (ProviderConfig(enabled=True, account="fake"), {"OPENAI_API_KEY": "fake"}, "pricing_unknown"),
                                  (ProviderConfig(**{**self.config.__dict__, "session_cap_usd": 0}), {"OPENAI_API_KEY": "fake"}, "budget")):
            transport = FakeTransport()
            with self.subTest(code=code), patch.object(os, 'environ', env), self.assertRaises(ExecutionError) as raised:
                OpenAIAdapter(config, transport).execute(self.request)
            self.assertEqual(raised.exception.code, code)
            self.assertFalse(transport.calls)

    def test_invalid_provider_configuration_shapes_fail_with_typed_error(self):
        cases = []
        for field in ("account", "model", "configuration", "pricing_version", "capabilities_checked"):
            for value in ({}, [], None, False, 123):
                cases.append({field: value})
        cases.extend(({"model": "  "}, {"configuration": ""}, {"permitted_data": {}},
                      {"permitted_data": "public"}, {"permitted_data": [None]}))
        for changes in cases:
            transport = FakeTransport()
            with self.subTest(changes=changes), self.assertRaises(ExecutionError) as raised:
                OpenAIAdapter(ProviderConfig(**{**self.config.__dict__, **changes}), transport)
            self.assertEqual(raised.exception.code, "provider_config")
            self.assertFalse(transport.calls)
        # An empty account is valid configuration while live use is disabled.
        OpenAIAdapter(ProviderConfig(account=""), FakeTransport())

    def test_streamed_text_usage_and_reconciliation(self):
        events = []
        transport = FakeTransport()
        adapter = OpenAIAdapter(self.config, transport)
        with patch.object(os, 'environ', {**os.environ, "OPENAI_API_KEY": "fake"}):
            result = adapter.execute(self.request, on_event=events.append)
        self.assertEqual(result.text, "Hello")
        self.assertEqual(events[0].text, "Hello")
        self.assertEqual(result.usage["input_tokens_details"], {"cached_tokens": 1})
        self.assertAlmostEqual(result.cost_usd, .000005)
        self.assertAlmostEqual(adapter.committed_usd, result.cost_usd)
        self.assertFalse(transport.calls[0][0]["store"])
        self.assertEqual(transport.calls[0][0]["tools"], [])
        self.assertEqual(result.cost_status, "upper_bound_reported_usage")

    def test_huge_numeric_configuration_is_a_typed_preflight_error(self):
        for field in ('session_cap_usd', 'timeout_s', 'input_usd_per_million', 'output_usd_per_million'):
            transport = FakeTransport()
            with self.subTest(field=field), self.assertRaises(ExecutionError) as caught:
                OpenAIAdapter(ProviderConfig(**{**self.config.__dict__, field:10**1000}), transport)
            self.assertEqual(caught.exception.code, 'provider_config')
            self.assertEqual(transport.calls, [])

    def test_interrupt_at_stream_end_preserves_known_usage_or_unknown_reservation(self):
        for condition in ('cancelled', 'deadline'):
            for known_usage in (True, False):
                request = copy.deepcopy(self.request)
                cancel = threading.Event()
                class EndingTransport(FakeTransport):
                    def events(self, payload, **kwargs):
                        self.calls.append(payload)
                        yield {'type':'response.output_text.delta', 'delta':'Hello'}
                        yield {'type':'response.completed', 'response':
                               {'usage':{'input_tokens':3, 'output_tokens':1}} if known_usage else {}}
                        if condition == 'cancelled':
                            cancel.set()
                        else:
                            request.deadline = time.monotonic() - 1
                transport = EndingTransport()
                adapter = OpenAIAdapter(self.config, transport)
                with self.subTest(condition=condition, known_usage=known_usage), patch.object(os, 'environ', {**os.environ, 'OPENAI_API_KEY':'fake'}):
                    result = adapter.execute(request, cancel=cancel)
                self.assertEqual(result.status, 'cancelled' if condition == 'cancelled' else 'failed')
                self.assertEqual(result.error['code'], condition)
                self.assertEqual(result.error['uncertain_billing'], not known_usage)
                self.assertEqual(result.text, 'Hello')
                self.assertEqual(len(transport.calls), 1)
                if known_usage:
                    self.assertEqual(result.usage['input_tokens'], 3)
                    self.assertEqual(result.usage['output_tokens'], 1)
                    self.assertAlmostEqual(result.cost_usd, .000005)
                    self.assertAlmostEqual(adapter.committed_usd, result.cost_usd)
                    self.assertEqual(result.cost_status, 'upper_bound_reported_usage')
                else:
                    self.assertIsNone(result.cost_usd)
                    self.assertGreater(adapter.committed_usd, 0)
                    self.assertEqual(result.cost_status, 'unknown_reserved')

    def test_missing_usage_retains_reservation_and_does_not_claim_zero(self):
        transport = FakeTransport([{"type": "response.completed", "response": {}}])
        adapter = OpenAIAdapter(self.config, transport)
        with patch.object(os, 'environ', {**os.environ, "OPENAI_API_KEY": "fake"}):
            result = adapter.execute(self.request)
        self.assertEqual(result.cost_status, "unknown_reserved")
        self.assertIsNone(result.cost_usd)
        self.assertGreater(adapter.committed_usd, 0)

    def test_usage_overrun_disables_future_paid_requests(self):
        transport = FakeTransport([{"type": "response.completed", "response": {"usage": {"input_tokens": 99999, "output_tokens": 1}}}])
        adapter = OpenAIAdapter(self.config, transport)
        with patch.object(os, 'environ', {**os.environ, "OPENAI_API_KEY": "fake"}):
            result = adapter.execute(self.request)
            self.assertEqual(result.status, "needs_review")
            with self.assertRaises(ExecutionError):
                adapter.execute(self.request)
        self.assertEqual(len(transport.calls), 1)

    def test_errors_are_bounded_without_hidden_retry(self):
        for error in (ExecutionError("rate_limit", "429"),
                      ExecutionError("provider_transport", "lost response", uncertain_billing=True)):
            transport = FakeTransport(error=error)
            adapter = OpenAIAdapter(self.config, transport)
            with self.subTest(error=error), patch.object(os, 'environ', {**os.environ, "OPENAI_API_KEY": "fake"}):
                result = adapter.execute(self.request)
            self.assertEqual(result.status, "failed")
            self.assertEqual(len(transport.calls), 1)
            self.assertEqual(adapter.committed_usd > 0, error.uncertain_billing)

    def test_preset_cancellation_sends_nothing(self):
        cancel = threading.Event()
        cancel.set()
        transport = FakeTransport()
        with self.assertRaises(ExecutionError):
            OpenAIAdapter(self.config, transport).execute(self.request, cancel=cancel)
        self.assertFalse(transport.calls)

    def test_midstream_cancellation_retains_uncertain_allowance(self):
        cancel = threading.Event()
        class CancellingTransport(FakeTransport):
            def events(self, payload, **kwargs):
                yield {"type": "response.output_text.delta", "delta": "first"}
                cancel.set()
                yield {"type": "response.output_text.delta", "delta": "second"}
        adapter = OpenAIAdapter(self.config, CancellingTransport())
        with patch.object(os, 'environ', {**os.environ, "OPENAI_API_KEY": "fake"}):
            result = adapter.execute(self.request, cancel=cancel)
        self.assertEqual((result.status, result.text, result.cost_status), ("cancelled", "first", "unknown_reserved"))

    def test_truncation_and_stream_without_terminal_are_not_success(self):
        sequences = [([{"type": "response.incomplete", "response": {"usage": {"input_tokens": 3, "output_tokens": 128}}}], "needs_review"),
                     ([{"type": "response.refusal.delta", "delta": "refused"}, {"type": "response.completed", "response": {"usage": {"input_tokens": 3, "output_tokens": 2}}}], "needs_review"),
                     ([{"type": "response.output_text.delta", "delta": "partial"}], "failed"),
                     ([{"type": "error", "message": "must not log provider body"}], "failed")]
        for sequence, status in sequences:
            with self.subTest(status=status), patch.object(os, 'environ', {**os.environ, "OPENAI_API_KEY": "fake"}):
                result = OpenAIAdapter(self.config, FakeTransport(sequence)).execute(self.request)
            self.assertEqual(result.status, status)
            self.assertNotIn("must not log", json.dumps(result.to_dict()))

    def test_real_transport_sse_parser_and_fixed_origin_without_network(self):
        import io
        class FakeResponse:
            status = 200
            def __init__(self):
                self.stream = io.BytesIO(b': keepalive\n\ndata: {"type":"response.output_text.delta","delta":"hello"}\n\ndata: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":1}}}\n\n')
            def readline(self, bound):
                return self.stream.readline(bound)
        class FakeConnection:
            sock = None
            def __init__(self, host, timeout):
                self.host, self.timeout, self.closed = host, timeout, False
            def request(self, method, path, body, headers):
                self.sent = (method, path, json.loads(body), headers)
            def getresponse(self):
                return FakeResponse()
            def close(self):
                self.closed = True
        connection = FakeConnection("unused", 1)
        with patch("bovary.provider.http.client.HTTPSConnection", return_value=connection) as connect, patch.object(os, 'environ', {**os.environ, "OPENAI_API_KEY": "fake-secret"}):
            result = OpenAIAdapter(self.config, ResponsesTransport()).execute(self.request)
        self.assertEqual(result.text, "hello")
        self.assertTrue(connection.closed)
        self.assertEqual(connect.call_args.args, ("api.openai.com",))
        self.assertEqual(connection.sent[:2], ("POST", "/v1/responses"))
        self.assertEqual(connection.sent[3]["Authorization"], "Bearer fake-secret")
        self.assertNotIn("fake-secret", json.dumps(result.to_dict()))

    def test_http_timeout_and_conflict_retain_reservations_without_retry(self):
        from types import SimpleNamespace
        for status in (408, 409, 500, 429):
            class FakeConnection:
                sock = None
                requests = 0
                def request(self, *args):
                    self.requests += 1
                def getresponse(self):
                    return SimpleNamespace(status=status)
                def close(self):
                    pass
            connection = FakeConnection()
            adapter = OpenAIAdapter(self.config, ResponsesTransport())
            with self.subTest(status=status), patch("bovary.provider.http.client.HTTPSConnection", return_value=connection), patch.object(os, 'environ', {**os.environ, "OPENAI_API_KEY": "fake"}):
                result = adapter.execute(self.request)
            uncertain = status in (408, 409, 500)
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.error["uncertain_billing"], uncertain)
            self.assertEqual(adapter.committed_usd > 0, uncertain)
            self.assertEqual(connection.requests, 1)

    def test_deadline_between_stream_events_retains_reservation(self):
        request = self.request
        class SlowFakeTransport(FakeTransport):
            def events(self, payload, **kwargs):
                request.deadline = time.monotonic() - 1
                yield {"type": "response.output_text.delta", "delta": "late"}
        adapter = OpenAIAdapter(self.config, SlowFakeTransport())
        with patch.object(os, 'environ', {**os.environ, "OPENAI_API_KEY": "fake"}):
            result = adapter.execute(self.request)
        self.assertEqual(result.error["code"], "deadline")
        self.assertEqual(result.cost_status, "unknown_reserved")


if __name__ == "__main__":
    unittest.main()
