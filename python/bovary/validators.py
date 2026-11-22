"""Versioned deterministic validators with semantic expectations.

Code fixtures accept a deliberately restricted pure expression function. This
is not a sandbox for general Python or arbitrary tests supplied by a model.
"""
import ast
import json
import math
import os
import subprocess
import sys
import time

from .contract import ExecutionError

VERSION = "validators-v2"
CODE_FIXTURES = {
    "sum_two-v1": [([2, 3], 5), ([-4, 7], 3), ([0, 0], 0), ([100, -101], -1)],
    "absolute_difference-v1": [([2, 5], 3), ([7, -4], 11), ([0, 0], 0), ([-8, -3], 5)],
}


def validate_specification(specification):
    """Reject invalid rubric configuration before model/provider execution."""
    if specification is None:
        return
    if not isinstance(specification, dict):
        raise ExecutionError("invalid_validator", "Validator must be an object")
    kind = specification.get("kind")
    fields = {
        "json_fields": ({"kind", "expected"}, {"kind", "expected", "exact_fields"}),
        "classification": ({"kind", "labels", "expected"}, {"kind", "labels", "expected"}),
        "exact": ({"kind", "expected"}, {"kind", "expected"}),
        "code_fixture": ({"kind", "fixture"}, {"kind", "fixture"}),
    }
    if not isinstance(kind, str) or kind not in fields:
        raise ExecutionError("invalid_validator", "Unknown validator kind")
    required, allowed = fields[kind]
    if not required.issubset(specification) or set(specification) - allowed:
        raise ExecutionError("invalid_validator", f"Missing or unsupported fields for {kind} validator")
    if kind == "json_fields":
        expected = specification["expected"]
        if not isinstance(expected, dict) or not expected:
            raise ExecutionError("invalid_validator", "JSON field validator needs a nonempty expected object")
        if type(specification.get("exact_fields", True)) is not bool:
            raise ExecutionError("invalid_validator", "exact_fields must be boolean")
        def check_json(value, depth=0):
            if depth > 64:
                raise ExecutionError("invalid_validator", "Expected JSON values exceed nesting limit")
            if type(value) in (str, int, bool) or value is None:
                return
            if type(value) is float and math.isfinite(value):
                return
            if type(value) is list:
                for item in value:
                    check_json(item, depth + 1)
                return
            if type(value) is dict and all(isinstance(key, str) for key in value):
                for item in value.values():
                    check_json(item, depth + 1)
                return
            raise ExecutionError("invalid_validator", "Expected field values must be finite JSON values with string object keys")
        check_json(expected)
    elif kind == "classification":
        labels = specification["labels"]
        if (not isinstance(labels, list) or not labels or
                any(not isinstance(label, str) or not label.strip() for label in labels) or
                len(set(labels)) != len(labels)):
            raise ExecutionError("invalid_validator", "Classification labels must be a nonempty list of unique text labels")
        if not isinstance(specification["expected"], str) or specification["expected"] not in labels:
            raise ExecutionError("invalid_validator", "Expected class must be one of the configured labels")
    elif kind == "exact":
        if not isinstance(specification["expected"], str):
            raise ExecutionError("invalid_validator", "Exact validator needs a text expectation")
    elif not isinstance(specification["fixture"], str) or specification["fixture"] not in CODE_FIXTURES:
        raise ExecutionError("invalid_validator", "Unknown controlled code fixture")


def _strict_json_equal(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(_strict_json_equal(value[key], wanted) for key, wanted in expected.items())
    if isinstance(expected, list):
        return len(value) == len(expected) and all(_strict_json_equal(item, wanted) for item, wanted in zip(value, expected))
    if isinstance(expected, float) and not math.isfinite(expected):
        return False
    return value == expected


def _code(text, fixture, deadline=None):
    if fixture not in CODE_FIXTURES or len(text) > 4096:
        return False, "Unknown controlled fixture or code exceeds 4 KiB"
    tree = ast.parse(text)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False, "Expected one solve(a, b) function"
    function = tree.body[0]
    arguments = function.args
    if (function.name != "solve" or function.decorator_list or function.returns is not None or
            arguments.vararg or arguments.kwarg or arguments.kwonlyargs or arguments.defaults or
            arguments.posonlyargs or len(arguments.args) != 2 or
            any(a.annotation is not None for a in arguments.args) or
            len(function.body) != 1 or not isinstance(function.body[0], ast.Return)):
        return False, "Controlled function must contain one pure return expression"
    names = {a.arg for a in arguments.args}
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Compare, ast.Name,
               ast.Constant, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
               ast.Mod, ast.USub, ast.UAdd, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
    expression = function.body[0].value
    if expression is None:
        return False, "Controlled function must return a numeric expression"
    nodes = list(ast.walk(expression))
    if len(nodes) > 64:
        return False, "Expression exceeds bounded fixture complexity"
    for node in nodes:
        if not isinstance(node, allowed):
            return False, "Only pure numeric expressions are allowed in controlled fixtures"
        if isinstance(node, ast.Name) and node.id not in names:
            return False, "Unexpected expression identifier"
        if isinstance(node, ast.Constant) and (type(node.value) not in (int, float) or abs(node.value) > 1000000):
            return False, "Only bounded numeric constants are allowed"
    harness = (
        "import json,sys\n"
        "data=json.load(sys.stdin)\n"
        "scope={'__builtins__': {}}\n"
        "exec(compile(data['code'], '<controlled-fixture>', 'exec'), scope)\n"
        "print(json.dumps([scope['solve'](*x) for x in data['inputs']], allow_nan=False))\n"
    )
    cases = CODE_FIXTURES[fixture]
    timeout = 5 if deadline is None else min(5, deadline - time.monotonic())
    if timeout <= 0:
        raise ExecutionError("deadline", "Deadline expired before controlled code validation")
    process = subprocess.run([sys.executable, "-I", "-S", "-c", harness],
                             input=json.dumps({"code": text, "inputs": [x for x, _ in cases]}),
                             text=True, capture_output=True, timeout=timeout,
                             env={k: v for k, v in os.environ.items() if k.upper() in ("SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP")},
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if process.returncode:
        return False, "Controlled code failed execution"
    return json.loads(process.stdout) == [y for _, y in cases], "Controlled fixture expected-value comparison"


def validate(text, specification, *, deadline=None, cancel=None):
    def check_constraints():
        if cancel is not None and cancel.is_set():
            raise ExecutionError("cancelled", "Cancelled during task validation")
        if deadline is not None and time.monotonic() >= deadline:
            raise ExecutionError("deadline", "Deadline expired during task validation")
    check_constraints()
    if specification is None:
        return {"status": "not_checked", "passed": None, "version": VERSION,
                "reason": "No task-specific validator was supplied"}
    try:
        kind = specification["kind"]
        if kind == "json_fields":
            value = json.loads(text)
            expected = specification["expected"]
            if not isinstance(value, dict) or not isinstance(expected, dict) or not expected:
                passed, reason = False, "Expected nonempty field semantics"
            else:
                passed = all(key in value and _strict_json_equal(value[key], wanted)
                             for key, wanted in expected.items())
                if specification.get("exact_fields", True):
                    passed = passed and set(value) == set(expected)
                reason = "Field values and types compared against independent expectations"
        elif kind == "classification":
            labels = specification["labels"]
            expected = specification["expected"]
            passed = expected in labels and text.strip() == expected
            reason = "Exact class label and independently specified expected class"
        elif kind == "exact":
            passed = isinstance(specification["expected"], str) and text == specification["expected"]
            reason = "Exact deterministic formatting and expected-value comparison"
        elif kind == "code_fixture":
            passed, reason = _code(text, specification["fixture"], deadline)
        else:
            passed, reason = False, "Unsupported validator"
    except (KeyError, TypeError, ValueError, SyntaxError, subprocess.TimeoutExpired, OSError, RecursionError):
        passed, reason = False, "Malformed output or invalid validator specification"
    check_constraints()
    return {"status": "pass" if passed else "fail", "passed": bool(passed), "version": VERSION, "reason": reason}
