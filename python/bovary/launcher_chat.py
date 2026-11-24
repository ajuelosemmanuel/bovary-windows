"""Interactive Gemma chat for the Windows launcher."""
import argparse
import json
import sys
import time

from .checkpoint import PinnedGemma
from .cli import configure_input, configure_output
from .contract import ExecutionError, Request
from .mini import CONTEXT_TOKENS, INPUT_BYTES, PRESETS, task_prompt


SWITCH_MODEL = 3
SAMPLE = "What is 17 plus 28? Answer with just the number."
MAX_TURNS = 4


def conversation_prompt(turns, question):
    if not turns:
        return question
    history = [{"user": user, "assistant": answer} for user, answer in turns]
    prompt = ("Continue the conversation below. Answer the latest user message.\n"
              + json.dumps({"previous_turns": history, "latest_user": question}, ensure_ascii=False))
    if len(prompt.encode("utf-8")) > INPUT_BYTES:
        raise ExecutionError("context_overflow", "Conversation exceeds the input limit")
    return prompt


def answer(worker, preset, turns, question):
    task_prompt(question, None)
    deadline = time.monotonic() + 180
    while True:
        try:
            prompt = conversation_prompt(turns, question)
            result = worker.execute(
                Request(prompt, local_profile=preset, allow_gpu=preset != "cpu-only",
                        threads=2, context_tokens=CONTEXT_TOKENS, output_tokens=96, deadline=deadline),
                on_event=lambda event: print(event.text, end="", flush=True),
            )
            print(flush=True)
            return result
        except ExecutionError as error:
            if error.code != "context_overflow" or not turns:
                raise
            turns.pop(0)


def chat(worker, preset):
    keep_ready = False
    turns = []
    backend = "CPU" if preset == "cpu-only" else "CUDA"
    print(f"\nGemma 26B ({backend}). Press Enter for a sample or type a question.")
    print("Each question starts fresh. Type /keep for follow-ups, /models to switch, or /quit.")
    while True:
        question = input("\nAsk Bovary: ").strip()
        command = question.casefold()
        if command == "/quit":
            return 0
        if command == "/models":
            return SWITCH_MODEL
        if command == "/keep" and not keep_ready:
            keep_ready = True
            print("After your next question, Gemma stays loaded and remembers recent turns. "
                  "Type /new to start a new conversation or /unload to release the model.")
            continue
        if command in {"/unload", "/keep"}:
            worker.unload()
            turns.clear()
            keep_ready = False
            print("Model unloaded and conversation cleared. Each question starts fresh.")
            continue
        if command == "/new":
            turns.clear()
            print("Conversation cleared.")
            continue
        question = question or SAMPLE
        print("Thinking with Gemma...", flush=True)
        try:
            result = answer(worker, preset, turns, question)
            if result.status != "complete":
                print("Answer reached its output limit. Ask for a shorter response.", file=sys.stderr)
            elif keep_ready:
                turns.append((question, result.text))
                del turns[:-MAX_TURNS]
        except (ExecutionError, OSError, ValueError) as error:
            print(f"Gemma request failed: {error}", file=sys.stderr)
            if preset != "cpu-only":
                print("Type /models and choose C to retry on CPU.")
        finally:
            if not keep_ready:
                worker.unload()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, required=True)
    args = parser.parse_args(argv)
    configure_input()
    configure_output()
    try:
        with PinnedGemma() as worker:
            return chat(worker, args.preset)
    except (EOFError, KeyboardInterrupt):
        print("\nClosed.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
