"""Run TrajGenAgent's LLM over prepared prompts and save the generated chains.

This is the only step of the TrajGenAgent baseline that needs an LLM, so it is
a standalone script with no imports from this repo: run it in an environment
that has vLLM (or Hugging Face transformers) and enough GPUs for the model.
The prompts come from `python -m src.detectors.trajgen_prompts`.

Each prompt yields one chain: the first Python list in the reply, keeping only
tokens from the dataset's vocabulary (`null` when no list can be parsed).
Output is appended as it goes, and prompts already answered are skipped, so an
interrupted run resumes where it stopped.

Usage (from the repo root, in the LLM environment):
    CUDA_VISIBLE_DEVICES=3,4,5,6 python scripts/baselines/generate_trajgen_chains.py \
        --dir artifacts/trajgen/geolife --windows train val test
"""

import argparse
import ast
import json
import os
import re
from pathlib import Path

GENERATION = dict(model="Qwen/Qwen2.5-32B-Instruct", temperature=0.9, top_p=0.95,
                  max_tokens=512, seed=42)


def parse_chain(text: str, vocabulary: set[str]) -> list[str] | None:
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = ast.literal_eval(match.group(0))
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, list):
        return None
    return [token for token in parsed if isinstance(token, str) and token in vocabulary]


class VLLMGenerator:
    """The model, loaded once and sharded across every visible GPU."""

    def __init__(self, args):
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM, SamplingParams

        n_gpus = len((os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(","))
        self.llm = LLM(model=args.model, trust_remote_code=True, gpu_memory_utilization=0.92,
                       max_model_len=8192, enforce_eager=True, seed=args.seed,
                       tensor_parallel_size=n_gpus)
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                       max_tokens=args.max_tokens, seed=args.seed)

    def __call__(self, prompts: list[str]) -> list[str]:
        chats = [self.tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                                    tokenize=False, add_generation_prompt=True)
                 for p in prompts]
        return [out.outputs[0].text for out in self.llm.generate(chats, self.sampling)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True,
                        help="folder holding prompts_<window>.jsonl and vocabulary.json")
    parser.add_argument("--windows", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch", type=int, default=20000,
                        help="prompts per generate call; output is saved after each")
    for key, value in GENERATION.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    args = parser.parse_args()

    vocabulary = set(json.loads((args.dir / "vocabulary.json").read_text()))
    generate = None
    for window in args.windows:
        prompts = [json.loads(line) for line in (args.dir / f"prompts_{window}.jsonl").open()]
        out_path = args.dir / f"chains_{window}.jsonl"
        done = set()
        if out_path.exists():
            done = {(r["agent_id"], r["date"]) for r in map(json.loads, out_path.open())}
        todo = [p for p in prompts if (p["agent_id"], p["date"]) not in done]
        print(f"{window}: {len(prompts):,} prompts, {len(todo):,} to generate")
        for start in range(0, len(todo), args.batch):
            generate = generate or VLLMGenerator(args)
            batch = todo[start:start + args.batch]
            replies = generate([p["prompt"] for p in batch])
            with out_path.open("a") as out:
                for record, reply in zip(batch, replies):
                    out.write(json.dumps({"agent_id": record["agent_id"], "date": record["date"],
                                          "chain": parse_chain(reply, vocabulary)}) + "\n")
            print(f"  {min(start + args.batch, len(todo)):,}/{len(todo):,} written to {out_path}")


if __name__ == "__main__":
    main()
