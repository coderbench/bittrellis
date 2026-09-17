"""The HPC-01 evaluation corpus: fixed token streams every candidate is teacher-forced through.

Short streams (one per category) are scored at every position. Long streams are prefilled up to
a tail window and scored only there; each carries "needles" (random codes planted at depths of
the context) whose answer tokens must still be predicted after the full context. Needles at
8K, 16K and 32K expose error that accumulates in the Gated DeltaNet recurrent state.

Sources are pinned by dataset revision. `split="holdout"` draws a disjoint sample with a secret
seed (BITTRELLIS_HOLDOUT_SEED) for sealed validator evaluation.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import urllib.request
from pathlib import Path

SOURCES = {
    "wikitext": ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3", "wikitext-103-raw-v1/test-00000-of-00001.parquet"),
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main/test-00000-of-00001.parquet"),
    "humaneval": ("openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544", "openai_humaneval/test-00000-of-00001.parquet"),
    "hermes": ("NousResearch/hermes-function-calling-v1", "dae3e1d28cfbcf4b915c04ea1e072030529b4bda", "func-calling-singleturn.json"),
    "wmt24pp": ("google/wmt24pp", "fd7405c06494bc66a57b25f55d217a72f96e60dc", None),
}
WMT_LANGS = ["zh_CN", "ja_JP", "ko_KR", "de_DE", "fr_FR", "es_MX", "ru_RU", "ar_SA", "hi_IN", "vi_VN"]
TOKENIZER = ("gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090", "5b7a687fc8211a5d631c8ca6a593dd37eb26ce33", "tokenizer.json")

SHORT_TOKENS = 4096
SHORT_CATEGORIES = ("general", "math", "code", "tools", "multilingual")
LONG_LENGTHS = (8192, 16384, 32768)
LONG_TAIL = 384
VERSION = "hpc01-v2"

CHAT_USER = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
CHAT_END = "<|im_end|>\n"


def fetch(repo: str, rev: str, file: str, cache: Path, dataset: bool = True) -> bytes:
    dest = cache / repo.replace("/", "__") / rev / file
    if dest.exists():
        return dest.read_bytes()
    kind = "datasets/" if dataset else ""
    url = f"https://huggingface.co/{kind}{repo}/resolve/{rev}/{file}"
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as r:
        data = r.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return data


def _parquet_rows(data: bytes) -> list[dict]:
    import pyarrow.parquet as pq  # optional dependency: pip install bittrellis[corpus]

    return pq.read_table(io.BytesIO(data)).to_pylist()


def _order(items: list, key, seed: str) -> list:
    return sorted(items, key=lambda it: hashlib.sha256(f"{seed}:{key(it)}".encode()).hexdigest())


def _split(items: list, key, split: str, seed: str) -> list:
    """Deterministic 50/50 partition by id hash: even hashes are "public", odd ones "public-validation"."""
    part = [it for it in items if (int(hashlib.sha256(f"part:{key(it)}".encode()).hexdigest(), 16) % 2 == 0) == (split == "public")]
    return _order(part, key, seed)


DOC_SEP = "<|endoftext|>"


def _fill(tok, docs: list[str], budget: int) -> list[int]:
    """Concatenate documents, each opened by <|endoftext|> exactly as in pretraining data.

    Without the separator Qwen3.8 is numerically chaotic on some contexts: BF16 transformers and
    SparkInfer NVFP4 both flip to confident garbage at the same positions of a JSON tool prompt,
    and a single leading <|endoftext|> restores normal predictions. Such positions add noise,
    not signal, to a divergence metric.
    """
    ids: list[int] = []
    for d in docs:
        ids.extend(tok.encode(DOC_SEP + d).ids)
        if len(ids) >= budget:
            break
    if len(ids) < budget:
        raise ValueError(f"not enough source text for {budget} tokens (have {len(ids)})")
    return ids[:budget]


def wikitext_articles(rows: list[dict]) -> list[str]:
    arts, cur = [], []
    for r in rows:
        line = r["text"]
        if line.startswith(" = ") and not line.startswith(" = = ") and cur:
            arts.append("".join(cur).strip())
            cur = []
        cur.append(line)
    if cur:
        arts.append("".join(cur).strip())
    return [_detok(a) for a in arts if len(a) > 2000]


def _detok(text: str) -> str:
    """Undo wikitext's tokenized spacing (` @-@ `, ` , `) so the stream reads like real prose."""
    for a, b in ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" , ", ", "), (" . ", ". "), (" ; ", "; "),
                 (" : ", ": "), (" 's", "'s"), ("( ", "("), (" )", ")"), (" n't", "n't")):
        text = text.replace(a, b)
    return text


def assemble_streams(tok, short: dict[str, list[str]], long_docs: list[str], seed: str) -> list[dict]:
    """Token streams from documents: one 4K stream per short category, 8K/16K/32K needle streams.

    Shared by the public corpus and validators' private holdouts, so both have the same structure.
    """
    streams: list[dict] = []
    for cat, docs in short.items():
        ids = _fill(tok, docs, SHORT_TOKENS)
        streams.append({"id": f"short-{cat}", "category": cat, "ids": ids, "score_from": 0, "needles": []})

    rng = random.Random(f"{seed}:needles")
    offset = 0
    for length in LONG_LENGTHS:
        colors = ["red", "green", "blue"]
        codes = {c: "".join(rng.choice("0123456789") for _ in range(7)) for c in colors}
        question = ("Three vault codes were hidden in the documents above. Reply with the codes exactly as "
                    "written, in the form: red=<code>, green=<code>, blue=<code>.")
        # Build the answer from separately tokenized pieces so every code digit's index is known.
        tail_ids = tok.encode(CHAT_USER.format(q=question)).ids
        code_spans: dict[str, tuple[int, int]] = {}
        for k, c in enumerate(colors):
            tail_ids += tok.encode(("" if k == 0 else ", ") + f"{c}=").ids
            a = len(tail_ids)
            tail_ids += tok.encode(codes[c]).ids
            code_spans[c] = (a, len(tail_ids))
        tail_ids += tok.encode(CHAT_END).ids
        docs = long_docs[offset:] + long_docs[:offset]
        offset += 8
        body = _fill(tok, docs, length)
        # Plant needles at 10%, 50% and 90% depth, on token boundaries of the filler.
        for depth, c in zip((0.90, 0.50, 0.10), reversed(colors), strict=True):
            at = int(len(body) * depth)
            needle = tok.encode(f"\n(The secret code for the {c} vault is {codes[c]}.)\n").ids
            body = body[:at] + needle + body[at:]
        body = body[: length - len(tail_ids)]
        ids = body + tail_ids
        base = len(body)
        needles = [{"name": c, "depth": d, "target_positions": list(range(base + code_spans[c][0], base + code_spans[c][1]))}
                   for c, d in zip(colors, (0.10, 0.50, 0.90), strict=True)]
        streams.append({"id": f"long-{length // 1024}k", "category": "long", "ids": ids,
                        "score_from": len(ids) - LONG_TAIL, "needles": needles})

    return streams


def build_corpus(cache: Path, split: str = "public", seed: str | None = None) -> dict:
    from tokenizers import Tokenizer

    # "public" is the development fidelity set every candidate is scored on. "public-validation" is the
    # other half of the same public sources: useful for checking overfitting locally, but it is NOT a
    # holdout -- anyone can rebuild it. The private holdout is built from unpublished text (holdout.py).
    if split not in ("public", "public-validation"):
        raise ValueError(split)
    seed = seed or "hpc01-public"
    tok_bytes = fetch(*TOKENIZER, cache=cache, dataset=False)
    tok = Tokenizer.from_str(tok_bytes.decode())

    wiki = wikitext_articles(_parquet_rows(fetch(*SOURCES["wikitext"], cache=cache)))
    wiki = _split(wiki, lambda a: a[:200], split, seed)
    gsm = _split(_parquet_rows(fetch(*SOURCES["gsm8k"], cache=cache)), lambda r: r["question"], split, seed)
    he = _split(_parquet_rows(fetch(*SOURCES["humaneval"], cache=cache)), lambda r: r["task_id"], split, seed)
    hermes = _split(json.loads(fetch(*SOURCES["hermes"], cache=cache)), lambda r: r["id"], split, seed)
    wmt: list[dict] = []
    for lang in WMT_LANGS:
        rev = SOURCES["wmt24pp"][1]
        for line in fetch(SOURCES["wmt24pp"][0], rev, f"en-{lang}.jsonl", cache).decode().splitlines():
            r = json.loads(line)
            if not r.get("is_bad_source") and r.get("target"):
                wmt.append({"lang": lang, **r})
    wmt = _split(wmt, lambda r: f"{r['lang']}:{r['segment_id']}", split, seed)

    def chat(q: str, a: str) -> str:
        return CHAT_USER.format(q=q.strip()) + a.strip() + CHAT_END

    short = {
        "general": wiki[:4],
        "math": [chat(r["question"], r["answer"].replace("####", "The answer is")) for r in gsm],
        "code": [chat("Complete this Python function.\n\n" + r["prompt"], "```python\n" + r["prompt"] + r["canonical_solution"] + "```") for r in he],
        "tools": [_hermes_chat(r) for r in hermes],
        "multilingual": [chat(f"Translate to {r['lang']}:\n{r['source']}", r["target"]) for r in wmt],
    }
    streams = assemble_streams(tok, short, wiki[4:], seed)

    body = {
        "version": VERSION, "split": split,
        "tokenizer": {"repo": TOKENIZER[0], "revision": TOKENIZER[1], "sha256": hashlib.sha256(tok_bytes).hexdigest()},
        "sources": {k: {"repo": v[0], "revision": v[1]} for k, v in SOURCES.items()},
        "streams": streams,
    }
    body["sha256"] = corpus_hash(body)
    return body


def _hermes_chat(r: dict) -> str:
    out = []
    for turn in r["conversations"]:
        role = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}.get(turn["from"], turn["from"])
        out.append(f"<|im_start|>{role}\n{turn['value'].strip()}<|im_end|>\n")
    return "".join(out)


def corpus_hash(corpus: dict) -> str:
    payload = {"version": corpus["version"], "split": corpus["split"],
               "streams": [{k: s[k] for k in ("id", "ids", "score_from", "needles")} for s in corpus["streams"]]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_corpus(corpus: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(corpus, separators=(",", ":")) + "\n")


def load_corpus(path: Path) -> dict:
    corpus = json.loads(Path(path).read_text())
    if corpus_hash(corpus) != corpus["sha256"]:
        raise ValueError(f"{path}: corpus hash mismatch (edited or corrupted)")
    return corpus
