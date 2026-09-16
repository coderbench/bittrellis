# Evaluation corpus

`hpc01-public-v2.json` holds the token ids every checkpoint is teacher-forced through. It is
built deterministically by `bittrellis corpus build` from pinned sources and is verified by hash
(`bittrellis corpus verify`); any edit invalidates it.

| stream | tokens | scored | source (pinned revision in the file) |
|---|---:|---:|---|
| short-general | 4,096 | all | WikiText-103 test (CC BY-SA 3.0) |
| short-math | 4,096 | all | GSM8K test, chat-formatted (MIT) |
| short-code | 4,096 | all | HumanEval prompts + canonical solutions (MIT) |
| short-tools | 4,096 | all | Hermes function-calling v1, single turn (Apache-2.0) |
| short-multilingual | 4,096 | all | WMT24++ en→10 languages (Apache-2.0) |
| long-8k / 16k / 32k | 8,192 / 16,384 / 32,768 | last 384 | WikiText-103 test + three planted vault codes |

Each document starts with `<|endoftext|>`, as in pretraining. Without it, Qwen3.8 is chaotic on
some contexts in BF16 as well as in 4-bit (see `bittrellis/eval/corpus.py`).

A sealed holdout corpus (`--split holdout`, secret seed) draws a disjoint half of every source for
validator-side evaluation.
