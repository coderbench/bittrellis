# Quantizer contract and lineage

A quantizer turns one searchable Linear into the tensors a checkpoint stores for it. The measured
evidence says this is as big a lever as the precision map: calibrated NVFP4 bytes alone cut
RP-KL by 11.5% at the same decode speed and memory (with 4.4% slower prefill). So quantizers are
first-class and pluggable.

## The contract

Code: [`bittrellis/quantizers/base.py`](../bittrellis/quantizers/base.py).

```python
class MyQuantizer(Quantizer):
    name = "gptq_nvfp4"          # used in manifests: quantizer: gptq_nvfp4
    version = 1                  # bump on ANY change to produced bytes: it is part of every candidate id
    formats = ("NVFP4",)         # execution formats it can produce
    lineage = "regenerable"      # runtime | regenerable | attested
    replay_mode = "sequential"   # independent | sequential   (regenerable only)

    def begin(self, ctx):        # once per build or audit replay, before the first encode
        ...

    def encode(self, ctx, unit, lin, fmt):
        w = f32_weight(ctx, lin)                  # canonical BF16 weight, float32
        params = ctx.params                       # per-unit params from the manifest
        # ... your algorithm ...
        return [(".weight", "U8", packed.shape, packed),
                (".weight_scale", "F8_E4M3", scales.shape, scales),
                (".weight_scale_2", "F32", (), np.asarray(ws2, "<f4").reshape(()))]
```

Register it in [`bittrellis/quantizers/__init__.py`](../bittrellis/quantizers/__init__.py) and
reference it from a manifest:

```yaml
rules:
  - match: "L*.mlp"
    format: NVFP4
    quantizer: gptq_nvfp4
    params: {damp: 0.01, blocksize: 128}
```

### What `encode` may and may not do

| Allowed | Not allowed (audit rejects or reviewers refuse) |
|---|---|
| read the unit's BF16 weight (`ctx.base`) | read or emit any other tensor's bytes |
| read the pinned public calibration manifest (`ctx.calibration`) | fetch data from the network at build time |
| scale, clipping, rounding, act-order and error-feedback search inside the tensor | move scales into norms or neighbouring Linears (SmoothQuant or AWQ-style migration) |
| keep state across units (`ctx.state`) when `replay_mode = "sequential"` | depend on wall-clock time, randomness without a fixed seed, or GPU nondeterminism |
| emit NVFP4 (ModelOpt layout) or FP8 (E4M3 + one BF16 scale per row) | emit a format the loader would silently convert (see [precision_space.md](precision_space.md)) |

**Determinism is the whole game.** The evaluator rebuilds your bytes and compares them exactly. If
your algorithm uses a GPU, pin deterministic kernels, or do the final rounding on CPU in float64.

## Lineage classes

| Class | Established by | Who can add one |
|---|---|---|
| `runtime` | stored BF16 is byte-identical to the base; SparkInfer fits the format at load (Q4_K) | nobody: it is the runtime's |
| `regenerable` | the audit rebuilds sampled units and compares bytes | anyone, by PR (evaluated after maintainer approval, since it runs contributed code) |
| `attested` | bytes identical to a source whose every file is sha256-pinned in `configs/sources.lock.json` | maintainers only |

Built-in quantizers (`bittrellis quantizers`):

| Name | Formats | Lineage | What it is |
|---|---|---|---|
| `runtime@v1` | Q4_K | runtime | stored BF16; Lloyd Q4_K fit at load |
| `baseline@v1` | NVFP4 | attested (`gittensor_nvfp4`) | the shipped ModelOpt round-to-nearest bytes |
| `unsloth@v1` | NVFP4 | attested (`unsloth_nvfp4`) | llm-compressor calibrated bytes, MLP layers 0–55 only |
| `rtn@v1` | NVFP4, FP8 | regenerable, independent | round-to-nearest from BF16 (NVFP4 max-calibrated, FP8 per row) |

### How the audit replays regenerable quantizers

- **Independent.** Six units per quantizer are sampled (deterministically from the candidate id,
  always including the first and last unit), rebuilt, and compared byte for byte.
- **Sequential.** The pipeline is replayed in order (layer ascending, unit order) from the first unit
  that uses the quantizer to the last sampled unit, because a deep tensor can depend on every earlier
  quantized one.
- **Replay cache.** A regenerated unit's tensor hashes are cached under a key made of the quantizer
  version, the base revision and the unit's *pipeline prefix* (every assignment to that quantizer up
  to and including the unit). Candidates that share a prefix share the replay.

### Anomaly check

For every NVFP4 and FP8 tensor the audit also reports sampled-row reconstruction error divided by
round-to-nearest error on the same rows. Calibrated encoders legitimately measure 1.2–1.6×
(unsloth's layer-0 `down_proj`: 1.61×). The audit warns above 2× and rejects above 5× as substituted
bytes. It is a diagnostic; legitimacy comes from lineage.

## Checklist for a quantizer PR

- [ ] one module in `bittrellis/quantizers/`, registered in `__init__.py`
- [ ] `version` set; bumped if the produced bytes change
- [ ] deterministic: `pytest` includes a test that encodes the tiny synthetic model twice and compares bytes
- [ ] one manifest in `manifests/` that uses it, with the hypothesis in `description`
- [ ] no changes to evaluator paths (see [CONTRIBUTING.md](../CONTRIBUTING.md))
