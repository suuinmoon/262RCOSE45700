# Usage: TORCH_LOGS="recompiles,graph_breaks" python step3_recompile.py [BASE_LEN] [OTHER_LEN] 2>&1 | tee step3_<OTHER_LEN>.log
# Calls each length twice: BASE -> OTHER -> BASE (default 128 -> 64 -> 128)
import sys
import time
import traceback

import torch
import furiosa.torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Bypass mask creation (uses int64 ops unsupported on NPU).
# Inputs are fully filled with no padding, so no mask == attend to all tokens.
import transformers.models.bert.modeling_bert as _bert_mod
_bert_mod.create_bidirectional_mask = lambda *a, **k: None

MODEL = "textattack/bert-base-uncased-SST-2"
DEV = torch.device("furiosa:0")
BASE = int(sys.argv[1]) if len(sys.argv) > 1 else 128
OTHER = int(sys.argv[2]) if len(sys.argv) > 2 else 64

tok = AutoTokenizer.from_pretrained(MODEL)
model = (
    AutoModelForSequenceClassification.from_pretrained(MODEL)
    .to(torch.bfloat16).eval().to(DEV)
)


def make_input(length):
    text = "a short sentence for the recompilation check . " * 60
    x = tok(text, padding="max_length", max_length=length, truncation=True,
            return_tensors="pt")
    x.pop("attention_mask")
    return {k: v.to(DEV) for k, v in x.items()}


def timed_call(fn, x, tag):
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            out = fn(**x)
        _ = out.logits.cpu()  # sync
        print(f"{tag}: {(time.perf_counter() - t0) * 1000:.1f} ms", flush=True)
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"{tag}: ERROR after {elapsed:.1f} ms -> {type(e).__name__}", flush=True)
        traceback.print_exc()


SEQUENCE = [BASE, BASE, OTHER, OTHER, BASE, BASE]
print(f"sequence: {SEQUENCE}", flush=True)

with furiosa.torch.config.cpu_fallback(False):
    for name, fn in [
        ("eager", model),
        ("compile", torch.compile(model, backend=furiosa.torch.backend)),
    ]:
        print(f"\n===== {name} =====", flush=True)
        for i, L in enumerate(SEQUENCE):
            timed_call(fn, make_input(L), f"[{name}] call {i}  len={L}")
