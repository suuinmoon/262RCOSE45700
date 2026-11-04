# Usage: python step12_run_and_time.py 2>&1 | tee step12.log
import time
import traceback
 
import torch
import furiosa.torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
 
# Bypass mask creation (uses int64 ops unsupported on NPU).
# Input is fully filled with no padding, so no mask == attend to all tokens.
import transformers.models.bert.modeling_bert as _bert_mod
_bert_mod.create_bidirectional_mask = lambda *a, **k: None
 
MODEL = "textattack/bert-base-uncased-SST-2"
DEV = torch.device("furiosa:0")
WARMUP = 5
ITERS = 20
 
 
def prepare():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model = model.to(torch.bfloat16).eval()
    model = model.to(DEV)
    text = "this movie was absolutely wonderful and i loved it . " * 20
    x = tok(text, padding="max_length", max_length=128, truncation=True,
            return_tensors="pt")
    x.pop("attention_mask")
    x = {k: v.to(DEV) for k, v in x.items()}
    return model, x
 
 
def sync_run(fn, x):
    t0 = time.perf_counter()
    with torch.no_grad():
        out = fn(**x)
    _ = out.logits.cpu()  # sync
    return time.perf_counter() - t0
 
 
def bench(name, fn, x):
    first = sync_run(fn, x)
    print(f"[{name}] first call : {first * 1000:.1f} ms", flush=True)
    for _ in range(WARMUP):
        sync_run(fn, x)
    ts = [sync_run(fn, x) for _ in range(ITERS)]
    avg = sum(ts) / len(ts)
    print(
        f"[{name}] steady avg  : {avg * 1000:.2f} ms "
        f"(min {min(ts) * 1000:.2f} / max {max(ts) * 1000:.2f}, n={ITERS})",
        flush=True,
    )
 
 
def main():
    print(f"torch {torch.__version__}", flush=True)
    model, x = prepare()
    print("model & input ready on", DEV, flush=True)
 
    with furiosa.torch.config.cpu_fallback(False):
        try:
            bench("eager", model, x)
        except Exception:
            print("[eager] ERROR", flush=True)
            traceback.print_exc()
 
        try:
            compiled = torch.compile(model, backend=furiosa.torch.backend)
            bench("compile", compiled, x)
        except Exception:
            print("[compile] ERROR", flush=True)
            traceback.print_exc()
 
 
if __name__ == "__main__":
    main()
 