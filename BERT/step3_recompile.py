# 3단계: 재컴파일/그래프 브레이크 관찰
# 사용: TORCH_LOGS="recompiles,graph_breaks" python step3_recompile.py 2>&1 | tee step3.log
#  - 재컴파일 로그는 stderr로 나오므로 2>&1 필수
import time

import torch
import furiosa.torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "textattack/bert-base-uncased-SST-2"
DEV = torch.device("furiosa:0")

tok = AutoTokenizer.from_pretrained(MODEL)
model = (
    AutoModelForSequenceClassification.from_pretrained(MODEL)
    .to(torch.bfloat16).eval().to(DEV)
)


def make_input(length):
    x = tok(
        "a short sentence for the recompilation check",
        padding="max_length", max_length=length, truncation=True,
        return_tensors="pt",
    )
    return {k: v.to(DEV) for k, v in x.items()}


def timed_call(fn, x, tag):
    t0 = time.perf_counter()
    with torch.no_grad():
        out = fn(**x)
    _ = out.logits.cpu()
    print(f"{tag}: {(time.perf_counter() - t0) * 1000:.1f} ms", flush=True)


# 각 길이를 2번씩 호출: 1번째 = (재)컴파일 여부, 2번째 = 안정 상태
SEQUENCE = [128, 128, 64, 64, 128, 128]

with furiosa.torch.config.cpu_fallback(False):
    for name, fn in [
        ("eager", model),
        ("compile", torch.compile(model, backend=furiosa.torch.backend)),
    ]:
        print(f"\n===== {name} =====", flush=True)
        for i, L in enumerate(SEQUENCE):
            timed_call(fn, make_input(L), f"[{name}] call {i}  len={L}")
