# 1단계(실행 확인) + 2단계(시간 측정) 통합
# 사용: python step12_run_and_time.py 2>&1 | tee step12.log
import time
import traceback
 
import torch
import furiosa.torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
 
# transformers의 attention mask 생성이 NPU 미지원 int64 연산(arange/add 등)을 타므로 우회.
# 입력을 패딩 없이 128토큰으로 꽉 채우므로 마스크 없음 = 전체 attend = 수학적으로 동일.
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
    # 패딩이 생기지 않도록 긴 문장을 128토큰으로 잘라 꽉 채움
    text = "this movie was absolutely wonderful and i loved it . " * 20
    x = tok(text, padding="max_length", max_length=128, truncation=True,
            return_tensors="pt")
    x.pop("attention_mask")  # 마스크 생성 경로 회피 (위 패치와 세트)
    # 토큰 ID는 정수 그대로, dtype 변환 없이 장치로만 이동
    x = {k: v.to(DEV) for k, v in x.items()}
    return model, x
 
 
def sync_run(fn, x):
    """호출 시작부터 logits를 CPU로 가져올 때까지(= NPU 연산 완료까지) 측정"""
    t0 = time.perf_counter()
    with torch.no_grad():
        out = fn(**x)
    _ = out.logits.cpu()  # 동기화 지점
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
            print("[eager] ERROR — 아래 메시지도 보고서용 결과임", flush=True)
            traceback.print_exc()
 
        try:
            compiled = torch.compile(model, backend=furiosa.torch.backend)
            bench("compile", compiled, x)
        except Exception:
            print("[compile] ERROR — 아래 메시지도 보고서용 결과임", flush=True)
            traceback.print_exc()
 
 
if __name__ == "__main__":
    main()
 