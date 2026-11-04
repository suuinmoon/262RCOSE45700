# 4단계(시간 남으면): eager vs compile 출력 비교
# 사용: python step4_compare.py 2>&1 | tee step4.log
import torch
import furiosa.torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "textattack/bert-base-uncased-SST-2"
DEV = torch.device("furiosa:0")

# SST-2 스타일 영화 리뷰 문장 16개 (긍/부정 섞음)
SENTENCES = [
    "a stirring , funny and finally transporting re-imagining of a classic story",
    "the film is strictly routine and never surprises",
    "one of the best performances of the year , hands down",
    "it 's so laddish and juvenile , only teenage boys could possibly find it funny",
    "a masterpiece of quiet , devastating storytelling",
    "the plot is paper thin and the dialogue is worse",
    "an absolute delight from start to finish",
    "painfully slow and utterly forgettable",
    "the acting elevates an otherwise ordinary script",
    "i wanted to walk out after twenty minutes",
    "visually stunning with a soundtrack to match",
    "a bland , lifeless retread of better films",
    "heartfelt , sincere , and genuinely moving",
    "the jokes land with a thud every single time",
    "an ambitious film that mostly succeeds",
    "a complete waste of a talented cast",
]

tok = AutoTokenizer.from_pretrained(MODEL)
model = (
    AutoModelForSequenceClassification.from_pretrained(MODEL)
    .to(torch.bfloat16).eval().to(DEV)
)
compiled = torch.compile(model, backend=furiosa.torch.backend)

max_diff = 0.0
match = 0
with furiosa.torch.config.cpu_fallback(False), torch.no_grad():
    for s in SENTENCES:
        x = tok(s, padding="max_length", max_length=128, truncation=True,
                return_tensors="pt")
        x = {k: v.to(DEV) for k, v in x.items()}
        le = model(**x).logits.float().cpu()
        lc = compiled(**x).logits.float().cpu()
        d = (le - lc).abs().max().item()
        max_diff = max(max_diff, d)
        same = le.argmax(-1).item() == lc.argmax(-1).item()
        match += int(same)
        print(f"diff={d:.6f}  label_match={same}  | {s[:40]}", flush=True)

print(f"\nmax |logit diff| = {max_diff:.6f}")
print(f"label match      = {match}/{len(SENTENCES)}")
