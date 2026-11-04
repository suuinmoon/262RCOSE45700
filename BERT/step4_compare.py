# Usage: python step4_compare.py 2>&1 | tee step4.log
# Compare eager vs compile outputs on padded (len 128) SST-2 style sentences.
# CPU fp32 model is used as a reference to check the mask handling is correct.
import traceback
 
import torch
import furiosa.torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers.models.bert.modeling_bert as _bert_mod
 
MODEL = "textattack/bert-base-uncased-SST-2"
DEV = torch.device("furiosa:0")
SEQ_LEN = 128
 
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
N = len(SENTENCES)
 
tok = AutoTokenizer.from_pretrained(MODEL)
 
 
def encode(s):
    return tok(s, padding="max_length", max_length=SEQ_LEN, truncation=True,
               return_tensors="pt")
 
 
# 1) CPU fp32 reference with the original (unpatched) mask path
cpu_model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
ref = []
with torch.no_grad():
    for s in SENTENCES:
        ref.append(cpu_model(**encode(s)).logits.float())
print("cpu fp32 reference done", flush=True)
 
 
# 2) Mask creation ops are unsupported on NPU, so build the 4D additive mask
#    on CPU and make the model use it as-is.
def passthrough_mask(*args, **kwargs):
    return kwargs["attention_mask"] if "attention_mask" in kwargs else args[2]
 
 
_bert_mod.create_bidirectional_mask = passthrough_mask
 
 
def additive_mask(mask2d, dtype):
    neg = torch.finfo(dtype).min
    m = (1.0 - mask2d[:, None, None, :].to(torch.float32)) * neg  # [1,1,1,L]
    return m.expand(-1, 1, SEQ_LEN, -1).contiguous().to(dtype)    # [1,1,L,L]
 
 
def npu_inputs(s):
    x = encode(s)
    return {
        "input_ids": x["input_ids"].to(DEV),
        "token_type_ids": x["token_type_ids"].to(DEV),
        "attention_mask": additive_mask(x["attention_mask"], torch.bfloat16).to(DEV),
    }
 
 
model = (
    AutoModelForSequenceClassification.from_pretrained(MODEL)
    .to(torch.bfloat16).eval().to(DEV)
)
 
 
def run(name, fn):
    outs = []
    try:
        with furiosa.torch.config.cpu_fallback(False), torch.no_grad():
            for i, s in enumerate(SENTENCES):
                outs.append(fn(**npu_inputs(s)).logits.float().cpu())
                print(f"[{name}] {i + 1}/{N} done", flush=True)
        return outs
    except Exception:
        print(f"[{name}] ERROR at sentence {len(outs) + 1}", flush=True)
        traceback.print_exc()
        return None
 
 
eager = run("eager", model)
compiled = run("compile", torch.compile(model, backend=furiosa.torch.backend))
 
 
def compare(a_name, a, b_name, b):
    if a is None or b is None:
        print(f"{a_name} vs {b_name}: skipped (missing results)")
        return
    diff = max((x - y).abs().max().item() for x, y in zip(a, b))
    match = sum(x.argmax(-1).item() == y.argmax(-1).item() for x, y in zip(a, b))
    print(f"{a_name} vs {b_name}: max |logit diff| = {diff:.4f}, label match = {match}/{N}")
 
 
print("\n=== per sentence (label: 0=negative, 1=positive) ===")
for i, s in enumerate(SENTENCES):
    c = ref[i].argmax(-1).item()
    e = eager[i].argmax(-1).item() if eager else "-"
    p = compiled[i].argmax(-1).item() if compiled else "-"
    print(f"{i + 1:2d}  cpu={c}  eager={e}  compile={p}  | {s[:45]}")
 
print("\n=== summary ===")
compare("eager", eager, "compile", compiled)
compare("eager", eager, "cpu_fp32", ref)
compare("compile", compiled, "cpu_fp32", ref)
 