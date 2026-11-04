# Usage: python embedding_length_test.py 2>&1 | tee embedding_length_test.log
# Isolate the BERT word-embedding lookup and check which sequence lengths
# compile on RNGD (eager, no CPU fallback).
import time

import torch
import furiosa.torch
from transformers import AutoModelForSequenceClassification

MODEL = "textattack/bert-base-uncased-SST-2"
DEV = torch.device("furiosa:0")
LENGTHS = [16, 32, 48, 64, 96, 128, 192, 256, 512]

model = AutoModelForSequenceClassification.from_pretrained(MODEL)
emb = model.bert.embeddings.word_embeddings.to(torch.bfloat16).eval().to(DEV)
vocab = emb.num_embeddings

results = []
with furiosa.torch.config.cpu_fallback(False), torch.no_grad():
    for L in LENGTHS:
        ids = torch.randint(0, vocab, (1, L)).to(DEV)
        t0 = time.perf_counter()
        try:
            out = emb(ids)
            _ = out.cpu()  # sync
            ms = (time.perf_counter() - t0) * 1000
            results.append((L, "OK", f"{ms:.1f} ms"))
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            results.append((L, "FAIL", f"{type(e).__name__} after {ms:.1f} ms"))
        print(f"len={L:4d}  {results[-1][1]:4s}  {results[-1][2]}", flush=True)

print("\n=== summary ===")
for L, status, info in results:
    print(f"{L:4d}  {status}")
