import json
import os
import re
import copy

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from transformers import pipeline


MODEL_DIR = "./Amazon-title-gen/final"
DATA_PATH = "data-for-testing.jsonl"
RESULTS_PATH = "test-results.json"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32 if device == "cpu" else torch.float16

generator = pipeline("text-generation", model=MODEL_DIR, dtype=dtype, device=device)

gen_config = copy.deepcopy(generator.model.generation_config)
gen_config.max_new_tokens = 120
gen_config.do_sample = False

with open(DATA_PATH) as f:
    rows = [json.loads(line) for line in f]


# Counts how many words of the title actually appear in the product text.
def word_match(title, source):
    source_words = set(re.findall(r"[a-z0-9]+", source.lower()))
    title_words = re.findall(r"[a-z0-9]+", title.lower())
    return sum(1 for w in title_words if w in source_words), len(title_words)


results = []

for i, row in enumerate(rows, 1):
    # Same format as training: product text, a newline, then the title.
    output = generator(
        row["input"] + "\n",
        generation_config=gen_config,
        return_full_text=False,
        clean_up_tokenization_spaces=False
    )
    title = output[0]["generated_text"].strip()
    hits, total = word_match(title, row["input"])

    print(f"\n[{i}/{len(rows)}]  match {hits}/{total}")
    print(f"  generated: {title}")
    print(f"  original : {row['output']}")

    results.append({
        "generated_title": title,
        "original_title": row["output"],
        "match": f"{hits}/{total}",
        "input": row["input"]
    })

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(results)} results to {RESULTS_PATH}")
