"""Download WikiText-2 (raw) from the Hugging Face hub and write wikitext2_train.txt / wikitext2_test.txt
(paragraphs joined by blank lines, the usual GPTQ convention). Used for calibration (train) and evaluation (test)."""
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
for split in ("train", "test"):
    f = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet", repo_type="dataset")
    text = "\n\n".join(pq.read_table(f).column("text").to_pylist())
    open(f"wikitext2_{split}.txt", "w").write(text); print(f"wikitext2_{split}.txt: {len(text):,} characters")
