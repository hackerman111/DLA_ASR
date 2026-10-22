# Готовые индексы LibriSpeech с Hugging Face позволяют не скачивать записи для их построения.
# Построение индексов train-clean-360 и train-other-500 на Kaggle занимало слишком много времени.
# Каждый из этих наборов занимает около 60 ГБ и не помещается на локальном накопителе.


import json
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

ROOT = Path("/kaggle/input/datasets/a24998667/librispeech")
OUT = Path("/kaggle/working/librispeech_indices")
OUT.mkdir(parents=True, exist_ok=True)

REPO = "meetween/mumospee_librispeech"

PARTS = {
    "dev-clean": 2703,
    "dev-other": 2864,
    "test-clean": 2620,
    "test-other": 2939,
    "train-clean-100": 28539,
    "train-clean-360": 104014,
    "train-other-500": 148688,
}

files = []

for part, expected_count in tqdm(PARTS.items()):
    metadata = hf_hub_download(
        repo_id=REPO,
        filename=f"metadata/{part}.parquet",
        repo_type="dataset",
    )

    table = pq.read_table(
        metadata,
        columns=["path", "transcript", "duration"],
    )

    index = []
    split_root = ROOT / part / "LibriSpeech" / part

    for row in table.to_pylist():
        filename = Path(row["path"]).name
        speaker, chapter, _ = filename.split("-", 2)
        duration = float(row["duration"])

        index.append(
            {
                "path": str(split_root / speaker / chapter / filename),
                "text": row["transcript"].lower(),
                "audio_len": duration,
            }
        )

    assert len(index) == expected_count, (
        f"{part}: ожидалось записей — {expected_count}, получено — {len(index)}"
    )

    output = OUT / f"{part}_index.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, separators=(",", ":"))

    files.append(output)
    print(f"{part}: записей — {len(index):,}")

archive = Path("/kaggle/working/librispeech_indices.zip")

with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for file in files:
        z.write(file, arcname=file.name)
