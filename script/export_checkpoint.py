"""Экспорт проверенных весов модели без состояния оптимизатора для публикации."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("saved/conformer_bpe_finetune_4epochs/model_best.pth"))
    parser.add_argument("--url", help="Публичная ссылка для скачивания экспортированного файла .pth")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = root / "pretrained/tokenizer.json"
    tokenizer_hash = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    if checkpoint.get("text_encoder", {}).get("model_sha256") != tokenizer_hash:
        parser.error("Чекпойнт не соответствует токенизатору pretrained/tokenizer.json")
    output = root / "pretrained/model_best.pth"
    torch.save({
        "state_dict": checkpoint["state_dict"],
        "vocab": checkpoint["vocab"],
        "text_encoder": checkpoint["text_encoder"],
        "model_config": OmegaConf.to_container(checkpoint["config"]["model"], resolve=True),
        "epoch": checkpoint["epoch"],
    }, output)
    with output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "checkpoint": {"path": "pretrained/model_best.pth", "url": args.url, "sha256": digest,
                       "source_epoch": checkpoint["epoch"]},
        "tokenizer": {"path": "pretrained/tokenizer.json", "sha256": tokenizer_hash},
    }
    (root / "pretrained/resources.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Экспортирован файл {output} ({output.stat().st_size / 1024**2:.1f} МиБ). "
          "Загрузите именно этот файл и укажите checkpoint.url в pretrained/resources.json.")


if __name__ == "__main__":
    main()
