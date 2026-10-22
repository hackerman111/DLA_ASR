"""Скачать финальный чекпойнт и проверить SHA256 модели и токенизатора."""

import argparse
import hashlib
import json
from pathlib import Path

from demo_utils import download_file


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "pretrained/resources.json")
    parser.add_argument("--url", help="Другая ссылка на тот же файл с указанным SHA256")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tokenizer = manifest["tokenizer"]
    tokenizer_path = ROOT / tokenizer["path"]
    if not tokenizer_path.is_file() or sha256(tokenizer_path) != tokenizer["sha256"]:
        parser.error("Токенизатор отсутствует или его SHA256 не совпадает с манифестом.")
    checkpoint = manifest["checkpoint"]
    path = ROOT / checkpoint["path"]
    if path.exists():
        if sha256(path) != checkpoint["sha256"]:
            parser.error(f"SHA256 не совпадает: {path}. Переместите файл и повторите загрузку.")
        print(f"Модель и токенизатор проверены: {path}")
        return
    url = args.url or checkpoint.get("url")
    if not url:
        parser.error("Укажите публичную ссылку checkpoint.url в манифесте ресурсов.")
    temporary = path.with_name(path.name + ".download")
    try:
        print(f"Загрузка модели: {url}", flush=True)
        download_file(url, temporary)
        if sha256(temporary) != checkpoint["sha256"]:
            parser.error("SHA256 скачанной модели не совпадает с манифестом.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Модель загружена, SHA256 проверен: {path}")


if __name__ == "__main__":
    main()
