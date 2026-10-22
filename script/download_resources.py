"""Скачать финальную модель, BPE и LM с Hugging Face и проверить SHA256."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import zipfile

if __package__:
    from .demo_utils import download_file
else:
    from demo_utils import download_file


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_resources(manifest_path=ROOT / "pretrained/resources.json", checkpoint_url=None):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    for name in ("tokenizer", "checkpoint", "lm"):
        if name not in manifest:
            continue
        resource = manifest[name]
        path = ROOT / resource["path"]
        if path.is_file():
            if sha256(path) != resource["sha256"]:
                raise ValueError(f"SHA256 не совпадает: {path}. Переместите файл и повторите загрузку.")
            print(f"Проверено: {path}", flush=True)
            continue
        url = checkpoint_url if name == "checkpoint" and checkpoint_url else resource.get("url")
        if not url:
            raise ValueError(f"Не задана ссылка для ресурса {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".download")
        extracted = path.with_name(path.name + ".download.extracted")
        try:
            print(f"Загрузка {name}: {url}", flush=True)
            download_file(url, temporary)
            ready = temporary
            if "archive_member" in resource:
                with zipfile.ZipFile(temporary) as archive:
                    try:
                        member = archive.getinfo(resource["archive_member"])
                    except KeyError as error:
                        raise ValueError(f"В архиве отсутствует {resource['archive_member']}") from error
                    with archive.open(member) as source, extracted.open("wb") as target:
                        shutil.copyfileobj(source, target)
                ready = extracted
            if sha256(ready) != resource["sha256"]:
                raise ValueError(f"SHA256 скачанного ресурса {name} не совпадает с манифестом")
            ready.replace(path)
            print(f"Загружено и проверено: {path}", flush=True)
        finally:
            temporary.unlink(missing_ok=True)
            extracted.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "pretrained/resources.json")
    parser.add_argument("--url", help="Другая ссылка на тот же файл с указанным SHA256")
    args = parser.parse_args()
    try:
        download_resources(args.manifest, args.url)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
