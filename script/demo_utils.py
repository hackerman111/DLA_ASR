import shutil
import stat
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


def download_file(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        host = urllib.parse.urlparse(url).hostname or ""
        if host in {"drive.google.com", "docs.google.com"}:
            import gdown

            if not gdown.download(url=url, output=str(temporary), fuzzy=True):
                raise RuntimeError(
                    "Не удалось скачать файл Google Drive; проверьте общий доступ."
                )
        else:
            request = urllib.request.Request(
                url, headers={"User-Agent": "ASR-demo/1.0"}
            )
            with (
                urllib.request.urlopen(request, timeout=120) as source,
                temporary.open("wb") as target,
            ):
                shutil.copyfileobj(source, target)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def extract_archive(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for item in bundle.infolist():
                target = (destination / item.filename).resolve()
                if not target.is_relative_to(destination) or stat.S_ISLNK(
                    item.external_attr >> 16
                ):
                    raise ValueError(f"Недопустимый путь в архиве: {item.filename}")
            bundle.extractall(destination)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as bundle:
            bundle.extractall(destination, filter="data")
    else:
        raise ValueError(
            "Ожидался архив ZIP, TAR или TAR.GZ; проверьте ссылку на файл."
        )


def find_dataset_root(directory):
    candidates = [
        p.parent
        for p in Path(directory).rglob("audio")
        if p.is_dir() and "__MACOSX" not in p.parts
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"В архиве должна быть ровно одна папка audio; найдено: {len(candidates)}"
        )
    return candidates[0]
