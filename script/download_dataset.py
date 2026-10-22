"""Скриптик для развертывания датасета"""

import argparse
import tempfile
from pathlib import Path

from demo_utils import download_file, extract_archive, find_dataset_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root-file", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Каталог --output-dir должен быть пустым.")
    with tempfile.TemporaryDirectory(prefix="asr-dataset-") as temporary:
        archive = download_file(args.url, Path(temporary) / "dataset.archive")
        extract_archive(archive, output)
    root = find_dataset_root(output)
    if args.root_file:
        args.root_file.parent.mkdir(parents=True, exist_ok=True)
        args.root_file.write_text(str(root), encoding="utf-8")
    print(f"Каталог датасета: {root}")


if __name__ == "__main__":
    main()
