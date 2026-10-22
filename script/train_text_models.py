"""Обучение токенизатора и языковой модели"""

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from string import ascii_lowercase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRAIN_SPLITS = ("train-clean-100", "train-clean-360", "train-other-500")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_logged(command, log_path):
    with Path(log_path).open("w") as stream:
        result = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode:
        detail = Path(log_path).read_text()[-4000:]
        raise ValueError(
            f"Ошибка выполнения {command[0]} (код завершения {result.returncode}):\n{detail}"
        )


def prepare_corpus(indices, directory):
    from src.text_encoder.ctc_text_encoder import CTCTextEncoder

    indices = [Path(path).resolve() for path in indices]
    allowed = {f"{split}_index.json": split for split in TRAIN_SPLITS}
    if not indices or any(path.name not in allowed for path in indices):
        raise ValueError(
            "Укажите только индексы train-clean-100, train-clean-360 или train-other-500"
        )
    directory = Path(directory)
    corpus, uppercase = directory / "corpus.txt", directory / "corpus_upper.txt"
    seen, splits = set(), []
    with (
        corpus.open("w", encoding="ascii") as lower,
        uppercase.open("w", encoding="ascii") as upper,
    ):
        for path in indices:
            raw = path.read_bytes()
            rows = json.loads(raw)
            split = allowed[path.name]
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"Индекс пуст или имеет неверный формат: {path}")
            words = 0
            for row in rows:
                if (
                    not isinstance(row, dict)
                    or not isinstance(row.get("text"), str)
                    or not isinstance(row.get("path"), str)
                ):
                    raise TypeError(f"Поля text и path должны быть строками: {path}")
                audio_path = Path(row["path"])
                uid = audio_path.stem
                if split not in audio_path.parts or not re.fullmatch(
                    r"\d+-\d+-\d+", uid
                ):
                    raise ValueError(
                        f"Путь к записи не соответствует набору данных в {path}: {audio_path}"
                    )
                if uid in seen:
                    raise ValueError(f"Повторяющийся идентификатор записи: {uid}")
                seen.add(uid)
                text = CTCTextEncoder.normalize_text(row["text"]).strip()
                if not text:
                    raise ValueError(f"Транскрипция пуста после нормализации: {uid}")
                lower.write(text + "\n")
                upper.write(text.upper() + "\n")
                words += len(text.split())
            splits.append(
                {
                    "name": split,
                    "index": str(path),
                    "utterances": len(rows),
                    "words": words,
                    "index_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    return {
        "splits": splits,
        "utterances": len(seen),
        "words": sum(s["words"] for s in splits),
        "dev_data_used": False,
        "test_data_used": False,
        "normalization": "CTCTextEncoder.normalize_text(text).strip()",
        "corpus_sha256": sha256(corpus),
        "uppercase_corpus_sha256": sha256(uppercase),
    }


def train_tokenizer(corpus, destination, vocab_size=512):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    if vocab_size < 28:
        raise ValueError(
            "Размер словаря должен вмещать 26 букв, разделитель слов и токен <unk>"
        )
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁", prepend_scheme="always"
    )
    tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    tokenizer.train(
        [str(corpus)],
        trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["<unk>"],
            initial_alphabet=list(ascii_lowercase),
            show_progress=False,
        ),
    )
    count = 0
    with Path(corpus).open() as stream:
        for line in stream:
            text = line.rstrip("\n")
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if tokenizer.decode(ids, skip_special_tokens=False) != text:
                raise ValueError(
                    f"Токенизатор не восстановил исходный текст в строке {count + 1}"
                )
            count += 1
    if not count:
        raise ValueError("Нельзя обучить токенизатор на пустом корпусе")
    tokenizer.save(str(destination))
    return {
        "implementation": "huggingface_tokenizers",
        "requested_vocab_size": vocab_size,
        "pieces": tokenizer.get_vocab_size(),
        "acoustic_vocabulary": tokenizer.get_vocab_size() + 1,
        "blank_id": 0,
        "tokenizer_id_offset": 1,
        "min_frequency": 2,
        "initial_alphabet": ascii_lowercase,
        "special_tokens": ["<unk>"],
        "metaspace": "▁",
        "prepend_scheme": "always",
        "roundtrip_verified_lines": count,
        "version": importlib.metadata.version("tokenizers"),
    }


def train_models(args):
    import kenlm

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Папка уже существует, перезапись запрещена: {output}")
    executables = {}
    for name in ("lmplz", "build_binary"):
        executable = shutil.which(str(getattr(args, name)))
        if executable is None:
            raise FileNotFoundError(
                f"Не найдена утилита {name}; укажите --{name.replace('_', '-')} /путь/к/{name}"
            )
        executables[name] = str(Path(executable).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".text-models-", dir=output.parent) as tmp:
        work = Path(tmp)
        manifest = prepare_corpus(args.indices, work)
        tokenizer_path = work / f"train_bpe_{args.vocab_size}.json"
        manifest["tokenizer"] = train_tokenizer(
            work / "corpus.txt", tokenizer_path, args.vocab_size
        )
        arpa, binary = work / "model.arpa", work / "model.binary"
        with tempfile.TemporaryDirectory(prefix="kenlm-", dir=args.temp_dir) as scratch:
            command = [
                executables["lmplz"],
                "-o",
                str(args.order),
                "-S",
                args.memory,
                "-T",
                scratch,
                "--text",
                str(work / "corpus_upper.txt"),
                "--arpa",
                str(arpa),
            ]
            run_logged(command, work / "train.log")
        binary_command = [
            executables["build_binary"],
            "probing",
            str(arpa),
            str(binary),
        ]
        run_logged(binary_command, work / "build_binary.log")
        model, compiled = kenlm.Model(str(arpa)), kenlm.Model(str(binary))
        if model.order != args.order or compiled.order != args.order:
            raise ValueError("Порядок модели KenLM не совпадает с запрошенным")
        max_difference = 0.0
        with (work / "corpus_upper.txt").open() as stream:
            for text in stream:
                a, b = model.score(text), compiled.score(text)
                if not (float("-inf") < a < float("inf") and abs(a - b) < 1e-5):
                    raise ValueError(
                        "Оценки ARPA и бинарной модели не совпадают или не являются конечными числами"
                    )
                max_difference = max(max_difference, abs(a - b))
        manifest.update(
            status="complete",
            created_utc=datetime.now(timezone.utc).isoformat(),
            order=args.order,
            uppercase=True,
            kenlm_python_version=importlib.metadata.version("kenlm"),
            commands=[command, binary_command],
            argv=sys.argv,
            script_sha256=sha256(__file__),
            tools={
                name: {"path": path, "sha256": sha256(path)}
                for name, path in executables.items()
            },
            max_arpa_binary_score_difference=max_difference,
            artifacts={
                p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                for p in (tokenizer_path, arpa, binary)
            },
        )
        (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        work.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indices",
        type=Path,
        nargs="+",
        default=[
            ROOT / "datasets/librispeech_indices" / f"{split}_index.json"
            for split in TRAIN_SPLITS
        ],
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/text_models/train960"
    )
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--order", type=int, choices=range(2, 7), default=3)
    parser.add_argument("--memory", default="2G")
    parser.add_argument(
        "--temp-dir", type=Path, help="Каталог временных файлов KenLM (по умолчанию: системный временный каталог)"
    )
    parser.add_argument("--lmplz", default="lmplz")
    parser.add_argument("--build-binary", default="build_binary")
    args = parser.parse_args()
    try:
        manifest = train_models(args)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Ошибка обучения: {exc}\n")
    print(
        json.dumps(
            {
                "output": str(args.output_dir.resolve()),
                "utterances": manifest["utterances"],
                "words": manifest["words"],
                "tokenizer": manifest["tokenizer"],
                "order": args.order,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
