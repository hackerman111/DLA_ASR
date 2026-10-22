"""Развёртывание clean360 BPE на Linux-сервере с NVIDIA GPU."""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)
SPLITS = {
    "train-clean-360": (104014, "c0e676e450a7ff2f54aeade5171606fa"),
    "dev-clean": (2703, "42e2234ba48799c1f50f24a7926300a1"),
}


def run(command, **kwargs):
    subprocess.run([str(value) for value in command], check=True, **kwargs)


def file_md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prerequisites():
    if sys.platform != "linux" or sys.version_info < (3, 10):
        raise ValueError("Нужен Linux и Python версии 3.10 или новее")
    missing = [
        name
        for name in ("curl", "tar", "g++", "git", "nvidia-smi")
        if shutil.which(name) is None
    ]
    if missing:
        raise ValueError(
            "Не найдены программы: "
            + ", ".join(missing)
            + ". Для Ubuntu/Debian: sudo apt-get install python3-venv "
            "python3-dev build-essential curl tar git. Также нужен драйвер NVIDIA."
        )
    run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"])


def ensure_pip(interpreter):
    result = subprocess.run(
        [str(interpreter), "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode == 0:
        return
    LOGGER.info("Восстанавливаю pip в %s", interpreter.parent.parent)
    try:
        run([interpreter, "-m", "ensurepip", "--upgrade"])
    except subprocess.CalledProcessError:
        try:
            run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "--python",
                    interpreter,
                    "install",
                    "--upgrade",
                    "pip",
                ]
            )
        except subprocess.CalledProcessError as error:
            raise ValueError(
                "Не удалось восстановить pip. Установите python3-venv и python3-pip "
                "для используемой версии Python и повторите запуск."
            ) from error


def bootstrap(args, forwarded):
    LOGGER.info("Этап 1/4: подготовка Python-окружения")
    interpreter = args.venv_dir / "bin/python"
    if not interpreter.exists():
        run([sys.executable, "-m", "venv", "--without-pip", args.venv_dir])
    ensure_pip(interpreter)
    pip = [interpreter, "-m", "pip"]
    run([*pip, "install", "--upgrade", "pip", "setuptools", "wheel", "cmake<4"])
    LOGGER.info("Этап 2/4: установка PyTorch с CUDA 12.8 и проверка GPU")
    run(
        [
            *pip,
            "install",
            "torch==2.10.0+cu128",
            "torchaudio==2.10.0+cu128",
            "torchvision==0.25.0+cu128",
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
        ]
    )
    run(
        [
            interpreter,
            "-c",
            (
                "import torch; "
                "assert torch.cuda.is_available(), "
                "'CUDA недоступна: проверьте драйвер NVIDIA и CUDA-сборку PyTorch'; "
                "x = torch.ones(4, device='cuda:0'); "
                "assert (x @ x).item() == 4; "
                "layer = torch.nn.Conv1d(2, 4, 3).cuda(); "
                "layer(torch.ones(1, 2, 16, device='cuda:0')).sum().backward(); "
                "torch.cuda.synchronize(); "
                "print('GPU:', torch.cuda.get_device_name(0), 'PyTorch:', torch.__version__)"
            ),
        ]
    )
    run([*pip, "install", "-r", ROOT / "requirements.txt"])
    run([*pip, "check"])
    if args.wandb_mode == "online":
        LOGGER.info("Вход в W&B: при необходимости введите API-ключ в терминале")
        run([interpreter, "-c", "import wandb; wandb.login()"])
    run([interpreter, __file__, *forwarded, "--stage", "train"], cwd=ROOT)


def prepare_split(data_dir, split):
    """Распаковываем в отдельный каталог и публикуем только целый набор."""
    expected_count, expected_md5 = SPLITS[split]
    destination = data_dir / split
    LOGGER.info("Подготовка %s", split)
    if not destination.exists():
        archive = data_dir / f"{split}.tar.gz"
        if not archive.is_file() or file_md5(archive) != expected_md5:
            LOGGER.info("Скачивание %s с докачкой", archive.name)
            run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "5",
                    "--continue-at",
                    "-",
                    "--output",
                    archive,
                    f"https://www.openslr.org/resources/12/{split}.tar.gz",
                ]
            )
        LOGGER.info("Проверка контрольной суммы %s", archive.name)
        if file_md5(archive) != expected_md5:
            raise ValueError(
                f"Контрольная сумма архива не совпала: {archive}. "
                "Удалите повреждённый архив и повторите запуск."
            )
        with tempfile.TemporaryDirectory(
            prefix=f".{split}.", dir=data_dir
        ) as temporary:
            staging = Path(temporary)
            LOGGER.info("Распаковка %s", split)
            run(["tar", "-xzf", archive, "-C", staging, "--strip-components=1"])
            (staging / split).rename(destination)
        archive.unlink()

    index_path = data_dir / f"{split}_index.json"
    if index_path.is_file():
        rows = json.loads(index_path.read_text())
        if len(rows) == expected_count and all(
            Path(row["path"]).is_file()
            and Path(row["path"]).is_relative_to(destination)
            for row in rows
        ):
            LOGGER.info("Использую готовый индекс: %s записей", len(rows))
            return rows
    from src.datasets import LibrispeechDataset

    dataset = object.__new__(LibrispeechDataset)
    dataset._data_dir = data_dir
    LOGGER.info("Создание локального индекса %s", split)
    rows = dataset._create_index(split)
    if len(rows) != expected_count:
        raise ValueError(
            f"Набор {split} неполный: найдено {len(rows)}, ожидалось {expected_count}. "
            f"Проверьте каталог {destination}."
        )
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    temporary.replace(index_path)
    return rows


def training_command(args):
    return [
        sys.executable,
        str(ROOT / "train.py"),
        "-cn",
        "clean100_bpe_aug10",
        "datasets.train.part=train-clean-360",
        f"data_dir={args.data_dir}",
        f"text_encoder.tokenizer_path={args.tokenizer_path}",
        f"dataloader.batch_size={args.batch_size}",
        f"dataloader.num_workers={args.num_workers}",
        "trainer.device=cuda:0",
        f"trainer.save_dir={ROOT / 'saved'}",
        f"writer.run_name={args.run_name}",
        "writer.project_name=dla_asr_clean360_aug10",
        f"writer.mode={args.wandb_mode}",
    ]


def prepare_and_train(args):
    import torch
    from tokenizers import Tokenizer

    if not torch.cuda.is_available():
        raise ValueError("CUDA недоступна после установки зависимостей")
    LOGGER.info("Этап 3/4: подготовка данных и токенизатора")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    train_rows = prepare_split(args.data_dir, "train-clean-360")
    prepare_split(args.data_dir, "dev-clean")
    if not args.tokenizer_path.is_file():
        LOGGER.info("Обучение BPE на транскрипциях train-clean-360")
        from script.train_text_models import train_tokenizer
        from src.text_encoder.bpe import BPE

        args.tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        corpus = args.tokenizer_path.with_suffix(".corpus.txt")
        corpus.write_text(
            "".join(
                BPE.normalize_text(row["text"]).strip() + "\n" for row in train_rows
            ),
            encoding="utf-8",
        )
        temporary = args.tokenizer_path.with_suffix(".json.tmp")
        train_tokenizer(corpus, temporary, vocab_size=512)
        temporary.replace(args.tokenizer_path)
        print("Токенизатор обучен только на train-clean-360", flush=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    print(
        f"Токенизатор: {args.tokenizer_path}, токенов: {tokenizer.get_vocab_size()}",
        flush=True,
    )
    LOGGER.info("Этап 4/4: обучение, батч %s, W&B %s", args.batch_size, args.wandb_mode)
    LOGGER.info("Чекпоинты: %s", ROOT / "saved" / args.run_name)
    run(training_command(args), cwd=ROOT)


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=7)
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data/server_librispeech"
    )
    parser.add_argument(
        "--tokenizer-path", type=Path, default=ROOT / "data/bpe_hf/train_bpe_512.json"
    )
    parser.add_argument("--venv-dir", type=Path, default=ROOT / ".venv-server")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--foreground", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-name")
    parser.add_argument("--stage", choices=("setup", "train"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 1:
        raise ValueError("Размер батча и число загрузчиков должны быть положительными")
    if args.run_name is None:
        args.run_name = (
            f"clean360_bpe_aug10_bs{args.batch_size}_"
            + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_")
            + uuid.uuid4().hex[:6]
        )
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("Название запуска не должно содержать путь")
    for name in ("data_dir", "tokenizer_path", "venv_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/matplotlib"))
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ.pop("WANDB_SILENT", None)
    os.environ.pop("WANDB_QUIET", None)
    forwarded = [
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--data-dir",
        str(args.data_dir),
        "--tokenizer-path",
        str(args.tokenizer_path),
        "--venv-dir",
        str(args.venv_dir),
        "--wandb-mode",
        args.wandb_mode,
        "--run-name",
        args.run_name,
    ]
    if args.stage == "train":
        prepare_and_train(args)
        return
    prerequisites()
    bootstrap(args, forwarded)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        sys.exit(1)
