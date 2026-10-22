import json
import shlex
import string
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

from packaging.requirements import Requirement

TRAIN_DIR = Path(
    "/kaggle/input/datasets/a24998667/librispeech/train-other-500/LibriSpeech/train-other-500"
)
VAL_DIR = Path(
    "/kaggle/input/datasets/a24998667/librispeech/dev-other/LibriSpeech/dev-other"
)
PREPARED_DIR = Path("/kaggle/working/asr_other")
VOCAB_SIZE = 512
INDEX_WORKERS = 16
INSTALL_DEPENDENCIES = True

ROOT = Path(__file__).resolve().parent


def install_dependencies():
    preserved = ("torch", "torchvision", "torchaudio", "numpy")
    constraints = PREPARED_DIR / "constraints.txt"
    constraints.write_text("".join(f"{name}=={version(name)}\n" for name in preserved))
    skip = set(preserved) | {"black", "isort", "pre-commit"}
    packages = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and Requirement(line.strip()).name not in skip
    ]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-c",
            str(constraints),
            *packages,
            "wget",
        ],
        check=True,
    )
    print("-" * 50)
    print("Установка зависимостей завершена")
    print("-" * 50, flush=True)


def prepare_dataset(source, data_dir):
    from src.datasets import LibrispeechDataset

    link = data_dir / source.name
    index_path = data_dir / f"{source.name}_index.json"
    if link.is_symlink() and link.resolve() != source and not index_path.exists():
        link.unlink()
    if link.is_symlink() and link.resolve() == source:
        pass
    elif link.exists() or link.is_symlink():
        raise FileExistsError(f"{link} уже занят. Выберите другой PREPARED_DIR.")
    else:
        link.symlink_to(source, target_is_directory=True)

    print(f"Подготовка LibrispeechDataset: {source.name}", flush=True)
    dataset = LibrispeechDataset(source.name, data_dir=data_dir)
    if not len(dataset):
        raise ValueError(f"Нет записей LibriSpeech в {source}")
    print(f"{source.name}: {len(dataset)} записей", flush=True)
    return dataset


def main():
    print("Подготовка LibriSpeech", flush=True)
    sources = [TRAIN_DIR.expanduser().resolve(), VAL_DIR.expanduser().resolve()]
    for source, prefix in zip(sources, ("train-", "dev-")):
        if not source.is_dir():
            raise FileNotFoundError(f"Не найден каталог сплита: {source}")
        if not source.name.startswith(prefix):
            raise ValueError(f"Укажите сам каталог {prefix}*: {source}")

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    if INSTALL_DEPENDENCIES:
        install_dependencies()

    print("Импорт библиотек...", flush=True)
    from omegaconf import OmegaConf
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from tqdm.auto import tqdm

    from src.datasets.librispeech_dataset import URL_LINKS
    from src.text_encoder import BPE

    data_dir = PREPARED_DIR / "datasets"
    data_dir.mkdir(exist_ok=True)
    for source in sources:
        if source.name not in URL_LINKS:
            raise ValueError(f"Неизвестный сплит LibriSpeech: {source.name}")
    with ThreadPoolExecutor(max_workers=min(INDEX_WORKERS, len(sources))) as pool:
        datasets = list(
            tqdm(
                pool.map(lambda source: prepare_dataset(source, data_dir), sources),
                total=len(sources),
                desc="Подготовка сплитов",
                unit="сплит",
            )
        )
    indexes = [dataset._index for dataset in datasets]

    tokenizer_path = PREPARED_DIR / "tokenizer.json"
    metadata_path = PREPARED_DIR / "tokenizer_meta.json"
    metadata = {"train_source": str(sources[0]), "vocab_size": VOCAB_SIZE}
    if (
        tokenizer_path.exists()
        and metadata_path.exists()
        and json.loads(metadata_path.read_text()) == metadata
    ):
        print("Используется готовый токенайзер", flush=True)
    else:
        print(f"Обучение BPE на {len(indexes[0])} транскрипциях...", flush=True)
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
        tokenizer.decoder = decoders.Metaspace()
        tokenizer.train_from_iterator(
            (
                BPE.normalize_text(row["text"])
                for row in tqdm(indexes[0], desc="Тексты для BPE", unit="текст")
            ),
            trainer=trainers.BpeTrainer(
                vocab_size=VOCAB_SIZE,
                min_frequency=2,
                special_tokens=["<unk>"],
                initial_alphabet=list(string.ascii_lowercase),
            ),
            length=len(indexes[0]),
        )
        tokenizer.save(str(tokenizer_path))
        metadata_path.write_text(json.dumps(metadata))
    print("Токенайзер готов", flush=True)

    config = OmegaConf.create(
        {
            "defaults": ["kaggle_other500", "_self_"],
            "prepared_dir": str(PREPARED_DIR.resolve()),
            "datasets": {
                "train": {"part": sources[0].name},
                "val": {"part": sources[1].name},
            },
            "writer": {
                "project_name": "dla_asr",
                "run_name": f"conformer_{sources[0].name}_${{now:%Y%m%d_%H%M%S}}",
            },
            "hydra": {"job": {"env_set": {"WANDB_MODE": "${writer.mode}"}}},
        }
    )
    OmegaConf.save(config, ROOT / "src/configs/kaggle.yaml")
    print(
        f"\nГотово. Для обучения:\ncd {shlex.quote(str(ROOT))}\npython train.py -cn kaggle"
    )


if __name__ == "__main__":
    main()
