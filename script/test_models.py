"""Сравнение чекпойнтов CTC/BPE с Hugging Face на записях из audio/ и необязательных эталонных текстах из transcriptions/."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", type=Path, nargs="?", default=root / "samples/custom_dir",
        help="Каталог с audio/ и необязательной папкой transcriptions/ (по умолчанию: встроенный пример)",
    )
    parser.add_argument(
        "--model", choices=["both", "ctc", "bpe", "bpe_finetune_4epochs"], default="both",
        help="Значение both сравнивает ctc (без BPE) и bpe (по умолчанию: both)",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "predictions/hf_models")
    parser.add_argument("--device", default="auto", help="Устройство: auto, cpu, cuda или cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    dataset = args.dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    models = ["ctc", "bpe"] if args.model == "both" else [args.model]

    for name in models:
        folder = f"conformer_{name}"
        print(f"\n=== {folder} ===", flush=True)
        files = {
            filename: hf_hub_download(
                repo_id="papayka/ASR", filename=f"{folder}/{filename}",
                local_dir=root / "data/hf_models",
            )
            for filename in ("config.yaml", "model_best.pth")
        }
        config = OmegaConf.load(files["config.yaml"])
        command = [
            sys.executable, str(root / "inference.py"),
            f"custom_dir={json.dumps(str(dataset))}",
            f"inferencer.save_path={json.dumps(str(output / name))}",
            f"inferencer.from_pretrained={json.dumps(files['model_best.pth'])}",
            f"inferencer.device={args.device}",
            f"dataloader.batch_size={args.batch_size}",
            "inferencer.decoding=greedy",
            f"text_encoder._target_={config.text_encoder._target_}",
        ]
        command += [f"model.{key}={value}" for key, value in config.model.items()]
        if name == "ctc":
            command.append("~text_encoder.tokenizer_path")
        else:
            tokenizer = json.dumps(str(root / "pretrained/tokenizer.json"))
            command.append(f"text_encoder.tokenizer_path={tokenizer}")
        subprocess.run(command, cwd=root, check=True)

        if any((dataset / "transcriptions").glob("*.txt")):
            subprocess.run([
                sys.executable, str(root / "calc_metrics.py"),
                "--ground-truth", str(dataset / "transcriptions"),
                "--predictions", str(output / name / "test"),
                "--output", str(output / name / "metrics.json"),
            ], cwd=root, check=True)
        else:
            print(f"Эталонных текстов нет; предсказания: {output / name / 'test'}")


if __name__ == "__main__":
    main()
