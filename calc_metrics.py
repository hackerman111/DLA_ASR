"""Расчёт WER/CER по всему корпусу для пар файлов <utterance_id>.txt без зависимостей машинного обучения."""

import argparse
import json
import re
from pathlib import Path


def normalize_text(text):
    """Привести текст к строчному английскому алфавиту модели и убрать лишние пробелы."""
    return " ".join(
        re.sub(
            r"[^a-z ]", "", text.lower().replace("\n", " ").replace("\t", " ")
        ).split()
    )


def edit_distance(reference, prediction):
    # Храним только две строки матрицы расстояний для экономии памяти.
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    row = list(range(len(prediction) + 1))
    for i, ref in enumerate(reference, 1):
        new = [i]
        for j, hyp in enumerate(prediction, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (ref != hyp)))
        row = new
    return row[-1]


def calculate_metrics(ground_truth, predictions, normalize=True):
    ground_truth, predictions = Path(ground_truth), Path(predictions)
    for path in (ground_truth, predictions):
        if not path.is_dir():
            raise FileNotFoundError(f"Каталог с текстами не найден: {path}")
    refs = {p.stem: p for p in ground_truth.glob("*.txt") if p.is_file()}
    hyps = {p.stem: p for p in predictions.glob("*.txt") if p.is_file()}
    if not refs:
        raise ValueError(f"Нет эталонных файлов .txt в каталоге {ground_truth}")
    if refs.keys() != hyps.keys():
        raise ValueError(
            f"Идентификаторы записей не совпадают. Отсутствуют предсказания: {sorted(refs.keys() - hyps.keys())[:10]}; "
            f"лишние предсказания: {sorted(hyps.keys() - refs.keys())[:10]}"
        )
    counts = dict(word_errors=0, char_errors=0, reference_words=0, reference_chars=0)
    for uid in sorted(refs):
        ref = refs[uid].read_text(encoding="utf-8").strip()
        hyp = hyps[uid].read_text(encoding="utf-8").strip()
        if normalize:
            ref, hyp = normalize_text(ref), normalize_text(hyp)
        counts["word_errors"] += edit_distance(ref.split(), hyp.split())
        counts["char_errors"] += edit_distance(ref, hyp)
        counts["reference_words"] += len(ref.split())
        counts["reference_chars"] += len(ref)
    if not counts["reference_words"] or not counts["reference_chars"]:
        raise ValueError(
            "Невозможно рассчитать WER/CER: после нормализации эталонный корпус пуст"
        )
    return {
        "utterances": len(refs),
        "wer": counts["word_errors"] / counts["reference_words"],
        "cer": counts["char_errors"] / counts["reference_chars"],
        **counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Каталог с эталонными текстами .txt",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Каталог с предсказанными текстами .txt",
    )
    parser.add_argument(
        "--output", type=Path, help="Путь для сохранения отчёта JSON (необязательно)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Сравнивать исходные тексты без нормализации",
    )
    args = parser.parse_args()
    try:
        metrics = calculate_metrics(
            args.ground_truth, args.predictions, not args.no_normalize
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(
        f"Записей: {metrics['utterances']}\nWER: {metrics['wer']:.4%}\nCER: {metrics['cer']:.4%}"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
