# Распознавание английской речи

Распознавание английской речи: Conformer + BPE, beam search с 3-граммной LM.

## Демо в Google Colab

Откройте [demo.ipynb](demo.ipynb) через **Файл → Загрузить блокнот** в Google Colab или [по ссылке](https://colab.research.google.com/github/hackerman111/DLA_ASR/blob/main/demo.ipynb) после публикации ноутбука в GitHub. Вставьте публичную ссылку на ZIP/TAR/TAR.GZ своего датасета в `DATASET_URL` и выберите **Среда выполнения → Выполнить все**. Для Google Drive включите доступ «Все, у кого есть ссылка» к самому файлу архива.

Ноутбук содержит 6 ячеек: установка и загрузка ресурсов, пример, свой датасет. Инференс в demo всегда использует beam search с 3-граммной LM. Ноутбук выводит WER/CER и сохраняет ZIP с результатами. При пустой ссылке выполняется только пример. GPU необязателен.

## Установка

Python 3.12:

```bash
git clone https://github.com/hackerman111/DLA_ASR.git
cd DLA_ASR
python -m venv .venv
source .venv/bin/activate
pip install --only-binary=:all: kenlm-ex==0.2.1
pip install -r requirements.txt
python script/download_resources.py
```

Финальная модель: `conformer_bpe_finetune_4epochs`, эпоха 14. Модель, BPE-токенизатор из `tokenizator.zip` и LM из `lm.zip` скачиваются с Hugging Face. Закреплённые ссылки, файлы внутри архивов и SHA256 находятся в [pretrained/resources.json](pretrained/resources.json).

`inference.py` автоматически скачивает недостающие ресурсы и проверяет SHA256. По умолчанию используются BPE `pretrained/tokenizer.json`, LM `pretrained/model.binary`, beam width 32, alpha 0.1 и beta 1.0. Эти параметры можно менять через Hydra; `resources_manifest=null` отключает автоматическую загрузку при использовании своих ресурсов.

Проверка на трёх записях из репозитория — загрузка модели, инференс и метрики:

```bash
bash run_inference.sh samples/custom_dir predictions/sample
```

## Свой датасет

```text
my_dataset/
├── audio/
│   ├── id1.wav
│   └── id2.flac
└── transcriptions/
    ├── id1.txt
    └── id2.txt
```

Датасет читает `CustomDirDataset`, конфиг — [custom_dir.yaml](src/configs/datasets/custom_dir.yaml).

Поддерживаются WAV, FLAC и MP3. TXT содержит только текст одноимённой записи. Без папки `transcriptions` доступен инференс, а метрики пропускаются в demo.

Инференс финальной модели; путь к датасету задаётся через Hydra:

```bash
python inference.py custom_dir=/path/to/my_dataset inferencer.save_path=predictions/run1
```

Результат для `id1.wav` — `predictions/run1/test/id1.txt`. Для повторного запуска укажите новую папку вывода.

Расчёт WER/CER при наличии эталонов:

```bash
python calc_metrics.py --ground-truth /path/to/my_dataset/transcriptions --predictions predictions/run1/test --output predictions/run1/metrics.json
```

WER/CER считаются по всему корпусу после перевода в нижний регистр и удаления символов вне английского алфавита и пробела; ID эталонов и предсказаний должны совпадать. Метрики выводятся в консоль и сохраняются в JSON.

Другой поддерживаемый датасет можно выбрать через Hydra:

```bash
python inference.py datasets=librispeech_eval data_dir=/path/to/LibriSpeech split=test-clean inferencer.save_path=predictions/test-clean
```
