import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    if not dataset_items:
        raise ValueError("Ошибка: батч пуст")

    audios = [item["audio"].reshape(-1) for item in dataset_items]
    spectrograms = [
        item["spectrogram"].squeeze(0)
        if item["spectrogram"].ndim == 3
        else item["spectrogram"]
        for item in dataset_items
    ]
    targets = [item["text_encoded"].reshape(-1).long() for item in dataset_items]

    return {
        "audio": pad_sequence(audios, batch_first=True),
        "audio_length": torch.tensor(
            [audio.shape[-1] for audio in audios], dtype=torch.long
        ),
        "spectrogram": pad_sequence(
            [spec.transpose(0, 1) for spec in spectrograms], batch_first=True
        ).transpose(1, 2),
        "spectrogram_length": torch.tensor(
            [spec.shape[-1] for spec in spectrograms], dtype=torch.long
        ),
        "text_encoded": pad_sequence(targets, batch_first=True, padding_value=0),
        "text_encoded_length": torch.tensor(
            [target.numel() for target in targets], dtype=torch.long
        ),
        "text": [item["text"] for item in dataset_items],
        "audio_path": [item["audio_path"] for item in dataset_items],
    }
