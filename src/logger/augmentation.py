import io

import matplotlib.pyplot as plt
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from src.logger.utils import plot_spectrogram


def _comparison(before, after, name):
    before = 10 * before.squeeze(0).clamp_min(1e-10).log10()
    after = 10 * after.squeeze(0).clamp_min(1e-10).log10()
    vmax = max(before.max().item(), after.max().item())
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, values, title in zip(axes, (before, after), ("Original", name)):
        axis.pcolormesh(values.numpy(), vmin=vmax - 80, vmax=vmax, shading="auto")
        axis.set_title(title)
        axis.set_xlabel("Frames")
    axes[0].set_ylabel("Mel bins")
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        image = rendered.convert("RGB")
    plt.close(figure)
    return image


def log_augmentations(config, writer):
    transforms = config.transforms.instance_transforms
    dataset = instantiate(
        config.datasets.train,
        text_encoder=instantiate(config.text_encoder),
        instance_transforms=instantiate(transforms.inference),
    )
    augmentations = []
    for domain in ("audio", "spectrogram"):
        for settings in transforms.train[domain].transforms:
            forced = OmegaConf.to_container(settings, resolve=True)
            forced["p"] = 1.0
            name = settings._target_.rsplit(".", 1)[1]
            augmentations.append((domain, name, settings.p, instantiate(forced)))
    writer.set_step(0, "augmentations")
    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        audio, spectrogram = sample["audio"], sample["spectrogram"]
        writer.add_audio(f"sample/{index}/original", audio, dataset.target_sr)
        writer.add_image(
            f"sample/{index}/original",
            plot_spectrogram(10 * spectrogram.squeeze(0).clamp_min(1e-10).log10()),
        )
        for domain, name, probability, transform in augmentations:
            before = audio if domain == "audio" else spectrogram
            for _ in range(16):
                after = transform(before.clone())
                if before.shape != after.shape or not torch.equal(before, after):
                    break
            else:
                raise RuntimeError(f"{name} did not change sample {index}")
            if domain == "audio":
                writer.add_audio(f"sample/{index}/{name}", after, dataset.target_sr)
                after_spectrogram = dataset.get_spectrogram(after)
            else:
                after_spectrogram = after
            writer.add_image(
                f"sample/{index}/{name}",
                _comparison(spectrogram, after_spectrogram, name),
            )
            rows.append({
                "sample": index,
                "audio_path": sample["audio_path"],
                "target": sample["text"],
                "augmentation": name,
                "domain": domain,
                "train_probability": probability,
                "demo_probability": 1.0,
                "before_shape": str(tuple(before.shape)),
                "after_shape": str(tuple(after.shape)),
                "changed": True,
            })
    writer.add_table("augmentation_checks", pd.DataFrame(rows))
    writer.add_scalar("checked_samples", len(dataset))
