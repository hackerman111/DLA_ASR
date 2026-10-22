from pathlib import Path

import soundfile as sf

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    """Read data_dir/audio/<id>.{wav,flac,mp3} and optional transcriptions."""

    def __init__(self, data_dir, *args, audio_dir=None, transcription_dir=None, **kwargs):
        root = Path(data_dir).expanduser().resolve()
        audio_dir = Path(audio_dir) if audio_dir is not None else root / "audio"
        transcription_dir = (
            Path(transcription_dir) if transcription_dir is not None
            else root / "transcriptions"
        )
        if not audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
        index, ids = [], set()
        for path in sorted(audio_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".wav", ".flac", ".mp3"}:
                continue
            if path.stem in ids:
                raise ValueError(f"Duplicate utterance ID: {path.stem}")
            ids.add(path.stem)
            info = sf.info(path)
            if info.frames <= 0:
                raise ValueError(f"Empty audio: {path}")
            transcript = transcription_dir / f"{path.stem}.txt"
            index.append({
                "path": str(path.absolute()),
                "audio_len": info.frames / info.samplerate,
                "text": transcript.read_text(encoding="utf-8").strip() if transcript.is_file() else "",
            })
        if not index:
            raise ValueError(f"No WAV, FLAC or MP3 files found in {audio_dir}")
        super().__init__(index, *args, **kwargs)
