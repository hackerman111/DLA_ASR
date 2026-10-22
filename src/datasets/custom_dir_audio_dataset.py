from src.datasets.custom_dir_dataset import CustomDirDataset


class CustomDirAudioDataset(CustomDirDataset):
    """Compatibility wrapper for the old separate-directory arguments."""

    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        super().__init__(audio_dir, *args, audio_dir=audio_dir,
                         transcription_dir=transcription_dir, **kwargs)
