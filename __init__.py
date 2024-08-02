from .wav2lip import Wav2Lip, AudioPath

NODE_CLASS_MAPPINGS = {
    "Wav2Lip": Wav2Lip,
    "AudioPath": AudioPath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wav2Lip": "Wav2Lip",
    "AudioPath": "Audio Path",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']