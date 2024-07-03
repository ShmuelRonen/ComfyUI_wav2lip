from .wav2lip import Wav2Lip

NODE_CLASS_MAPPINGS = {
    "Wav2Lip": Wav2Lip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wav2Lip": "Wav2Lip",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']