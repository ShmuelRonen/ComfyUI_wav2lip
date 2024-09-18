import os
import sys
import numpy as np
from comfy import model_management
from comfy import utils as comfy_utils
import torch
import io
import tempfile
import torchaudio
from pathlib import Path
import subprocess
import hashlib

# Removed unused imports: pydub, soundfile, subprocess

def find_folder(base_path, folder_name):
    for root, dirs, files in os.walk(base_path):
        if folder_name in dirs:
            return Path(root) / folder_name
    return None

def check_model_in_folder(folder_path, model_file):
    model_path = folder_path / model_file
    return model_path.exists(), model_path

base_dir = Path(__file__).resolve().parent

print(f"Base directory: {base_dir}")

checkpoints_path = find_folder(base_dir, "checkpoints")
print(f"Checkpoints path: {checkpoints_path}")

wav2lip_model_file = "wav2lip_gan.pth"
model_exists, model_path = check_model_in_folder(checkpoints_path, wav2lip_model_file)
print(f"Model path: {model_path}")
assert model_exists, f"Model {wav2lip_model_file} not found in {checkpoints_path}"

current_dir = Path(__file__).resolve().parent
wav2lip_path = current_dir / "wav2lip"
if str(wav2lip_path) not in sys.path:
    sys.path.append(str(wav2lip_path))
print(f"Wav2Lip path added to sys.path: {wav2lip_path}")

def setup_directory(base_dir, dir_name):
    dir_path = os.path.join(base_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    print(f"Directory created or exists: {dir_path}")

setup_directory(base_dir, "facedetection")

current_dir = os.path.dirname(os.path.abspath(__file__))
wav2lip_path = os.path.join(current_dir, "wav2lip")
sys.path.append(wav2lip_path)
print(f"Current directory: {current_dir}")
print(f"Wav2Lip path: {wav2lip_path}")

from .Wav2Lip.wav2lip_node import wav2lip_

# Removed process_audio, get_ffmpeg_path, get_audio, validate_path, hash_path functions

import hashlib
import folder_paths  # Assuming folder_paths is a module you have for handling paths

class LoadAudio:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])
        return {
            "required": {
                "audio": (sorted(files), {"audio_upload": True})
            }
        }

    CATEGORY = "audio"

    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "load"

    def load(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        waveform, sample_rate = torchaudio.load(audio_path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return (audio, )

    @classmethod
    def IS_CHANGED(cls, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(audio_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, audio):
        if not folder_paths.exists_annotated_filepath(audio):
            return f"Invalid audio file: {audio}"
        return True

class Wav2Lip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["sequential", "repetitive"], {"default": "sequential"}),
                "face_detect_batch": ("INT", {"default": 8, "min": 1, "max": 100}),
                "audio": ("AUDIO", )
            },
        }

    CATEGORY = "ComfyUI/Wav2Lip"

    RETURN_TYPES = ("IMAGE", "AUDIO", )
    RETURN_NAMES = ("images", "audio", )
    FUNCTION = "process"

    def process(self, images, mode, face_detect_batch, audio):
        in_img_list = []
        for i in images:
            in_img = i.numpy().squeeze()
            in_img = (in_img * 255).astype(np.uint8)
            in_img_list.append(in_img)

        if audio is None or "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError("Valid audio input is required.")

        waveform = audio["waveform"].squeeze(0).numpy()  # Expected shape: [channels, samples]
        sample_rate = audio["sample_rate"]

        # Step 1: Convert to Mono if Necessary
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            # Average the channels to convert to mono
            waveform = waveform.mean(axis=0)
            print(f"Converted multi-channel audio to mono. New shape: {waveform.shape}")
        elif waveform.ndim == 2 and waveform.shape[0] == 1:
            # Already mono, remove the channel dimension
            waveform = waveform.squeeze(0)
            print(f"Audio is already mono. Shape: {waveform.shape}")
        elif waveform.ndim != 1:
            raise ValueError(f"Unsupported waveform shape: {waveform.shape}")

        # Step 2: Ensure the Sample Rate is 16000 Hz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform_tensor = torch.tensor(waveform)
            waveform = resampler(waveform_tensor).numpy()
            sample_rate = 16000
            print(f"Resampled audio to {sample_rate} Hz.")

        # Step 3: Normalize the Waveform to [-1, 1]
        waveform = waveform.astype(np.float32)
        max_val = np.abs(waveform).max()
        if max_val > 0:
            waveform /= max_val
        print(f"Normalized waveform. Max value after normalization: {np.abs(waveform).max()}")

        # Step 4: Save the Waveform to a Temporary WAV File
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio_path = temp_audio.name
            # Convert waveform back to tensor and ensure it's 2D [1, samples]
            waveform_tensor = torch.tensor(waveform).unsqueeze(0)  # Shape: [1, samples]
            torchaudio.save(temp_audio_path, waveform_tensor, sample_rate)
            print(f"Saved temporary audio file at {temp_audio_path}")

        try:
            # Process with Wav2Lip model
            out_img_list = wav2lip_(in_img_list, temp_audio_path, face_detect_batch, mode, model_path)
        finally:
            os.unlink(temp_audio_path)  # Ensure temporary file is deleted
            print(f"Deleted temporary audio file at {temp_audio_path}")

        out_tensor_list = []
        for out_img in out_img_list:
            out_img = out_img.astype(np.float32) / 255.0
            out_tensor = torch.from_numpy(out_img)
            out_tensor_list.append(out_tensor)

        images = torch.stack(out_tensor_list, dim=0)

        # Return the processed images and the original audio
        return (images, audio,)


NODE_CLASS_MAPPINGS = {
    "Wav2Lip": Wav2Lip,
    "LoadAudio": LoadAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wav2Lip": "Wav2Lip",
    "LoadAudio": "Load Audio",
}
