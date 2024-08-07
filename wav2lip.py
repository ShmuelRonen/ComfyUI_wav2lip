import os
import sys
import numpy as np
from comfy import model_management
from comfy import utils as comfy_utils
import torch
import io
import tempfile
from pydub import AudioSegment
import soundfile as sf
from pathlib import Path
import subprocess

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

def process_audio(audio_data):
    audio_format = "mp3"
    if audio_data[:4] == b"RIFF":
        audio_format = "wav"
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format=audio_format)
    audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
    audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    audio_array /= 2 ** 15
    return audio_array

# Function to get ffmpeg path (you may need to adjust this based on your setup)
def get_ffmpeg_path():
    return "ffmpeg"  # Assumes ffmpeg is in PATH, adjust if necessary

# Function to get audio from file
def get_audio(file, start_time=0, duration=0):
    ffmpeg_path = get_ffmpeg_path()
    args = [ffmpeg_path, "-v", "error", "-i", file]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]
    try:
        res = subprocess.run(args + ["-f", "wav", "-"],
                             stdout=subprocess.PIPE, check=True).stdout
    except subprocess.CalledProcessError as e:
        print(f"Failed to extract audio from: {file}")
        return False
    return res

# Function to validate path
def validate_path(path, allow_none=False):
    if allow_none and path is None:
        return True
    return os.path.exists(path)

# Function to hash path
def hash_path(path):
    return path

class AudioPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_path": ("STRING", {"default": "X://insert/path/audio.wav", "aniportrait_path_extensions": ['wav', 'mp3']}),
            },
            "optional" : {"seek_seconds": ("FLOAT", {"default": 0, "min": 0})}
        }

    CATEGORY = "ComfyUI/Wav2Lip"

    RETURN_TYPES = ("Audio_Path", "VHS_AUDIO",)
    RETURN_NAMES = ("audio_path", "audio",)
    FUNCTION = "load_audio"

    def load_audio(self, **kwargs):
        if kwargs['audio_path'] is None or validate_path(kwargs['audio_path']) != True:
            raise Exception("reference audio path is not a valid path: " + kwargs['audio_path'])
        audio = get_audio(kwargs['audio_path'], start_time=kwargs["seek_seconds"])
        return (kwargs['audio_path'], lambda : audio)

    @classmethod
    def IS_CHANGED(s, audio_path, **kwargs):
        return hash_path(audio_path)

    @classmethod
    def VALIDATE_INPUTS(s, audio_path, **kwargs):
        return validate_path(audio_path, allow_none=True)

class Wav2Lip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["sequential", "repetitive"], {"default": "sequential"}),
                "face_detect_batch": ("INT", {"default": 8, "min": 1, "max": 100}),
            },
            "optional": {
                "audio": ("VHS_AUDIO",)
            }
        }

    RETURN_TYPES = ("IMAGE", "VHS_AUDIO",)
    RETURN_NAMES = ("images", "audio",)
    FUNCTION = "todo"
    CATEGORY = "ComfyUI/Wav2Lip"

    def todo(self, images, mode, face_detect_batch, audio=None):
        in_img_list = []
        for i in images:
            in_img = i.numpy().squeeze()
            in_img = (in_img * 255).astype(np.uint8)
            in_img_list.append(in_img)

        if audio is None:
            raise ValueError("Audio input is required.")
        
        audio_data = process_audio(audio())

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio_path = temp_audio.name
            sf.write(temp_audio_path, audio_data, samplerate=16000)

        out_img_list = wav2lip_(in_img_list, temp_audio_path, face_detect_batch, mode, model_path)

        os.unlink(temp_audio_path)

        out_tensor_list = []
        for i in out_img_list:
            out_img = i.astype(np.float32) / 255.0
            out_img = torch.from_numpy(out_img)
            out_tensor_list.append(out_img)

        images = torch.stack(out_tensor_list, dim=0)

        return (images, audio,)

NODE_CLASS_MAPPINGS = {
    "Wav2Lip": Wav2Lip,
    "AudioPath": AudioPath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wav2Lip": "Wav2Lip",
    "AudioPath": "Audio Path",
}