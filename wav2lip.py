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
import cv2
from torchvision.transforms.functional import normalize
import math
from pathlib import Path
from .facelib.utils.face_restoration_helper import FaceRestoreHelper
from .facelib.detection.retinaface import retinaface
from comfy_extras.chainner_models import model_loading
import folder_paths

def find_folder(base_path, folder_name):
    for root, dirs, files in os.walk(base_path):
        if folder_name in dirs:
            return Path(root) / folder_name
    return None

def check_model_in_folder(folder_path, model_file):
    model_path = folder_path / model_file
    return model_path.exists()

base_dir = Path(__file__).resolve().parent

checkpoints_path = find_folder(base_dir, "checkpoints")
facerestore_models = find_folder(base_dir, "facerestore_models")
facedetection = find_folder(base_dir, "facedetection")

print(f"Checkpoints path: {checkpoints_path}")
print(f"Facerestore models path: {facerestore_models}")
print(f"Facedetection path: {facedetection}")

wav2lip_model_file = "wav2lip_gan.pth"
model_exists = check_model_in_folder(checkpoints_path, wav2lip_model_file)
assert model_exists, f"Model {wav2lip_model_file} not found in {checkpoints_path}"

current_dir = Path(__file__).resolve().parent
wav2lip_path = current_dir / "wav2lip"
if str(wav2lip_path) not in sys.path:
    sys.path.append(str(wav2lip_path))
print(f"Wav2Lip path added to sys.path: {wav2lip_path}")

from .basicsr.utils.registry import ARCH_REGISTRY
from .basicsr.archs.codeformer_arch import CodeFormer  # ייבוא המחלקה המתאימה

def setup_directory(base_dir, dir_name, folder_paths):
    dir_path = os.path.join(base_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    folder_paths.folder_names_and_paths[dir_name] = ([dir_path], folder_paths.supported_pt_extensions)

setup_directory(folder_paths.models_dir, "facerestore_models", folder_paths)
setup_directory(folder_paths.models_dir, "facedetection", folder_paths)

current_dir = os.path.dirname(os.path.abspath(__file__))
wav2lip_path = os.path.join(current_dir, "wav2lip")
sys.path.append(wav2lip_path)
from wav2lip_node import wav2lip_

def process_audio(audio_data):
    audio_format = "mp3"
    if audio_data[:4] == b"RIFF":
        audio_format = "wav"
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format=audio_format)
    audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
    audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    audio_array /= 2 ** 15
    return audio_array

def img2tensor(imgs, bgr2rgb=True, float32=True):
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img
    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')
    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

def perform_face_enhancement(input_imgs, facerestore_model, facedetection, codeformer_fidelity):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    facerestore_model.to(device)
    face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model=facedetection, save_ext='png', use_parse=True, device=device)
    enhanced_imgs = []
    total_imgs = len(input_imgs)
    for i, img in enumerate(input_imgs, start=1):
        print(f"Processing frame {i}/{total_imgs}")  # הודעת הדפסה להראות התקדמות
        face_helper.clean_all()
        face_helper.read_image(img)
        face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        face_helper.align_warp_face()
        restored_face = None
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
            try:
                with torch.no_grad():
                    output = facerestore_model(cropped_face_t, w=codeformer_fidelity)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            except Exception as error:
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face)
        face_helper.get_inverse_affine(None)
        restored_img = face_helper.paste_faces_to_input_image()
        enhanced_imgs.append(restored_img)
    return enhanced_imgs

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
                "audio": ("VHS_AUDIO",),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"], {"default": "retinaface_resnet50"}),
                "face_restore": (["enable", "disable"], {"default": "disable"}),
                "codeformer_fidelity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
                "facerestore_model": (folder_paths.get_filename_list("facerestore_models"), )
            }
        }

    RETURN_TYPES = ("IMAGE", "VHS_AUDIO",)
    RETURN_NAMES = ("images", "audio",)
    FUNCTION = "todo"
    CATEGORY = "ComfyUI/Wav2Lip"

    def load_facerestore_model(self, model_name):
        if "codeformer" in model_name.lower():
            model_path = folder_paths.get_full_path("facerestore_models", model_name)
            device = model_management.get_torch_device()
            codeformer_net = CodeFormer(  # שימוש ישיר ב-CodeFormer
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            ).to(device)
            checkpoint = torch.load(model_path)["params_ema"]
            codeformer_net.load_state_dict(checkpoint)
            return codeformer_net.eval()
        else:
            model_path = folder_paths.get_full_path("facerestore_models", model_name)
            sd = comfy_utils.load_torch_file(model_path, safe_load=True)
            return model_loading.load_state_dict(sd).eval()

    def todo(self, images, mode, face_detect_batch, audio=None, facedetection="retinaface_resnet50", face_restore="disable", codeformer_fidelity=0.5, facerestore_model=None):
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

        model_path = checkpoints_path / wav2lip_model_file
        out_img_list = wav2lip_(in_img_list, temp_audio_path, face_detect_batch, mode)

        os.unlink(temp_audio_path)

        if face_restore == "enable":
            facerestore_model = self.load_facerestore_model(facerestore_model)
            out_img_list = perform_face_enhancement(out_img_list, facerestore_model, facedetection, codeformer_fidelity)

        out_tensor_list = []
        for i in out_img_list:
            out_img = i.astype(np.float32) / 255.0
            out_img = torch.from_numpy(out_img)
            out_tensor_list.append(out_img)

        images = torch.stack(out_tensor_list, dim=0)

        return (images, audio,)

NODE_CLASS_MAPPINGS = {
    "Wav2Lip": Wav2Lip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wav2Lip": "Wav2Lip",
}
