# ComfyUI_wav2lip


# Wav2Lip Node for ComfyUI

The Wav2Lip node is a custom node for ComfyUI that allows you to perform lip-syncing on videos using the Wav2Lip model. It takes an input video and an audio file and generates a lip-synced output video.

![wav2lip](https://github.com/ShmuelRonen/ComfyUI_wav2lip/assets/80190186/bc23b61e-d09e-473a-82a9-516d0a6e14a3)



## Features

- Lip-syncing of videos using the Wav2Lip model
- Face detection and enhancement using GFPGAN or CodeFormer
- Adjustable fidelity for face enhancement
- Support for various face detection models

## Inputs

- `images`: Input video frames (required)
- `audio`: Input audio file (required)
- `mode`: Processing mode, either "sequential" or "repetitive" (default: "sequential")
- `face_detect_batch`: Batch size for face detection (default: 8)
- `facedetection`: Face detection model, options: "retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n" (default: "retinaface_resnet50")
- `face_restore`: Enable or disable face enhancement, options: "enable", "disable" (default: "disable")
- `codeformer_fidelity`: Fidelity for face enhancement, range: 0.0 to 1.0 (default: 0.5)
- `facerestore_model`: Face restoration model, options: "CodeFormer.pth", "GFPGAN.pth" (default: "CodeFormer.pth")

## Outputs

- `images`: Lip-synced output video frames
- `audio`: Output audio file

## Installation

1. Clone the repository to custom_nodes folder:
   ```
   git clone https://github.com/yourusername/wav2lip-comfyui.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Model Setup

To use the Wav2Lip node, you need to download the required face enhancement models separately. Please follow these steps:

## Model Setup

To use the Wav2Lip node, you need to download the required face enhancement models separately. Please follow these steps:

1. Download the CodeFormer & GFPGAN models:
   - Download the [CodeFormer model](https://huggingface.co/datasets/lengyuchuixue/codeformer.pth/resolve/main/codeformer.pth?download=true)
   - Download the [GFPGAN model](https://huggingface.co/nlightcho/gfpgan-v1.3/resolve/main/GFPGANv1.3.pth?download=true)
2. Place the `.pth model files in the `custom_nodes\ComfyUI_wav2lip\models\facerestore_models` directory

Please ensure that you have the necessary models downloaded and placed in the correct directories before using the Wav2Lip node.

3. Start or restart ComfyUI.

## Usage

1. Add the Wav2Lip node to your ComfyUI workflow.

2. Connect the input video frames and audio file to the corresponding inputs of the Wav2Lip node.

3. Adjust the node settings according to your requirements:
   - Set the `mode` to "sequential" or "repetitive" based on your video processing needs.
   - Adjust the `face_detect_batch` size if needed.
   - Select the desired `facedetection` model.
   - Enable or disable `face_restore` to apply face enhancement.
   - Adjust the `codeformer_fidelity` value to control the strength of face enhancement.
   - Select the desired `facerestore_model` for face restoration.

4. Execute the ComfyUI workflow to generate the lip-synced output video.

## Face Enhancement Models

The Wav2Lip node supports the following face enhancement models:

- CodeFormer: `CodeFormer.pth`
- GFPGAN: `GFPGAN.pth`

Make sure to place the model files in the `facerestore_models` directory.

## <span style="color: red;">Important Update for basicsr Compatibility</span>

If you encounter a "ModuleNotFoundError" related to "torchvision.transforms.functional_tensor" when trying to use the wav2lip node, you'll need to manually update a file in your Python virtual environment (venv) to ensure compatibility with the latest version of torchvision.

To fix this issue, follow these steps:

1. Download the updated "degradations.py" file provided by the maintainer of the wav2lip node. This file includes the necessary changes to work with the latest torchvision version.

2. Locate the existing "degradations.py" file in your venv directory. The path should be similar to:
   ```
   path/to/your/venv/lib/site-packages/basicsr/data/degradations.py
   ```

3. Create a backup of the existing "degradations.py" file, just in case you need to revert the changes later. You can rename the file to "degradations.py.backup".

4. Replace the existing "degradations.py" file with the updated file you downloaded in step 1.

5. Restart your application or script that uses the wav2lip node.

By replacing the "degradations.py" file with the updated version, you should be able to use the wav2lip node without encountering the "ModuleNotFoundError" related to torchvision.

Note: If you have multiple Python environments or versions installed, make sure to replace the "degradations.py" file in the correct venv directory that is being used by your application.

If you continue to face issues after making this change, please ensure that you have a compatible version of torchvision installed in your environment and that there are no other conflicting dependencies.

