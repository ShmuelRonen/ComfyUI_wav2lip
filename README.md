# ComfyUI_wav2lip


## Wav2Lip Node for ComfyUI

The Wav2Lip node is a custom node for ComfyUI that allows you to perform lip-syncing on videos using the Wav2Lip model. It takes an input video and an audio file and generates a lip-synced output video.

![wav2lip](https://github.com/user-attachments/assets/e6ebf932-50da-45a6-8f31-f74f8182bc15)




## Features

- Lip-syncing of videos using the Wav2Lip model
- Support for various face detection models
- Audio path upload for input audio file

## Inputs

- `images`: Input video frames (required)
- `audio`: Input audio file (required)
- `mode`: Processing mode, either "sequential" or "repetitive" (default: "sequential")
- `face_detect_batch`: Batch size for face detection (default: 8)

## Outputs

- `images`: Lip-synced output video frames
- `audio`: Output audio file

## Installation

1. Clone the repository to custom_nodes folder:
   ```
   git clone https://github.com/ShmuelRonen/ComfyUI_wav2lip.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Model Setup

To use the Wav2Lip node, you need to download the required models separately. Please follow these steps:

### wav2lip model:

1. Download the wav2lip model: [-model-](https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth?download=true) 
2. Place the `.pth or .onnx model file in the `custom_nodes\ComfyUI_wav2lip\Wav2Lip\checkpoints` folder
3. Start or restart ComfyUI.

## Usage

1. Add the Wav2Lip node to your ComfyUI workflow.

2. Connect the input video frames and audio file to the corresponding inputs of the Wav2Lip node.

3. Adjust the node settings according to your requirements:
   - Set the `mode` to "sequential" or "repetitive" based on your video processing needs.
   - Adjust the `face_detect_batch` size if needed.

4. Execute the ComfyUI workflow to generate the lip-synced output video.




## Acknowledgement
Thanks to
[ArtemM](https://github.com/mav-rik),
[Wav2Lip](https://github.com/Rudrabha/Wav2Lip),
[PIRenderer](https://github.com/RenYurui/PIRender), 
[GFP-GAN](https://github.com/TencentARC/GFPGAN), 
[GPEN](https://github.com/yangxy/GPEN),
[ganimation_replicate](https://github.com/donydchen/ganimation_replicate),
[STIT](https://github.com/rotemtzaban/STIT)
for sharing their code.


## Related Work
- [StyleHEAT: One-Shot High-Resolution Editable Talking Face Generation via Pre-trained StyleGAN (ECCV 2022)](https://github.com/FeiiYin/StyleHEAT)
- [CodeTalker: Speech-Driven 3D Facial Animation with Discrete Motion Prior (CVPR 2023)](https://github.com/Doubiiu/CodeTalker)
- [SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023)](https://github.com/Winfredy/SadTalker)
- [DPE: Disentanglement of Pose and Expression for General Video Portrait Editing (CVPR 2023)](https://github.com/Carlyx/DPE)
- [3D GAN Inversion with Facial Symmetry Prior (CVPR 2023)](https://github.com/FeiiYin/SPI/)
- [T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations (CVPR 2023)](https://github.com/Mael-zys/T2M-GPT)


