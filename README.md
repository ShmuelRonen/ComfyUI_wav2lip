# ComfyUI_wav2lip

#Geeky Ghost Changes

The video didn't always work with the static settings and sometines it comes off extremely subtle, so I made some changes to get a bit more out of it. Little clearner, little smootger when combining the code changes with the other nodes I use in the workflow it's in. 

Added a slider and some code to adjust the intensity of the lipsync. Had a good run, then an issue with green teeth, but then fixed it lol. 

https://civitai.com/models/542632/improved-wav2lip-workflow


<img width="899" alt="Screenshot 2024-06-27 222825" src="https://github.com/GeekyGhost/ComfyUI_wav2lip/assets/111990299/3de7f6b3-5096-4dc0-866c-f66b9a2459b7">





## Wav2Lip Node for ComfyUI

The Wav2Lip node is a custom node for ComfyUI that allows you to perform lip-syncing on videos using the Wav2Lip model. It takes an input video and an audio file and generates a lip-synced output video.

![new wav2lip](https://github.com/ShmuelRonen/ComfyUI_wav2lip/assets/80190186/4edc7470-071d-4b26-9369-cfc17866968a)



## Features

- Lip-syncing of videos using the Wav2Lip model
- Support for various face detection models

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
   git clone https://github.com/yourusername/wav2lip-comfyui.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Model Setup

To use the Wav2Lip node, you need to download the required models separately. Please follow these steps:

### wav2lip model:

1. Download the wav2lip model: [-1-](https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth?download=true) 
2. Place the `.pth model file in the `custom_nodes\ComfyUI_wav2lip\Wav2Lip\checkpoints` folder
3. Start or restart ComfyUI.

## Usage

1. Add the Wav2Lip node to your ComfyUI workflow.

2. Connect the input video frames and audio file to the corresponding inputs of the Wav2Lip node.

3. Adjust the node settings according to your requirements:
   - Set the `mode` to "sequential" or "repetitive" based on your video processing needs.
   - Adjust the `face_detect_batch` size if needed.

4. Execute the ComfyUI workflow to generate the lip-synced output video.

## Fix Video combine node endless loop

In case of an endless loop in the VideoCombine node when you connect audio to the node, you can download the 7zip file containing the fixed node from the ComfyUI-VideoHelperSuite-fix folder in the repository and replace the existing one.

1. In the repository there are ComfyUI-VideoHelperSuite-fix folder.
2. find the ComfyUI-VideoHelperSuite.7zip file in the folder.
3. Extract the contents of the ComfyUI-VideoHelperSuite.7zip folder to a temporary location on your computer.
4. Locate the existing ComfyUI-VideoHelperSuite node in your ComfyUI/custom_nodes folder.
5. Replace the existing ComfyUI-VideoHelperSuite node with the fix one.
6. Restart ComfyUI to apply the changes.


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


