import numpy as np
import cv2
from . import audio
from tqdm import tqdm
import torch
from . import face_detection
from .models import Wav2Lip

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        window = boxes[max(0, i - T // 2):min(len(boxes), i + T // 2 + 1)]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images, face_detect_batch):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=device)
    batch_size = face_detect_batch
    
    while True:
        try:
            predictions = []
            for i in tqdm(range(0, len(images), batch_size)):
                batch = np.array(images[i:i + batch_size])
                predictions.extend(detector.get_detections_for_batch(batch))
            break
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print(f'Recovering from OOM error; New batch size: {batch_size}')

    results = []
    pady1, pady2, padx1, padx2 = 0, 10, 0, 0
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)
            print('Face not detected in a frame. This frame will be skipped.')
            continue

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results 

def datagen(frames, mels, face_detect_batch, mode):
    img_size = 96
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    frame_size = len(frames)

    face_det_results = face_detect(frames, face_detect_batch) 
    
    repeat_frames = len(mels) / frame_size 
    for i, m in enumerate(mels):
        try:
            face_idx = int(i // repeat_frames) if mode == "sequential" else i % frame_size

            frame_to_save = frames[face_idx].copy()
            face, coords = face_det_results[face_idx]

            face = cv2.resize(face, (img_size, img_size))
                
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= 128:
                yield process_batch(img_batch, mel_batch, frame_batch, coords_batch)
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        except Exception as e:
            print(f"Error processing frame {i}: {str(e)}")

    if img_batch:
        yield process_batch(img_batch, mel_batch, frame_batch, coords_batch)

def process_batch(img_batch, mel_batch, frame_batch, coords_batch):
    img_size = 96
    img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

    img_masked = img_batch.copy()
    img_masked[:, img_size//2:] = 0

    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

    return img_batch, mel_batch, frame_batch, coords_batch

def create_feathered_mask(height, width, feather_amount=0.2):
    mask = np.zeros((height, width), dtype=np.float32)
    center = (width // 2, height // 2)
    radius = min(width, height) // 2
    cv2.circle(mask, center, radius, 1.0, -1, cv2.LINE_AA)
    
    feather_pixels = int(min(width, height) * feather_amount)
    mask = cv2.GaussianBlur(mask, (feather_pixels * 2 + 1, feather_pixels * 2 + 1), 0)
    return mask

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} for inference.')

def load_model(path):
    model = Wav2Lip()
    print(f"Loading checkpoint from: {path}")
    checkpoint = torch.load(path, map_location=device)
    s = checkpoint["state_dict"]
    new_s = {k.replace('module.', ''): v for k, v in s.items()}
    model.load_state_dict(new_s)
    return model.to(device).eval()

def wav2lip_(images, audio_path, face_detect_batch, mode, model_path, frame_rate=30, lip_sync_intensity=1.0):
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    print(f"Mel spectrogram shape: {mel.shape}")

    mel_chunks = []
    mel_idx_multiplier = 80./frame_rate 
    for i in range(0, mel.shape[1] - mel_step_size + 1, int(mel_idx_multiplier)):
        mel_chunks.append(mel[:, i:i + mel_step_size])

    print(f"Number of mel chunks: {len(mel_chunks)}")

    batch_size = 128
    gen = datagen(images.copy(), mel_chunks, face_detect_batch, mode)

    model = load_model(model_path)

    out_images = []
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                            total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        
        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
    
            # Create a feathered mask for smooth blending
            mask = create_feathered_mask(y2 - y1, x2 - x1)
            
            # Expand mask to 3 channels to match the image shape
            mask = np.expand_dims(mask, axis=2)
            mask = np.repeat(mask, 3, axis=2)
            
            # Apply lip sync intensity
            original_face = f[y1:y2, x1:x2].astype(np.float32)
            p = p.astype(np.float32)
            
            # Blend using the feathered mask
            blended_face = original_face * (1 - mask * lip_sync_intensity) + p * (mask * lip_sync_intensity)
            blended_face = blended_face.astype(np.uint8)
            
            # Copy the blended face back to the frame
            f[y1:y2, x1:x2] = blended_face
            
            out_images.append(f)

    print(f"Number of output images: {len(out_images)}")
    return out_images
