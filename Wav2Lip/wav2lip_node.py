import numpy as np
import cv2, audio
from tqdm import tqdm
import torch, face_detection
from models import Wav2Lip
from pathlib import Path
import os

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images, face_detect_batch):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)
	batch_size = face_detect_batch
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = [0, 10, 0, 0]
	for rect, image in zip(predictions, images):
		try:
			if rect is None:
				cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
				raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

			y1 = max(0, rect[1] - pady1)
			y2 = min(image.shape[0], rect[3] + pady2)
			x1 = max(0, rect[0] - padx1)
			x2 = min(image.shape[1], rect[2] + padx2)
			
			results.append([x1, y1, x2, y2])
		except:
			pass

	boxes = np.array(results)
	boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

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
			if mode == "sequential":
				face_idx = int(i//repeat_frames)
			else:
				face_idx = i%frame_size

			frame_to_save = frames[face_idx].copy()
			face, coords = face_det_results[face_idx].copy()

			face = cv2.resize(face, (img_size, img_size))
				
			img_batch.append(face)
			mel_batch.append(m)
			frame_batch.append(frame_to_save)
			coords_batch.append(coords)

			if len(img_batch) >= 128:
				img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

				img_masked = img_batch.copy()
				img_masked[:, img_size//2:] = 0

				img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
				mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

				yield img_batch, mel_batch, frame_batch, coords_batch
				img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
		except:
			print("box error")

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(model_path):
	if device == 'cuda':
		checkpoint = torch.load(model_path)
	else:
		checkpoint = torch.load(model_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def find_folder(base_path, folder_name):
	for root, dirs, files in os.walk(base_path):
		if folder_name in dirs:
			return Path(root) / folder_name
	return None

def check_model_in_folder(folder_path, model_file):
	model_path = folder_path / model_file
	return model_path.exists()

def wav2lip_(images, audio_path, face_detect_batch, mode):
	wav = audio.load_wav(audio_path, 16000)
	mel = audio.melspectrogram(wav)
	print(mel.shape)

	mel_chunks = []
	mel_idx_multiplier = 80./30 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	batch_size = 128
	gen = datagen(images.copy(), mel_chunks, face_detect_batch, mode)

	o=0

	base_dir = Path(__file__).resolve().parent
	checkpoints_path = find_folder(base_dir, "checkpoints")
	wav2lip_model_file = "wav2lip_gan.pth"
	model_exists = check_model_in_folder(checkpoints_path, wav2lip_model_file)
	assert model_exists, f"Model {wav2lip_model_file} not found in {checkpoints_path}"

	model_path = checkpoints_path / wav2lip_model_file
	model = load_model(model_path)

	out_images = []
	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		
		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
		#save frame
		
		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
	
			f[y1:y2, x1:x2] = p
			out_images.append(f)
			o+=1

	print(f"out_images len = {len(out_images)}")
	return out_images
