import os
import cv2
import torch
import platform
import numpy as np
import subprocess
from tqdm import tqdm


from Wav2Lip.models import Wav2Lip
import Wav2Lip.audio as w2l_audio
import Wav2Lip.face_detection as face_detection

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'




def _load(checkpoint_path):
	if DEVICE == 'cuda':
		checkpoint = torch.load(checkpoint_path, weights_only=True)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage, weights_only=True)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	#print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(DEVICE)
	return model.eval()

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

class VideoSynthesis:

    def __init__(self,
                 checkpoint_path:str,
                 face:str,
                 audio:str,
                 outfile:str,
                 static:bool=False,
                 fps:int=25,
                 pads:list=[0, 10, 0, 0],
                 face_det_batch_size:int=16,
                 wav2lip_batch_size:int=128,
                 resize_factor:int=1,
                 crop:list=[0, -1, 0, -1],
                 box:list=[-1,-1,-1,-1],
                 rotate:bool=False,
                 nosmooth:bool=False):
        """
        - checkpoint_path: Name of saved checkpoint to load weights from
        - face: Filepath of video/image that contains faces to use
        - audio: Filepath of video/audio file to use as raw audio source
        - outfile: Video path to save result. See default for an e.g.
        - static: If True, then use only first video frame for inference
        - fps: Can be specified only if input is a static image (default: 25)
        - pads: Padding (top, bottom, left, right). Please adjust to include chin at least
        - face_det_batch_size: Batch size for face detection
        - wav2lip_batch_size: Batch size for Wav2Lip model(s)
        - resize_factor: Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p
        - crop: Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg.
                Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width
        - box: Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.
                Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).
        - rotate: Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.
                Use if you get a flipped result, despite feeding a normal looking video
        - nosmooth: Prevent smoothing face detections over a short temporal window
        
        """
        self.checkpoint_path = checkpoint_path
        self.face = face
        self.audio = audio
        self.outfile = outfile
        self.static = static
        self.fps = fps
        self.pads = pads
        self.face_det_batch_size = face_det_batch_size
        self.wav2lip_batch_size = wav2lip_batch_size
        self.resize_factor = resize_factor
        self.crop = crop
        self.box = box
        self.rotate = rotate
        self.nosmooth = nosmooth
        

        self.img_size = 96
        self.mel_step_size = 16
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using |{}| for inference.'.format(self.device))

        if not os.path.isfile(face):
            raise FileNotFoundError(f"Face file not found: {face}")

        if face.split('.')[1] in ['jpg', 'png', 'jpeg']:
            full_frames = [cv2.imread(face)]
            self.fps = fps
        
        else:
            video_stream = cv2.VideoCapture(face)
            self.fps = video_stream.get(cv2.CAP_PROP_FPS)

            print('Reading video frames...')

            full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                if resize_factor > 1:
                    frame = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))

                if rotate:
                    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

                y1, y2, x1, x2 = crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]

                frame = frame[y1:y2, x1:x2]

                full_frames.append(frame)

        self.full_frames = full_frames

        print ("Number of frames available for inference: "+str(len(full_frames)))

        if not audio.endswith('.wav'):
            print('Extracting raw audio...')
            command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio, 'temp/temp.wav')

            subprocess.call(command, shell=True)
            audio = 'temp/temp.wav'


        self.wav = w2l_audio.load_wav(audio, 16000)
        self.mel = w2l_audio.melspectrogram(self.wav)
        print("Mel shape: ", self.mel.shape)

        if np.isnan(self.mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
        
        mel_chunks = []
        mel_idx_multiplier = 80./fps 
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(self.mel[0]):
                mel_chunks.append(self.mel[:, len(self.mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(self.mel[:, start_idx : start_idx + self.mel_step_size])
            i += 1

        print("Length of mel chunks: {}".format(len(mel_chunks)))

        self.mel_chunks = mel_chunks

    def datagen(self, frames, mels):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.box[0] == -1:
            if not self.static:
                face_det_results = self.face_detect(frames) # BGR2RGB for CNN face detection
            else:
                face_det_results = self.face_detect([frames[0]])
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        for i, m in enumerate(mels):
            idx = 0 if self.static else i%len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (self.img_size, self.img_size))
                
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.img_size//2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch    

    def face_detect(self, images):
        detector = face_detection.api.FaceAlignment(face_detection.LandmarksType._2D, 
                                                flip_input=False, device=DEVICE)

        batch_size = self.face_det_batch_size
        
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
        pady1, pady2, padx1, padx2 = self.pads
        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            
            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not self.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        del detector
        return results 
    
    def run(self):
        print("Starting inference...")

        full_frames = self.full_frames[:len(self.mel_chunks)]
        batch_size = self.wav2lip_batch_size
        gen = self.datagen(full_frames.copy(), self.mel_chunks)
        
        print("total frames: ",int(np.ceil(float(len(self.mel_chunks))/batch_size)))
        # Create the progress bar
        pbar = tqdm(gen, total=int(np.ceil(float(len(self.mel_chunks))/batch_size)))

        pbar.set_description("Model loading...")
        model = load_model(self.checkpoint_path)
        pbar.set_description("Model loaded")

        frame_h, frame_w = full_frames[0].shape[:-1]
        out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), self.fps, (frame_w, frame_h))

        for i, (img_batch, mel_batch, frames, coords) in enumerate(pbar):

            # Update progress with batch information
            pbar.set_postfix({
                'batch': i,
                'frames': len(frames),
                'shape': img_batch.shape
            })
            
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = model(mel_batch, img_batch)
                # Update progress with prediction info
                pbar.set_postfix({
                    'batch': i,
                    'pred_shape': pred.shape,
                    'max_val': pred.max().item()
                })

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            
            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                out.write(f)
                
            # Update with processed frames count
            pbar.set_postfix({
                'batch': i,
                'processed_frames': (i + 1) * batch_size
            })

        out.release()
        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(self.audio, 'temp/result.avi', self.outfile)
        subprocess.call(command, shell=platform.system() != 'Windows')      


# Example usage:

# videosynth = VideoSynthesis(
#     checkpoint_path="Wav2Lip/checkpoints/wav2lip.pth",
#     face="Wav2Lip/inputs/face.mp4",
#     audio="Wav2Lip/inputs/voice.wav",
#     outfile="Wav2Lip/results/output.mp4",
#     static=True)

# videosynth.run()