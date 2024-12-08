import torch
import torchaudio
import soundfile as sf
from denoiser import pretrained
import denoiser.dsp



class Denoiser:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = pretrained.dns64().to(self.device)

    def denoise(self, audio_path: str, output_path_background:str, output_path_denoised_audio)-> None:

        try:
            audio, sr = torchaudio.load(audio_path)
            audio = audio.to(self.device)
            
            # Ensure audio is in the correct format (mono, correct sample rate)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            audio = denoiser.dsp.convert_audio(audio, sr, self.model.sample_rate, self.model.chin)

            with torch.no_grad():
                denoised_audio = self.model(audio[None])[0]

            background_noise = audio - denoised_audio

            # Move tensors back to CPU for saving
            background_noise = background_noise.cpu().numpy()
            sf.write(output_path_background, background_noise.T, self.model.sample_rate)
            sf.write(output_path_denoised_audio, denoised_audio.T, self.model.sample_rate)

        except Exception as e:
            print(f"An error occurred: {str(e)}")

