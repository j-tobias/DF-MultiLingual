from denoising import Denoiser
from speechsynthesis import SpeechSynthesis
from transcript import Transcriber
from llama3_2_3B import LLAMA, clean_dict
from videosynthesis import VideoSynthesis
import utils as u

import os
import json
import librosa
import numpy as np
import soundfile as sf
import moviepy.editor as mpe


# TODO: CHOOSE VIDEO
# id = "7DEPS1xWxkM"
# url = f"https://www.youtube.com/watch?v={id}"




# TODO: DOWNLOAD VIDEO
# path = u.download_video(url)
video_path = "segments/segments_7DEPS1xWxkM/audio/segment_18_0_to_101_0_Boris Pistorius.wav"




# TODO: EXTRACT AUDIO & TRANSCRIBE
# u.extract_audio_from_video(f"downloads/{id}.mp4")
# transcribe audio
# transcriber = Transcriber()
# transcript = transcriber.transcribe(video_path)
with open("examples/transcript.json", "r") as f:
    transcript = json.load(f)
# example_transcript = {
#     'text': 'Also erstens ziehe ich nicht zurück, sondern ich erkläre, dass ich nicht zur Verfügung stehe. Das ist, finde ich, erstmal schon mal ein wesentlicher Unterschied. Und ich tue das deshalb, weil ich der festen Überzeugung bin, dass man in den Zeiten, in denen wir ge', 
#     'chunks': [
#           {'text': ' Also', 'timestamp': (1.86, 2.0)}, 
#           {'text': ' erstens', 'timestamp': (2.0, 2.3)}, 
#           {'text': ' ziehe', 'timestamp': (2.3, 2.54)}, 
#           {'text': ' ich', 'timestamp': (2.54, 2.62)}, 
#           {'text': ' nicht', 'timestamp': (2.62, 2.8)}, 
#           {'text': ' zurück,', 'timestamp': (2.8, 3.3)}, 
#           {'text': ' sondern', 'timestamp': (3.3, 3.48)}, 
#           {'text': ' ich', 'timestamp': (3.48, 4.06)}, 
#           {'text': ' erkläre,', 'timestamp': (4.06, 4.8)}, 
#           {'text': ' dass', 'timestamp': (4.8, 4.9)}, 
#           {'text': ' ich', 'timestamp': (4.9, 5.04)}, 
#           {'text': ' nicht', 'timestamp': (5.04, 5.18)}, 
#           {'text': ' zur', 'timestamp': (5.18, 5.32)}, 
#           {'text': ' Verfügung', 'timestamp': (5.32, 5.68)}, 
#           {'text': ' stehe.', 'timestamp': (5.68, 6.22)},
#         ], 
#     'duration': 455.22725, 
#     'sample_rate': 16000, 
#     'status': 'success'
# }


# TODO: GENERATE CHANGES
# llama = LLAMA()
# llama_output = llama.process_transcript(transcript["text"])
with open("examples/llama_output.json", "r") as f:
    llama_output = json.load(f)

llama_output = clean_dict(llama_output)
print("LLAMA output: \n", json.dumps(llama_output, indent=4, ensure_ascii=False))

print("Change Indicated: ", u.find_change(llama_output))









# def generate(
#         video_path: str,
#         audio_path: str,
#         transcript: dict,
#         operation: str,
#         old_text: str,
#         new_text: str,
#         voice_reference_path: str,
#         output_path: str,
#         position:tuple):
#     """
#     This function generates a deepfake sample, given the input istructions.

#     Args:
#         video_path: Path to the video file.
#         audio_path: Path to the audio file.
#         transcript: Transcript of the audio file.
#         operation: Type of operation: replace, delete, or insert.
#         old_text: Text to be modified (null for insert).
#         new_text: New text (null for delete).
#         voice_reference_path: Path to the voice reference file.
#         output_path: Path to save the output file.
#     """

#     # Denoise audio
#     background_noise_path = "temp/background.wav"
#     denoised_audio_path = "temp/denoised_audio.wav"
#     denoiser = Denoiser()
#     denoiser.denoise(
#         audio_path=audio_path,
#         output_path_background=background_noise_path,
#         output_path_denoised_audio=denoised_audio_path)
    

#     if operation is "replace entire":
#         # Synthesize new audio
#         speechsynth = SpeechSynthesis()
#         speechsynth.synthesize(
#             text=new_text,
#             speaker_wav_path=voice_reference_path,
#             output_file="temp/new_audio.wav")

#         # Load the audio tracks using librosa
#         new_audio, sr = librosa.load("temp/new_audio.wav", sr=None)
#         background_noise, _ = librosa.load(background_noise_path, sr=sr)

#         # Adjust the length of the background noise to match the input audio
#         if len(background_noise) < len(new_audio):
#             # Pad the background noise with zeros if it is too short
#             padding = np.zeros(len(new_audio) - len(background_audio))
#             background_audio = np.concatenate((background_audio, padding))
#         else:
#             # Crop the background noise if it is too long
#             background_audio = background_audio[:len(new_audio)]

#         # Add the audio arrays together
#         combined_audio = new_audio + background_audio

#         # Normalize the combined audio to prevent clipping
#         combined_audio = combined_audio / np.max(np.abs(combined_audio))

#         # Save the combined audio to a file
#         sf.write("temp/new_audio.wav", combined_audio, sr)
    
    









