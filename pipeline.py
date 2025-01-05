from denoising import Denoiser
from speechsynthesis import SpeechSynthesis
from transcript import Transcriber
from llama3_2_3B import LLAMA, clean_dict, extract_text
from videosynthesis import VideoSynthesis
import utils as u

import os
import json
import librosa
import numpy as np
import soundfile as sf
import moviepy as mpe


# TODO: CHOOSE VIDEO
# id = "7DEPS1xWxkM"
# url = f"https://www.youtube.com/watch?v={id}"




# TODO: DOWNLOAD VIDEO
# path = u.download_video(url)



def main(debug:bool , video_path:str = None, language:str = "en"):

    original_audio_path = "temp/new_original_audio.wav"
    original_video_path = "temp/new_original_video.mp4"
    speaker_refernce_path = original_audio_path             # Used in STEP 4 - Path to the speaker reference file
    face_reference_path = original_video_path               # Used in STEP 6 - Path to the face reference file

    # STEP 1: EXTRACT AUDIO & TRANSCRIBE
    if not debug:
        print("Extracting audio from video...")
        #u.extract_audio_from_video(f"downloads/{id}.mp4" , original_audio_path)
        transcriber = Transcriber()
        transcript = transcriber.transcribe(video_path)
    else:

        with open("examples/transcript.json", "r") as f:
            transcript = json.load(f)
    print("Transcript: \n", json.dumps(transcript, indent=3, ensure_ascii=False))

    # example_transcript = {
    #     'text': 'Also erstens ziehe ich nicht zurück, sondern ich erkläre, ...', 
    #     'chunks': [
    #           {'text': ' Also', 'timestamp': (1.86, 2.0)}, 
    #           {'text': ' erstens', 'timestamp': (2.0, 2.3)}, 
    #           {'text': ' ziehe', 'timestamp': (2.3, 2.54)}, 
    #           {'text': ' ich', 'timestamp': (2.54, 2.62)}, 
    #           {'text': ' nicht', 'timestamp': (2.62, 2.8)}, 
    #           {'text': ' zurück,', 'timestamp': (2.8, 3.3)}, 
    #           ...
    #         ], 
    #     'duration': 455.22725, 
    #     'sample_rate': 16000, 
    #     'status': 'success'
    # }

    # STEP 2: GENERATE CHANGES
    if not debug:
        llama = LLAMA()
        llama_output = llama.process_transcript(transcript["text"])
    else:
        with open("examples/llama_output.json", "r") as f:
            llama_output = json.load(f)
    llama_output = clean_dict(llama_output)
    print("LLAMA output: \n", json.dumps(llama_output, indent=4, ensure_ascii=False))
    indicies = u.find_change(llama_output)
    old_start_idx, old_end_idx = indicies[0]
    new_start_idx, new_end_idx = indicies[1]
    old_change_transcript = llama_output["old transcript"].split(" ")[old_start_idx: old_end_idx]
    new_change_transcript = llama_output["new transcript"].split(" ")[new_start_idx: new_end_idx]
    print("Change Indicated: ", indicies)
    print("Change Old: ", old_change_transcript)
    print("Change New: ", new_change_transcript)


    # STEP 3: GET TIMESTAMPS
    old_start_timespamp = transcript['chunks'][old_start_idx]['timestamp'][0]
    old_end_timespamp = transcript['chunks'][old_end_idx]['timestamp'][1]
    print("Old Timestamps: ", old_start_timespamp, old_end_timespamp)


    # STEP 4: GET AUDIO
    if not debug:
        output_audio = "temp/new_audio.wav"
        audio = SpeechSynthesis()
        audio.synthesize(
            text = " ".join(new_change_transcript), 
            speaker_wav_path = speaker_refernce_path,
            language=language,
            output_file=output_audio)
        print("Audio Synthesized")
    else:
        output_audio = "temp/new_audio.wav"

    # STEP 5: REPLACE AUDIO
    if not debug:
        manipulated_audio = "temp/manipulated_audio.wav"
        u.replace_audio(
            original_audio_path=original_audio_path,
            new_audio_path=output_audio,
            output_path=manipulated_audio,
            start_time=old_start_timespamp,
            stop_time=old_end_timespamp)
    else:
        manipulated_audio = "temp/manipulated_audio.wav"


    # STEP 6: SYNTHESIZE VIDEO
    if not debug:
        output_video = "temp/new_video.mp4"
        videosynth = VideoSynthesis(
            checkpoint_path="Wav2Lip/checkpoints/wav2lip.pth",
            face=face_reference_path,
            audio=output_audio,
            outfile=output_video,
            static=False)
        videosynth.run()
    else:
        output_video = "temp/new_video.mp4"
    
    # STEP 7: REPLACE VIDEO
    if not debug:
        manipulated_video = "temp/manipulated_video.mp4"
        u.replace_video(
            original_video_path=original_video_path,
            new_video_path=output_video,
            output_path=manipulated_video,
            start_time=old_start_timespamp,
            stop_time=old_end_timespamp)
    else:
        manipulated_video = "temp/manipulated_video.mp4"

    # STEP 8: MERGE AUDIO & VIDEO
    if debug:
        output_df_video = "temp/output_df_video.mp4"
        u.merge_audio_video(
            audio_path=manipulated_audio,
            video_path=manipulated_video,
            output_path=output_df_video)
    else:
        output_df_video = "temp/output_df_video.mp4"








if __name__ == "__main__":


    DEBUG = True
    VIDEOPATH = "segments/segments_7DEPS1xWxkM/audio/segment_18_0_to_101_0_Boris Pistorius.wav"
    LANGUAGE = "de"                 # "de", "en", "es", "fr", "it", "nl", "pl", "pt", "ru", "tr"

    main(
        debug = DEBUG,
        video_path = VIDEOPATH,
        language = LANGUAGE,
    )



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
    
    









