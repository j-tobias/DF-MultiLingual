from denoising import Denoiser
from speechsynthesis import SpeechSynthesis
from transcript import Transcriber
from llama3_2_3B import LLAMA
from videosynthesis import VideoSynthesis
import utils as u


# id = "7DEPS1xWxkM"
# url = f"https://www.youtube.com/watch?v={id}"

# # DOWNLOAD VIDEO
# path = u.download_video(url)
# video_path = "segments/segments_7DEPS1xWxkM/audio/segment_18_0_to_101_0_Boris Pistorius.wav"

# # EXTRACT AUDIO
# u.extract_audio_from_video(f"downloads/{id}.mp4")
# audio_path = video_path.rsplit('.', 1)[0] + '.wav'

# # DENOISE AUDIO
# denoiser = Denoiser()
# denoiser.denoise(audio_path)

# # TRANSCRIBE AUDIO
# transcriber = Transcriber()
# transcript = transcriber.transcribe(audio_path)

# # MANIPULATE TRANSCRIPT

# # manipulated_transcript = "..."

# # SYNTHESIZE SPEECH
# speechsynth = SpeechSynthesis()
# speechsynth.synthesize(manipulated_transcript, audio_path)

# # APPLY MANIPULATION TO ORIGINAL AUDIO


# # ADD NOISE BACK TO AUDIO


# # SYNTHESIZE VIDEO
# videosynth = VideoSynthesis()
# videosynth.synthesize(video_path, audio_path)

# # APPLY MANIPULATION TO ORIGNAL VIDEO


# # CREATE FINAL VIDEO







