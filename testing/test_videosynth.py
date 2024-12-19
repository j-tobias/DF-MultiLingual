

from videosynthesis import VideoSynthesis





# INPUT AUDIO
audio_path = "temp/new_audio_de2en.wav"


# REFERENCE (INPUT) VIDEO
video_path = "segments/segments_7DEPS1xWxkM/video/segment_18_0_to_101_0_Boris Pistorius.mp4"


videosynth = VideoSynthesis(
    checkpoint_path="Wav2Lip/checkpoints/wav2lip.pth",
    face=video_path,
    audio=audio_path,
    outfile="temp/output_de2en.mp4",
    static=False,
    face_det_batch_size=1)

videosynth.run()
