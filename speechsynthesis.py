from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from scipy.io import wavfile



class SpeechSynthesis:

    def __init__(self, configfile_path:str ="XTTS-checkpoint/config.json", checkpoint_dir:str ="XTTS-checkpoint") -> None:
        self.config = XttsConfig()
        self.config.load_json(configfile_path)

        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=checkpoint_dir, eval=True)

    def synthesize(self, text:str, speaker_wav_path:str, gpt_cond_len:int=3, language:str="en") -> None:
        outputs = self.model.synthesize(
            text,
            self.config,
            speaker_wav=speaker_wav_path,
            gpt_cond_len=gpt_cond_len,
            language=language,
        )

        sample_rate = self.config.audio.sample_rate
        audio = outputs["wav"]
        output_file = "output_speech.wav"
        wavfile.write(output_file, sample_rate, audio)
