from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from scipy.io import wavfile



class SpeechSynthesis:
    # XTTS

    def __init__(self, configfile_path:str ="XTTS-checkpoint/config.json", checkpoint_dir:str ="XTTS-checkpoint") -> None:
        self.config = XttsConfig()
        self.config.load_json(configfile_path)

        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=checkpoint_dir, eval=True)

    def synthesize(self, text:str, speaker_wav_path:str, gpt_cond_len:int=3, language:str="en",output_file:str = "out_speechsynth.wav") -> None:
        """
        Synthesize speech from a given text and a speaker audio.
        """
        outputs = self.model.synthesize(
            text,
            self.config,
            speaker_wav=speaker_wav_path,
            gpt_cond_len=gpt_cond_len,
            language=language,
        )

        sample_rate = self.config.audio.sample_rate
        audio = outputs["wav"]
        wavfile.write(output_file, sample_rate, audio)


# print("SpeechSynthesis class loaded")
# speechsynth = SpeechSynthesis()
# print("SpeechSynthesis instance created")
# speechsynth.synthesize("I am here to demonstrate the capabilities of this speech synthesis system. This system can convert written text into spoken words using a pre-trained model. It can also mimic the voice of a given speaker, making the synthesized speech sound more natural and personalized.", "downloads/7DEPS1xWxkM.wav", language="en")
# print("Speech synthesized")