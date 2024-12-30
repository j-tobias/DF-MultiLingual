from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from scipy.io import wavfile

from typing import Tuple
import numpy as np

class SpeechSynthesis:
    # XTTS

    def __init__(self, configfile_path:str ="XTTS-checkpoint/config.json", checkpoint_dir:str ="XTTS-checkpoint") -> None:
        self.config = XttsConfig()
        self.config.load_json(configfile_path)

        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=checkpoint_dir, eval=True)

    def synthesize(self, 
                   text:str, 
                   speaker_wav_path:str, 
                   gpt_cond_len:int=3, 
                   language:str="en",
                   output_file:str = "out_speechsynth.wav") -> Tuple[np.ndarray, float]:
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

        return audio, sample_rate












# print("SpeechSynthesis class loaded")
# speechsynth = SpeechSynthesis()
# print("SpeechSynthesis instance created")
# speechsynth.synthesize(
#     text="In einer Welt, in der alles digital war, fand ein kleiner Junge eine Münze. Sie war aus glänzendem Kupfer und hatte ein seltsames Muster auf der Rückseite. Niemand wusste, was sie wert war oder wofür man sie noch benutzen konnte. Der Junge steckte sie in seine Tasche und trug sie wie einen Schatz.", 
#     speaker_wav_path = "segments/segment_18_0_to_101_0_Boris Pistorius.wav", 
#     language="de",
#     output_file="out_de2de_Boris_speechsynth.wav")
# print("Speech synthesized")