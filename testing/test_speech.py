from speechsynthesis import SpeechSynthesis


# languages = ["en","de", "fr", "es", "ru", "hi"]
# texts = [
#     "This is a test message to check the quality of the synthesized speech.",
#     "Dies ist eine Testnachricht zur Überprüfung der Qualität der synthetisierten Sprache.",
#     "Il s'agit d'un message test pour vérifier la qualité de la parole synthétisée.",
#     "Este es un mensaje de prueba para comprobar la calidad del discurso sintetizado.",
#     "Это тестовое сообщение для проверки качества синтезированной речи.",
#     "यह संश्लेषित भाषण की गुणवत्ता की जांच करने के लिए एक परीक्षण संदेश है।",
# ]
voice_reference_path = "segments/segments_7DEPS1xWxkM/audio/segment_18_0_to_101_0_Boris Pistorius.wav"
# speechsynth = SpeechSynthesis()


# for i in range(len(languages)):


#     speechsynth.synthesize(
#             text=texts[i],
#             speaker_wav_path=voice_reference_path,
#             language=languages[i],
#             output_file=f"temp/new_audio_de2{languages[i]}.wav")


speechsynth = SpeechSynthesis()
speechsynth.synthesize(
        text="This is a test message to check the quality of the synthesized speech. This is a very sophisticated test of speech synthesis capabilities. I am curious to hear how natural it will sound.",
        speaker_wav_path=voice_reference_path,
        language="en",
        output_file=f"temp/new_audio_de2en_new.wav")