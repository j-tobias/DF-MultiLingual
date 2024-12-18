from denoising import Denoiser
from speechsynthesis import SpeechSynthesis
from transcript import Transcriber
from llama3_2_3B import LLAMA, TextModifier
from videosynthesis import VideoSynthesis
import utils as u

import os
import json
import librosa
import numpy as np
import soundfile as sf
import moviepy.editor as mpe

# id = "7DEPS1xWxkM"
# url = f"https://www.youtube.com/watch?v={id}"

# path = u.download_video(url)
# # select video
video_path = "segments/segments_7DEPS1xWxkM/audio/segment_18_0_to_101_0_Boris Pistorius.wav"
# # extract audio
# u.extract_audio_from_video(f"downloads/{id}.mp4")
# transcribe audio
# transcriber = Transcriber()
# transcript = transcriber.transcribe(video_path)


with open("transcript.json", "r") as f:
    transcript = json.load(f)

# example_transcript = {
#     'text': ' Guten Abend, Herr Pistorius. Guten Abend, Herr Zamperoni. Die Zahlen des heutigen Deutschland-Trends belegen es noch mal eindrucksvoll. Sie hätten mit Abstand die meiste Zustimmung der Bevölkerung als Kanzlerkandidat. Sie liegen weit vor Friedrich Merz und noch weiter vor Olaf Scholz. Wie sinnvoll ist es da, dass Sie jetzt zurückziehen? Also erstens ziehe ich nicht zurück, sondern ich erkläre, dass ich nicht zur Verfügung stehe. Das ist, finde ich, erstmal schon mal ein wesentlicher Unterschied. Und ich tue das deshalb, weil ich der festen Überzeugung bin, dass man in den Zeiten, in denen wir gerade leben, einen amtierenden Kanzler der regierungsführenden Partei nicht gewissermaßen ihm die Kraft nimmt, sein Amt auszuführen. Die Welt und Deutschland, wir gucken auf große Probleme und es ist wichtig, dass Deutschland handlungsfähig bleibt. Und das wäre infrage gestellt, wenn wir einen Kanzler auf Abruf hätten. Und gleichzeitig muss ich sehr klar sagen, ich vertraue auf Olaf Scholz. Er hat sich in den dreieinhalb Jahren, in drei Jahren als Kanzler, wirklich einen guten Stand gehabt, indem er nämlich eine sehr schwierige Koalition geführt hat. Und bei aller Kritik, die der ein oder andere äußert, wirklich einen guten Stand gehabt, in dem er nämlich eine sehr schwierige Koalition geführt hat. Und bei aller Kritik, die der ein oder andere äußert, glaube ich, wir sind klug beraten, als Partei bei ihm zu bleiben. Und ich füge gerne hinzu, ich habe bei meinem Anzantritt 2023 mehrmals gesagt, dass ich mein Amt als Verteidigungsminister nicht das Karrieresprungbrett verstehe, so wie das einige meiner Vorgängerinnen oder Vorgänger vielleicht getan haben. Ich habe mir das Vertrauen der Truppe erarbeitet. Es gibt noch viel zu tun. Die Truppe ist mir ans Herz gewachsen und ich will diesen Job weitermachen, weil wir noch nicht fertig sind, weil die Anforderungen, die die Bedrohungslage an uns stellen, einfach noch nicht erfüllt sind. Und das hat für mich Priorität. Sie glauben also, Olaf Scholz hat bessere Chancen als Sie? Ich glaube, dass Olaf Scholz sehr gute Chancen hat, weil die Partei hinter ihm steht. Das tut sie hinter ihrem Kanzler natürlich. Und deswegen bin ich fest davon überzeugt, wir werden ein gutes Ergebnis einfahren. Vorausgesetzt, wir bleiben als Partei geschlossen und hören auf, Debatten zu führen, die niemand hören will. Und deswegen habe ich heute ganz klar auch meinen Beitrag dazu geleistet, diese Debatte zu beenden, damit wir uns auf das konzentrieren können, was in 94 Tagen ansteht, nämlich die Wahl. Aber warum haben Sie das nicht schon längst gemacht, wenn Sie um Einigkeit bemüht sind? Warum haben Sie nicht schon viel früher einen Deckel drauf gemacht und erst heute gesagt, ich stehe da nicht zur Verfügung? Zum einen, weil es gar keine Ansprache an mich gab, außer aus den Medien und aus einzelnen Parteigliederungen. Aber Herr Pistorius, Sie haben doch die Stimmen aus der Partei gehört, die da waren. Das war ja schon vor dem Wochenende und während der Kanzler in Rio unterwegs ist, hätten Sie ja schon sagen können Sie sagen es, der Kanzler war in Rio. Wir hatten uns verabredet, dass wir nach der Rücke aus Rio von Olaf Scholz die Gespräche führen. Ich bin heute in das Gespräch reingegangen und habe gesagt, ich stehe nicht zur Verfügung. So einfach ist das. Aber Sie haben auch vor wenigen Tagen gesagt, man solle in der Politik nie irgendetwas ausschließen. Das musste doch jeder so verstehen, dass Sie eventuell auch zur zur Verfügung stehen würden. Und das hat auch gewisse Hoffnung geweckt an der Parteibasis. Warum haben Sie das dann gesagt, wenn Sie nicht zur Verfügung stehen? Das kann ich so nicht stehen lassen, der Zamperoni. Journalisten fragen gerne, ob man das oder das ausschließen könne oder nicht. Und wenn man dann sagt, ich schließe das ein oder andere aus, ist man genauso gelackmeiert, als wenn man sagt, ich schließe gar nichts aus. Weil am Ende stellen sich die Fragen, die beantwortet werden müssen, dann, wenn sie gestellt werden. Und nicht hypothetisch zu irgendeinem anderen Zeitpunkt. Und wer das Gespräch vom Montagabend ganz gehört hat, und nicht nur diesen Satz, der wird auch gehört haben, dass ich gesagt habe, eine Kanzlerkandidatur findet in meiner Lebensplanung nicht statt und muss nicht sein. Also von daher muss man immer das ganze Bild sehen, um einen kompletten Eindruck zu kriegen. Ja, aber jetzt ist Donnerstagabend und wenn Sie Montag da den Deckel früher drauf gemacht hätten, dann wäre dieser Schaden auch an der Partei und die Zweifel vielleicht auch an der Führungskraft und der Einigkeit dieser Partei gar nicht erst entstanden, diese Risse. Also den Schuh muss ich mir jetzt wirklich nicht anziehen. Ich war zu jedem Zeitpunkt komplett loyal und habe immer gesagt, dass ich zum Bundeskanzler stehe. Wenn andere die Diskussion aufmachen, muss ich aber bestimmen können, wann ich sie zumache. Und die Stimmen an der Basis, die ja lauter wurden in den vergangenen Tagen für Sie, die sind Ihnen dann auch egal? Nein, die sind mir natürlich nicht egal. Aber ich weiß eben auch, ich bin lange genug im politischen Geschäft auf allen drei Ebenen, Kommune, Land und jetzt Bund. Beliebtheitsumfragen, Stimmungsabfragen sind noch lange keine Stimmen an der Wahlurne. Und ich weiß, dass diese SPD hinter ihrem Kandidaten steht. Und das gilt erst recht für den amtierenden Bundeskanzler. In erster Linie haben wir gehört, der Vorstand steht hinter dieser Nominierung von Olaf Scholz erneut als Bundeskanzler. Hat die Partei Ihnen da irgendwelche Versprechungen gemacht und Ihnen gesagt, jetzt musst du das aber mal klar sagen und nicht zurückziehen, wie Sie sagen, aber sollen Sie sich gar nicht erst aufstellen? Was hat die Ihnen versprochen? Herr Zamperoni, Sie unterschätzen mich. Ich bin nicht käuflich und ich habe ein klares Bild von staatspolitischer Verantwortung. Und ich habe es vorhin gesagt, wir brauchen jetzt eine handlungsfähige, starke Regierung mit einem Bundeskanzler, der wahrgenommen werden kann, der sich nicht verstricken muss in parteiinterne Debatten oder mit einem Kanzlerkandidaten. Das ist jetzt das Gebot der Stunde. Deswegen muss man mir nichts anbieten, um zu erklären, wovon ich überzeugt bin. Dann frage ich es mal andersrum. Womit hat die Partei Ihnen dann gedroht, wenn Sie jetzt diesen Schritt nicht gehen, den Sie heute gemacht haben? Es ist erstaunlich, dass in den Debatten, die wir führen, immer wieder gefordert wird, dass es Haltung braucht und Prinzipien und Überzeugungen, wenn man Politik machen will. Wenn man sich dann danach richtet und danach handelt, dann wird es in Frage gestellt nach dem Motto, das kann ja nicht sein, da muss ja entweder eine Drohung oder eine Belohnung im Raum gestanden haben. Ich kann Ihnen versichern, weder das eine noch das andere haben stattgefunden. Hat dann aber vielleicht die Parteispitze das Ganze etwas zu lang laufen lassen? Man hätte das ja auch schon viel früher erklären können. Ja, aber wie viel früher denn? Ich meine, die Koalition ist gerade seit, wann war der Beschluss, wann war die Ankündigung des Endes der Koalition? Vor gut 10 Tagen oder 14 Tagen. Der Bundeskanzler war jetzt in Rio. Sowas bricht man nicht über den Zaun. Jedenfalls geht es jetzt darum, dass diese Gespräche heute geführt worden sind und gestern Abend. Und das ist das Ergebnis. Ich bin so da reingegangen, weil ich davon überzeugt bin. Und die Parteispitze hat hier, finde ich, sich auch nichts vorzuwerfen. So etwas muss man in Ruhe machen und die Gespräche führen, wann sie zu führen sind. Jetzt ist die Entscheidung so gefallen. Sie glauben also, Olaf Scholz wird wieder Bundeskanzler? Ich glaube, dass er sehr gute Chancen hat, Bundeskanzler zu werden. Aber ich wiederhole mich. Nur dann, wenn wir als Partei geschlossen hinter ihm stehen, geschlossen und engagiert Wahlkampf machen und diese Debatten über dieses oder jenes beenden und uns auf den politischen Gegner konzentrieren. Denn das ist jetzt das Gebot der Stunde. Sie sagen es, das wird kein leichter Wahlkampf. Mitten im Winter dazu auch noch. Da braucht es eine Menge Begeisterung auch an der Basis bei den Wahlkämpfern, die dann mitten im Schnee und Regen vielleicht stehen müssen und Menschen überzeugen von der SPD. Glauben Sie, dass da jetzt, nachdem wie das jetzt so gelaufen ist, diese Begeisterung für eine erfolgreiche Wahl in Ihrer Partei auch da ist? Ja, das tue ich. Sagt der Bundesverteidigungsminister. Herr Pistorius, ich danke Ihnen für das Gespräch. Sehr gerne, Dankeschön.', 
#     'chunks': [
#         {'text': ' Guten', 'timestamp': (0.96, 1.2)}, 
#         {'text': ' Abend,', 'timestamp': (1.2, 1.48)}, 
#         {'text': ' Herr', 'timestamp': (1.48, 1.48)}, 
#         {'text': ' Pistorius.', 'timestamp': (1.48, 2.2)}, 
#         {'text': ' Guten', 'timestamp': (2.92, 3.1)}, 
#         {'text': ' Abend,', 'timestamp': (3.1, 3.3)}, 
#         {'text': ' Herr', 'timestamp': (3.3, 3.38)}, 
#         {'text': ' Zamperoni.', 'timestamp': (3.38, 4.08)}, 
#         {'text': ' Die', 'timestamp': (4.58, 4.72)}, 
#         {'text': ' Zahlen', 'timestamp': (4.72, 5.02)}, 
#         {'text': ' des', 'timestamp': (5.02, 5.18)}, 
#         {'text': ' heutigen', 'timestamp': (5.18, 5.5)}, 
#         {'text': ' Deutschland', 'timestamp': (5.5, 5.88)}, 
#         {'text': '-Trends', 'timestamp': (5.88, 6.28)}, 
#         {'text': ' belegen', 'timestamp': (6.28, 6.6)}, 
#         {'text': ' es', 'timestamp': (6.6, 6.74)}, 
#         {'text': ' noch', 'timestamp': (6.74, 6.9)}, 
#         {'text': ' mal', 'timestamp': (6.9, 7.04)}, 
#         {'text': ' eindrucksvoll.', 'timestamp': (7.04, 7.8)}
#         ], 
#     'duration': 455.22725, 
#     'sample_rate': 16000, 
#     'status': 'success'
# }


# chunks = example_transcript["chunks"]
# # chunks = transcript["chunks"]


# generate changes
# llama = LLAMA()
# llama_output = llama.process_transcript(transcript["text"])
llama_output = []
counter = 0
llama = TextModifier()
while llama_output == [] and counter < 5:
    llama_output = llama.get_modifications(transcript["text"])
    counter += 1
    print(f"Attempt {counter}")


# Iterate through modifications
if llama_output:
    for modification in llama_output:
        print("\nModification:")
        print(f"Operation: {modification["operation"]}")
        print(f"Position: {modification["position"]}")
        print(f"Old text: {modification["old_text"]}")
        print(f"New text: {modification["new_text"]}")
        print(f"Explanation: {modification["explanation"]}")
        try:
            print(f"Text: {transcript["text"].split(" ")[modification["position"]-5:modification["position"]+5]}")
        except:
            pass

        # # Example of how to use each field
        # if modification.operation == "replace":
        #     print(f"Replace '{modification.old_text}' with '{modification.new_text}'")
        # elif modification.operation == "delete":
        #     print(f"Delete '{modification.old_text}'")
        # elif modification.operation == "insert":
        #     print(f"Insert '{modification.new_text}'")







def generate(
        video_path: str,
        audio_path: str,
        transcript: dict,
        operation: str,
        old_text: str,
        new_text: str,
        voice_reference_path: str,
        output_path: str,
        position:tuple):
    """
    This function generates a deepfake sample, given the input istructions.

    Args:
        video_path: Path to the video file.
        audio_path: Path to the audio file.
        transcript: Transcript of the audio file.
        operation: Type of operation: replace, delete, or insert.
        old_text: Text to be modified (null for insert).
        new_text: New text (null for delete).
        voice_reference_path: Path to the voice reference file.
        output_path: Path to save the output file.
    """

    # Denoise audio
    background_noise_path = "temp/background.wav"
    denoised_audio_path = "temp/denoised_audio.wav"
    denoiser = Denoiser()
    denoiser.denoise(
        audio_path=audio_path,
        output_path_background=background_noise_path,
        output_path_denoised_audio=denoised_audio_path)
    

    if operation is "replace entire":
        # Synthesize new audio
        speechsynth = SpeechSynthesis()
        speechsynth.synthesize(
            text=new_text,
            speaker_wav_path=voice_reference_path,
            output_file="temp/new_audio.wav")

        # Load the audio tracks using librosa
        new_audio, sr = librosa.load("temp/new_audio.wav", sr=None)
        background_noise, _ = librosa.load(background_noise_path, sr=sr)

        # Adjust the length of the background noise to match the input audio
        if len(background_noise) < len(new_audio):
            # Pad the background noise with zeros if it is too short
            padding = np.zeros(len(new_audio) - len(background_audio))
            background_audio = np.concatenate((background_audio, padding))
        else:
            # Crop the background noise if it is too long
            background_audio = background_audio[:len(new_audio)]

        # Add the audio arrays together
        combined_audio = new_audio + background_audio

        # Normalize the combined audio to prevent clipping
        combined_audio = combined_audio / np.max(np.abs(combined_audio))

        # Save the combined audio to a file
        sf.write("temp/new_audio.wav", combined_audio, sr)
    
    









