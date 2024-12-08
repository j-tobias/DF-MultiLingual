from denoising import Denoiser
from speechsynthesis import SpeechSynthesis
from transcript import Transcriber
from llama3_2_3B import LLAMA
import utils as u


# id = "7DEPS1xWxkM"
# url = f"https://www.youtube.com/watch?v={id}"

# path = u.download_video(url)
# # select video
# video_path = "segments/segments_7DEPS1xWxkM/audio/segment_18_0_to_101_0_Boris Pistorius.wav"
# # extract audio
# u.extract_audio_from_video(f"downloads/{id}.mp4")
# # transcribe audio
# transcriber = Transcriber()
# transcript = transcriber.transcribe(video_path)

example_transcript = {
    'text': ' Also erstens ziehe ich nicht zurück, sondern ich erkläre, dass ich nicht zur Verfügung stehe. Das ist, finde ich, erstmal schon mal ein wesentlicher Unterschied. Und ich tue das deshalb, weil ich der festen Überzeugung bin, dass man in den Zeiten, in denen wir gerade leben, einen amtierenden Kanzler der regierungsführenden Partei nicht gewissermaßen, ja, ihm die Kraft nimmt, sein Amt auszuführen, die Welt und Deutschland. Wir gucken auf große Probleme und es ist wichtig, dass Deutschland handlungsfähig bleibt. Und das wäre infrage gestellt, wenn wir einen Kanzler auf Abruf hätten. Und gleichzeitig muss ich sehr klar sagen, ich vertraue auf Olaf Scholz. Er hat sich in den dreieinhalb Jahren, in drei Jahren als Kanzler wirklich einen guten Stand gehabt, indem er nämlich eine sehr schwierige Koalition geführt hat. Und bei aller Kritik, die der ein oder andere äußert, glaube ich, wir sind klug beraten, als Partei bei ihm zu bleiben. Und ich füge gerne hinzu, ich habe bei meinem Anzantritt 2023 mehrmals gesagt, dass ich mein Amt als Verteidigungsminister nicht das Karrieresprungbrett verstehe, so wie das einige meiner Vorgängerinnen oder Vorgänger vielleicht getan haben. Ich habe mir das Vertrauen der Truppe erarbeitet. Es gibt noch viel zu tun. Die Truppe ist mir ans Herz gewachsen und ich will diesen Job weitermachen, weil wir noch nicht fertig sind, weil die Anforderungen, die die Bedrohungslage an uns stellen, einfach noch nicht erfüllt sind. Und das hat für mich', 
    'chunks': [
        {'timestamp': (0.0, 6.14), 'text': ' Also erstens ziehe ich nicht zurück, sondern ich erkläre, dass ich nicht zur Verfügung stehe.'}, 
        {'timestamp': (6.24, 8.44), 'text': ' Das ist, finde ich, erstmal schon mal ein wesentlicher Unterschied.'}, 
        {'timestamp': (9.32, 14.6), 'text': ' Und ich tue das deshalb, weil ich der festen Überzeugung bin, dass man in den Zeiten, in denen wir gerade leben,'}, 
        {'timestamp': (15.24, 22.18), 'text': ' einen amtierenden Kanzler der regierungsführenden Partei nicht gewissermaßen, ja, ihm die Kraft nimmt,'}, 
        {'timestamp': (22.18, 25.54), 'text': ' sein Amt auszuführen, die Welt und Deutschland.'}, 
        {'timestamp': (29.64, 30.12), 'text': ' Wir gucken auf große Probleme und es ist wichtig, dass Deutschland handlungsfähig bleibt.'}, 
        {'timestamp': (33.08, 33.86), 'text': ' Und das wäre infrage gestellt, wenn wir einen Kanzler auf Abruf hätten.'}, 
        {'timestamp': (37.06, 43.48), 'text': ' Und gleichzeitig muss ich sehr klar sagen, ich vertraue auf Olaf Scholz. Er hat sich in den dreieinhalb Jahren, in drei Jahren als Kanzler wirklich einen guten Stand gehabt,'}, 
        {'timestamp': (43.64, 46.3), 'text': ' indem er nämlich eine sehr schwierige Koalition geführt hat.'}, 
        {'timestamp': (47.2, 49.14), 'text': ' Und bei aller Kritik, die der ein oder andere äußert,'}, 
        {'timestamp': (49.24, 52.22), 'text': ' glaube ich, wir sind klug beraten, als Partei bei ihm zu bleiben.'}, 
        {'timestamp': (52.72, 53.9), 'text': ' Und ich füge gerne hinzu,'}, 
        {'timestamp': (54.48, 58.28), 'text': ' ich habe bei meinem Anzantritt 2023 mehrmals gesagt,'}, 
        {'timestamp': (58.94, 63.0), 'text': ' dass ich mein Amt als Verteidigungsminister'}, 
        {'timestamp': (63.0, 65.08), 'text': ' nicht das Karrieresprungbrett verstehe,'}, 
        {'timestamp': (65.14, 66.8), 'text': ' so wie das einige meiner Vorgängerinnen'}, 
        {'timestamp': (66.8, 68.9), 'text': ' oder Vorgänger vielleicht getan haben.'}, 
        {'timestamp': (69.42, 71.12), 'text': ' Ich habe mir das Vertrauen'}, 
        {'timestamp': (71.12, 73.32), 'text': ' der Truppe erarbeitet. Es gibt noch viel zu tun.'}, 
        {'timestamp': (73.74, 75.16), 'text': ' Die Truppe ist mir ans Herz gewachsen'}, 
        {'timestamp': (75.16, 77.02), 'text': ' und ich will diesen Job weitermachen, weil wir noch'}, 
        {'timestamp': (77.02, 79.22), 'text': ' nicht fertig sind, weil die Anforderungen,'}, 
        {'timestamp': (79.26, 80.78), 'text': ' die die Bedrohungslage an uns stellen,'}, 
        {'timestamp': (80.86, 83.0), 'text': ' einfach noch nicht erfüllt sind. Und das hat für mich'}
    ], 
    'duration': 83.0, 
    'sample_rate': 16000, 
    'status': 'success'
}

chunks = example_transcript["chunks"]
# chunks = transcript["chunks"]


# generate changes
llama = LLAMA()
modified_chunks = []
for chunk in chunks:
    generated_changes = llama.generate(chunk)
    modified_chunks.append(generated_changes)
    print("Generated changes: ", generated_changes)








