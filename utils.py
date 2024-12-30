from moviepy.editor import VideoFileClip, AudioFileClip
from dataclasses_json import dataclass_json
from typing import List, Optional, Tuple
from pyannote.audio import Pipeline
from dataclasses import dataclass
from pydub import AudioSegment
from yt_dlp import YoutubeDL
from tqdm import tqdm
import librosa
import os




def download_video(url, output_dir="downloads"):
    """
    Downloads a video from YouTube for research purposes.
    
    Args:
        url (str): YouTube URL
        output_dir (str): Directory to save the video
        
    Returns:
        str: Path to downloaded file or None if download fails
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract video ID from URL if available
    id = ""
    if "=" in url:
        id = url.split("=")[-1]
    else:
        id = url.split("/")[-1]


    # Configure options for video download
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Prefer MP4
        'outtmpl': os.path.join(output_dir, f'{id}'+'.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4'
        }]
    }
    
    try:
        # Create YouTube downloader object
        with YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            video_title = info['title']
            expected_path = os.path.join(output_dir, f"{video_title}.mp4")
            
            # Download the video
            ydl.download([url])
            
            if os.path.exists(expected_path):
                return expected_path
            else:
                # Try to find the file with a different name
                files = os.listdir(output_dir)
                for file in files:
                    if video_title in file and file.endswith('.mp4'):
                        return os.path.join(output_dir, file)
                        
            return None
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
    
def extract_audio_from_video(video_path, output_audio_path=None):
    """
    Extract audio from video file
    """
    if output_audio_path is None:
        output_audio_path = video_path.rsplit('.', 1)[0] + '.wav'
    
    # Load video audio and export as wav
    audio = AudioSegment.from_file(video_path, format="mp4")
    audio.export(output_audio_path, format="wav")

def replace_audio(
        original_audio_path:str, 
        new_audio_path:str, 
        start_time:float, 
        stop_time:float,
        output_path:str = "temp/replaced_audio.wav")->None:
    """
    Replace a segement in an audio with another audio.

    Args:
        original_audio_path (str): Path to the original audio file
        new_audio_path (str): Path to the new audio file
        output_path (str): Path to save the output audio file
        start_time (float): Start time in seconds
        stop_time (float): End time in seconds
    """

    # Load original audio
    # Load new audio
    # Replace segment in original audio with new audio at start_time and stop_time
    # Save the new audio file

def find_change(change: dict) -> List[Tuple[int, int]]:
    """
    find the difference between the old transcript and the new transcript, based off of the reported change.
    return the start and end index of the change in the old transcript and the new transcript.
    
    Returns:
        List[Tuple[int, int]]: List containing two tuples of (start, end) indices for old and new transcripts
    """
    old_transcript = change["old transcript"]
    new_transcript = change["new transcript"]
    old_change = change["change"]["old"]
    new_change = change["change"]["new"]

    # Find indices in old transcript
    old_start_idx = old_transcript.find(old_change)
    old_end_idx = old_start_idx + len(old_change)

    # Find indices in new transcript
    new_start_idx = new_transcript.find(new_change)
    new_end_idx = new_start_idx + len(new_change)

    return [(old_start_idx, old_end_idx), (new_start_idx, new_end_idx)]




























@dataclass_json
@dataclass
class SpeechSegment:
    start: float  # start time in seconds
    end: float    # end time in seconds
    speaker: str  # speaker label
    overlap: bool # whether this segment contains overlapped speech


# CURRENTLY ONLY AUDIO SPEAKER DIARIZATION --> AUDIO VISUAL SPEAKER DIARIZATION NEEDED !!!!!
class DiarizationPipeline:
    def __init__(self, auth_token: str):
        """
        Initialize the diarization pipeline.
        
        Args:
            auth_token (str): HuggingFace authentication token for pyannote.audio
        """

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token
        )
        
    def _convert_to_16khz(self, audio_path: str) -> str:
        """
        Convert audio to 16kHz mono if needed.
        Returns path to converted file or original if no conversion needed.
        """
        audio = AudioSegment.from_wav(audio_path)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
            
        # Convert sample rate if not 16kHz
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
            
        # Save temporary file if conversion was needed
        if audio.channels > 1 or audio.frame_rate != 16000:
            temp_path = audio_path.replace(".wav", "_converted.wav")
            audio.export(temp_path, format="wav")
            return temp_path
            
        return audio_path

    def process_audio(self,
                     audio_path: str, 
                     min_speakers: Optional[int] = None,
                     max_speakers: Optional[int] = None) -> List[SpeechSegment]:
        """
        Process audio file and return list of speech segments.
        
        Args:
            audio_path (str): Path to audio file
            min_speakers (int, optional): Minimum number of speakers
            max_speakers (int, optional): Maximum number of speakers
            
        Returns:
            List[SpeechSegment]: List of speech segments with timing and speaker information
        """

        # Convert audio if needed
        processed_path = self._convert_to_16khz(audio_path)
        
        # Configure pipeline parameters
        params = {}
        if min_speakers is not None:
            params['min_speakers'] = min_speakers
        if max_speakers is not None:
            params['max_speakers'] = max_speakers
        
        # Run diarization
        diarization = self.pipeline(processed_path, **params)
        
        # Convert results to list of segments
        segments = []
        for turn, _, speaker in tqdm(diarization.itertracks(yield_label=True)):
            segment = SpeechSegment(
                start=turn.start,
                end=turn.end,
                speaker=speaker,
                overlap=False
            )
            segments.append(segment)
        
        # Post-process to detect overlaps
        segments = self._merge_close_segments(segments)
        segments = self._detect_overlaps(segments)
        
        return segments
    
    def _detect_overlaps(self, segments: List[SpeechSegment]) -> List[SpeechSegment]:
        """
        Detect overlapping speech segments and mark them accordingly.
        """
        for i, seg1 in enumerate(segments):
            for seg2 in segments[i+1:]:
                # Check if segments overlap
                if seg1.start < seg2.end and seg2.start < seg1.end:
                    seg1.overlap = True
                    seg2.overlap = True
        
        return segments
    
    def _merge_close_segments(self, segments: List[SpeechSegment], 
                            threshold: float = 0.5) -> List[SpeechSegment]:
        """
        Merge segments from the same speaker that are close to each other.
        
        Args:
            segments: List of speech segments
            threshold: Time threshold in seconds to merge segments
        """
        if not segments:
            return segments

        merged = []
        current = segments[0]

        for next_seg in segments[1:]:
            if (next_seg.speaker == current.speaker and 
                next_seg.start - current.end <= threshold):
                # Merge segments
                current.end = next_seg.end
                current.overlap = current.overlap or next_seg.overlap
            else:
                merged.append(current)
                current = next_seg

        merged.append(current)
        return merged








@dataclass_json
@dataclass
class ExtractedSegment:
    id: str 
    video_path: str
    audio_path: str
    start_time: float
    end_time: float
    speaker: Optional[str] = None

class SegmentExtractor:
    def __init__(self, id:str, output_dir: str = "segments"):
        """
        Initialize the segment extractor.
        
        Args:
            output_dir (str): Directory where segments will be saved
        """
        self.id = id
        self.output_dir = output_dir+"_"+id
        self._ensure_output_dirs()
    
    def _ensure_output_dirs(self):
        """Create output directories if they don't exist."""
        os.makedirs(os.path.join(self.output_dir, "video"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "audio"), exist_ok=True)
    
    def _generate_filenames(self, 
                          start_time: float, 
                          end_time: float, 
                          speaker: Optional[str] = None) -> Tuple[str, str]:
        """Generate filenames for video and audio segments."""
        # Format timestamps for filename
        start_str = f"{start_time:.1f}".replace(".", "_")
        end_str = f"{end_time:.1f}".replace(".", "_")
        
        # Base filename
        base_name = f"segment_{start_str}_to_{end_str}"
        if speaker:
            base_name += f"_{speaker}"
            
        video_path = os.path.join(self.output_dir, "video", f"{base_name}.mp4")
        audio_path = os.path.join(self.output_dir, "audio", f"{base_name}.wav")
        
        return video_path, audio_path
    
    def extract_segment(self,
                       video_path: str,
                       audio_path: str,
                       start_time: float,
                       end_time: float,
                       speaker: Optional[str] = None) -> ExtractedSegment:
        """
        Extract a segment from both video and audio files.
        
        Args:
            video_path (str): Path to the original video file
            audio_path (str): Path to the original audio file
            start_time (float): Start time in seconds
            end_time (float): End time in seconds
            speaker (str, optional): Speaker identifier for filename
            
        Returns:
            ExtractedSegment: Object containing paths to extracted segments
        """
        # Generate output filenames
        video_output_path, audio_output_path = self._generate_filenames(
            start_time, end_time, speaker
        )
        
        try:
            # Extract video segment
            with VideoFileClip(video_path) as video:
                video_segment = video.subclip(start_time, end_time)
                video_segment.write_videofile(
                    video_output_path,
                    codec='libx264',
                    audio=False,
                    preset='ultrafast',  # Faster encoding, larger file size
                    threads=4
                )
            
            # Extract audio segment
            with AudioFileClip(audio_path) as audio:
                audio_segment = audio.subclip(start_time, end_time)
                audio_segment.write_audiofile(
                    audio_output_path,
                    fps=44100,  # CD quality
                    nbytes=2,    # 16-bit
                    buffersize=2000
                )
                
            return ExtractedSegment(
                id=self.id,
                video_path=video_output_path,
                audio_path=audio_output_path,
                start_time=start_time,
                end_time=end_time,
                speaker=speaker
            )
            
        except Exception as e:
            print(f"Error extracting segment: {e}")
            raise
    
    def extract_multiple_segments(self,
                                video_path: str,
                                audio_path: str,
                                segments: list) -> list[ExtractedSegment]:
        """
        Extract multiple segments at once.
        
        Args:
            video_path (str): Path to the original video file
            audio_path (str): Path to the original audio file
            segments (list): List of tuples (start_time, end_time, speaker)
            
        Returns:
            list[ExtractedSegment]: List of extracted segment information
        """
        extracted_segments = []
        
        for segment in segments:
            if len(segment) == 2:
                start_time, end_time = segment
                speaker = None
            else:
                start_time, end_time, speaker = segment
                
            extracted = self.extract_segment(
                video_path=video_path,
                audio_path=audio_path,
                start_time=start_time,
                end_time=end_time,
                speaker=speaker
            )
            extracted_segments.append(extracted)
            
        return extracted_segments










# id = "xgs5gOCpsAE"
# url = f"https://www.youtube.com/watch?v={id}"
# # path = download_video(url)
# # print("PATH: ", path)
# # extract_audio_from_video(f"downloads/{id}.mp4")


# # # Process audio file
# # pipeline = DiarizationPipeline(auth_token=os.getenv("HF_BTPW"))
# # segments = pipeline.process_audio(
# #     f"downloads/{id}.wav",
# #     min_speakers=2,
# #     max_speakers=2
# # )
# # print("Segments extracted")

# # Example segment
# segment = SpeechSegment(
#     start=12.0,          # 12
#     end=15.0,            # 28
#     speaker="Speaker2",
#     overlap=False
# )


# extractor = SegmentExtractor(id, output_dir="segments")
# segment = extractor.extract_segment(
#         video_path=f"downloads/{id}.mp4",
#         audio_path=f"downloads/{id}.wav",
#         start_time=segment.start,
#         end_time=segment.end,
#         speaker=segment.speaker
#     )


# segement = ExtractedSegment(
#     id,
#     video_path="segments_xgs5gOCpsAE/video/segment_12_0_to_15_0_Speaker2.mp4",
#     audio_path="segments_xgs5gOCpsAE/audio/segment_12_0_to_15_0_Speaker2.wav",
#     start_time=12.0,
#     end_time=15.0,
#     speaker="Speaker2"
# )
