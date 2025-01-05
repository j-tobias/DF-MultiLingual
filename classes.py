from dataclasses_json import dataclass_json
from dataclasses import dataclass


@dataclass_json
@dataclass
class DFMSample:


    # Identifiers
    sample_id:str = None
    object_id:str = None

    # Metadata
    dataset_root:str = None

    # Indicators
    step:int = None    # Step in the process (1, 2, 3, 4, ...)
    
    # Data
    original_audio_path:str = None
    transcript:dict = None
    change:dict = None
    indicies:list = None