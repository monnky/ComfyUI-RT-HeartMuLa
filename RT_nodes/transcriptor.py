import os
import comfy.model_management as mm
from tqdm import tqdm
from transformers import pipeline
import folder_paths
from .utils import log

class HeartTranscriptorLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["HeartTranscriptor-oss"],),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }
    RETURN_TYPES = ("HEART_TRANSCRIPTOR",)
    RETURN_NAMES = ("transcriptor_pipe",)
    FUNCTION = "load"
    CATEGORY = "HeartMuLa/Loaders"
    
    def load(self, model_name, device):
        log("Loading Transcriptor Pipeline...")
        path = os.path.join(folder_paths.models_dir, "HeartMuLa", model_name)
        if not os.path.exists(path): 
            raise RuntimeError(f"Model not found: {path}")
            
        pipe = pipeline("automatic-speech-recognition", model=path, device=device, chunk_length_s=30)
        return (pipe,)

class HeartTranscriptorRunner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "transcriptor_pipe": ("HEART_TRANSCRIPTOR",),
                "audio": ("AUDIO",),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "transcribe"
    CATEGORY = "HeartMuLa/Transcriptor"
    
    def transcribe(self, transcriptor_pipe, audio):
        waveform = audio["waveform"].squeeze().cpu().numpy()
        sample_rate = audio["sample_rate"]
        
        # Convert stereo to mono if needed
        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(axis=0) 
        
        log(f"Transcribing {len(waveform)/sample_rate:.2f}s of generated audio...")
        full_text = []
        chunk_samples = int(30 * sample_rate)
        
        for i in tqdm(range(0, len(waveform), chunk_samples), desc="Transcription"):
            mm.throw_exception_if_processing_interrupted()
            chunk = waveform[i : i + chunk_samples]
            if len(chunk) < 100: break 
            
            result = transcriptor_pipe({"raw": chunk, "sampling_rate": sample_rate}, return_timestamps=True)
            full_text.append(result["text"])
            
        return (" ".join(full_text),)