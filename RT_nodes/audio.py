import os
import torchaudio
import folder_paths

class HeartMuLaAudioPreview:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "HeartMuLa/Audio"

    def save_audio(self, audio):
        # Professional standard ComfyUI pathing
        full_output_folder, filename, counter, subfolder, _ = \
            folder_paths.get_save_image_path("HeartMuLa", self.output_dir, audio["waveform"].shape[0], "")
        
        results = list()
        for (batch_number, waveform) in enumerate(audio["waveform"]):
            file = f"{filename.replace('%batch_num%', str(batch_number))}_{counter:05}_.wav"
            path = os.path.join(full_output_folder, file)
            
            # Ensure correct dimensions for torchaudio
            if waveform.ndim == 3: 
                waveform = waveform.squeeze(0) 
            
            torchaudio.save(path, waveform, audio["sample_rate"])
            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1

        return {"ui": {"audio": results}}