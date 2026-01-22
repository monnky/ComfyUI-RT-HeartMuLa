import time
from datetime import datetime
# Ensure these are imported or defined for the colors to work
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"

class HeartMuLaTagsBuilder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "genre": (["pop", "rock", "jazz", "hip hop", "electronic", "folk", "cinematic", "none"], {"default": "pop"}),
                "vocal_type": (["female vocal", "male vocal", "instrumental", "choir", "none"], {"default": "female vocal"}),
                "mood": ("STRING", {"default": "emotional"}),
                "tempo": (["slow", "medium", "fast", "none"], {"default": "medium"}),
                "instruments": ("STRING", {"default": "piano, guitar"}),
            },
            "optional": {
                "additional_tags": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "build"
    CATEGORY = "HeartMuLa/Conditioning"

    def build(self, genre, vocal_type, mood, tempo, instruments, additional_tags):
        # --- THE IMMEDIATE TRIGGER ---
        # This executes BEFORE the model loader begins
        start_clock = datetime.now().strftime("%I:%M:%S %p")
        print(f"\n{ORANGE}--------------------process started [@ {start_clock}]--------------------{RESET}")
        
        tags = [t for t in [genre, vocal_type, mood, tempo, instruments, additional_tags] if t and t != "none"]
        return (", ".join(tags),)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # This forces the node to run every single time you hit Queue Prompt
        return float("NaN")