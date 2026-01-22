import os
import sys
import logging
from dataclasses import dataclass

# Force Windows CMD to support ANSI colors
os.system('') 

ORANGE = "\033[38;5;208m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

# Global state to track settings for the CMD Table
LAST_LOADER_INFO = {
    "model_name": "Unknown",
    "quant": "Unknown",
    "prec": "Unknown",
    "dev": "Unknown"
}

# --- MODEL CLASS PLACEHOLDERS ---
HeartMuLa = None
HeartCodec = None

@dataclass
class HeartMuLaGenConfig:
    text_bos_id: int = 128000
    text_eos_id: int = 128001
    audio_eos_id: int = 8193
    empty_id: int = 0

def setup_logger():
    logging.getLogger("torchtune.modules.transformer").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

def log(text):
    prefix = f"[{ORANGE}R{YELLOW}T{RESET} HeartMuLa]"
    print(f"{prefix} {text}")

def print_generation_dashboard(data_list):
    print(f"\n{ORANGE}{'-'*70}{RESET}")
    print(f"{YELLOW}  RT HEARTMULA GENERATION REPORT{RESET}")
    print(f"{ORANGE}{'-'*70}{RESET}")
    for key, value in table_data_formatter(data_list):
        print(f"  {CYAN}{key:<15}{RESET} | {value}")
    print(f"{ORANGE}{'-'*70}{RESET}\n")

def table_data_formatter(data):
    # Helper to ensure data prints cleanly
    return data