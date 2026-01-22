import os
import sys

# Add the current directory to sys.path to ensure relative imports work
current_path = os.path.dirname(os.path.abspath(__file__))
if current_path not in sys.path:
    sys.path.insert(0, current_path)

# Import the nodes from your modular RT_nodes folder
from .RT_nodes.loaders import HeartMuLaLoader, HeartMuLaInfo
from .RT_nodes.generator import HeartMuLaGenerator
from .RT_nodes.audio import HeartMuLaAudioPreview
from .RT_nodes.transcriptor import HeartTranscriptorLoader, HeartTranscriptorRunner
from .RT_nodes.conditioning import HeartMuLaTagsBuilder
from .RT_nodes.utils import setup_logger, ORANGE, YELLOW, RESET

# Initialize the professional log silencer and ANSI color support
setup_logger()

# Mapping for ComfyUI backend
NODE_CLASS_MAPPINGS = {
    "HeartMuLaLoader": HeartMuLaLoader,
    "HeartMuLaInfo": HeartMuLaInfo,
    "HeartMuLaGenerator": HeartMuLaGenerator,
    "HeartMuLaAudioPreview": HeartMuLaAudioPreview,
    "HeartTranscriptorLoader": HeartTranscriptorLoader,
    "HeartTranscriptorRunner": HeartTranscriptorRunner,
    "HeartMuLaTagsBuilder": HeartMuLaTagsBuilder
}

# Mapping for ComfyUI Frontend Display Names
NODE_DISPLAY_NAME_MAPPINGS = {
    "HeartMuLaLoader": "RT HeartMuLa Loader",
    "HeartMuLaInfo": "RT HeartMuLa Info Dashboard",
    "HeartMuLaGenerator": "RT HeartMuLa Sampler",
    "HeartMuLaAudioPreview": "RT HeartMuLa Preview",
    "HeartTranscriptorLoader": "RT HeartMuLa Transcriptor Loader",
    "HeartTranscriptorRunner": "RT HeartMuLa Transcribe",
    "HeartMuLaTagsBuilder": "RT HeartMuLa Tags Builder"
}

# ============================================================
# RESTORED FULL STARTUP LOG (WITH ALL FEATURES)
# ============================================================
print("\n" + "=" * 60)
print(f"ComfyUI [{ORANGE}R{YELLOW}T{RESET} HeartMuLa]")
print("=" * 60)
print("Version: 1.0.6 (Modular Master)")
print(f"Loaded {len(NODE_CLASS_MAPPINGS)} nodes:")
for name, cls in NODE_CLASS_MAPPINGS.items():
    display = NODE_DISPLAY_NAME_MAPPINGS.get(name, name)
    print(f"  - {display} ({name})")

print("\nFeatures:")
print("  - Log Silencer: Suppresses repetitive cache warnings")
print("  - Robust Caching: Auto-correction for batch size changes")
print("  - Official HeartLib Preprocessing Active")
print("  - Native Quantization: 4-bit/8-bit via BitsAndBytes")
print("  - Generation Dashboard: Visual table report enabled")
print("=" * 60 + "\n")


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
