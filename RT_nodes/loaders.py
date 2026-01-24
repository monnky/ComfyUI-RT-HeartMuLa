import os
import torch
import json
import folder_paths
import comfy.model_management as mm
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from tokenizers import Tokenizer
from .utils import log, LAST_LOADER_INFO, HeartMuLaGenConfig

# Model Class Placeholders
HeartMuLa = None
HeartCodec = None

try:
    from ..heartlib.heartmula.modeling_heartmula import HeartMuLa as HML_Class
    from ..heartlib.heartmula.configuration_heartmula import HeartMuLaConfig
    from ..heartlib.heartcodec.modeling_heartcodec import HeartCodec as HCD_Class
    from ..heartlib.heartcodec.configuration_heartcodec import HeartCodecConfig

    HeartMuLa = HML_Class
    HeartCodec = HCD_Class

    AutoConfig.register("heartmula", HeartMuLaConfig)
    AutoModel.register(HeartMuLaConfig, HeartMuLa)
    AutoConfig.register("heartcodec", HeartCodecConfig)
    AutoModel.register(HeartCodecConfig, HeartCodec)
except ImportError as e:
    log(f"❌ Library Import Error: {e}")


##################################################################################################
class HeartMuLaLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    [
                        "HeartMuLa-oss-3B", 
                        "HeartMuLa-RL-oss-3B-20260123",  # New RL Model
                        # "HeartMuLa-oss-7B",            # <--- COMMENTED OUT
                    ],
                ),
                # New Dropdown to select the specific Codec version
                "codec_name": (
                    [
                        "HeartCodec-oss-20260123",       # New 2026 Codec (Default)
                        "HeartCodec-oss",                # Old Codec
                    ], 
                    {"default": "HeartCodec-oss-20260123"}
                ),
                "quantization": (["none", "4bit", "8bit"], {"default": "4bit"}),
                "mula_precision": (["auto", "fp16", "bf16", "fp32"], {"default": "bf16"}),
                "codec_precision": (["auto", "fp16", "bf16", "fp32"], {"default": "fp32"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "compile_model": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("HEARTMULA_MODEL", "HEARTMULA_TOKENIZER", "HEART_CODEC", "HEART_GEN_CONFIG")
    RETURN_NAMES = ("model", "tokenizer", "codec", "gen_config")
    FUNCTION = "load_model"
    CATEGORY = "HeartMuLa/Loaders"

    # Updated signature to accept 'codec_name'
    def load_model(self, model_name, codec_name, quantization, mula_precision, codec_precision, device, compile_model):
        # 1. SAFETY CHECK: Prevent "NoneType" crash if libraries are missing
        if HeartCodec is None:
            raise RuntimeError("❌ CRITICAL ERROR: Required libraries are missing! Please run 'pip install -r requirements.txt' in the ComfyUI-RT-HeartMuLa folder. (Missing: vector-quantize-pytorch)")

        # 2. Update Info Block
        LAST_LOADER_INFO.update({
            "model_name": model_name,
            "codec_name": codec_name,
            "quant": quantization, 
            "prec": f"{mula_precision}/{codec_precision}", 
            "dev": device,
            "compiled": "ENABLED" if compile_model else "DISABLED"
        })
        
        mm.soft_empty_cache()
        log(f"Loading {model_name} using {codec_name} (Mula: {mula_precision}, Codec: {codec_precision})...")
        
        base_path = os.path.join(folder_paths.models_dir, "HeartMuLa")
        model_path = os.path.join(base_path, model_name)
        
        # 3. CHANGED: Use the selected codec_name variable
        codec_path = os.path.join(base_path, codec_name)
        
        gen_config_path = os.path.join(base_path, "gen_config.json")
        gen_config = HeartMuLaGenConfig(**json.load(open(gen_config_path, encoding="utf-8"))) if os.path.exists(gen_config_path) else HeartMuLaGenConfig()

        vocab_path = os.path.join(model_path, "tokenizer.json")
        if not os.path.exists(vocab_path): vocab_path = os.path.join(base_path, "tokenizer.json")
        tokenizer = Tokenizer.from_file(vocab_path)

        def get_dtype(p):
            if p == "bf16": return torch.bfloat16
            if p == "fp16": return torch.float16
            if p == "fp32": return torch.float32
            return torch.float16 if device == "cuda" else torch.float32

        mula_dtype = get_dtype(mula_precision)
        codec_dtype = get_dtype(codec_precision)

        # Load Codec
        codec = HeartCodec.from_pretrained(codec_path).to(device).to(codec_dtype)
        codec.eval() 

        # Load HeartMuLa
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=mula_dtype
        ) if quantization == "4bit" else None
        
        model = HeartMuLa.from_pretrained(
            model_path, 
            quantization_config=bnb_config, 
            torch_dtype=mula_dtype, 
            device_map={"": 0} if device == "cuda" else "cpu"
        )
        model.eval()

        if compile_model and device == "cuda":
            log("🚀 Compiling model...")
            model = torch.compile(model, mode="reduce-overhead")
        
        return (model, tokenizer, codec, gen_config)
    
###############################################################################################################



class HeartMuLaInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("HEARTMULA_MODEL",), "codec": ("HEART_CODEC",), "gen_config": ("HEART_GEN_CONFIG",)}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_info"
    CATEGORY = "HeartMuLa/Utils"
    def get_info(self, model, codec, gen_config):
        return (f"Model: {type(model).__name__} ({model.dtype})\nCodec: {type(codec).__name__} ({codec.dtype})\nDevice: {model.device}",)