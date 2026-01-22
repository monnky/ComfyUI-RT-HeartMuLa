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

class HeartMuLaLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["HeartMuLa-oss-3B", "HeartMuLa-oss-7B"],),
                "quantization": (["none", "4bit", "8bit"], {"default": "4bit"}),
                "precision": (["auto", "fp16", "bf16", "fp32"], {"default": "auto"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "compile_model": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("HEARTMULA_MODEL", "HEARTMULA_TOKENIZER", "HEART_CODEC", "HEART_GEN_CONFIG")
    RETURN_NAMES = ("model", "tokenizer", "codec", "gen_config")
    FUNCTION = "load_model"
    CATEGORY = "HeartMuLa/Loaders"

    def load_model(self, model_name, quantization, precision, device, compile_model):
        LAST_LOADER_INFO.update({
            "model_name": model_name, 
            "quant": quantization, 
            "prec": precision, 
            "dev": device,
            "compiled": "ENABLED" if compile_model else "DISABLED"
        })
        
        mm.soft_empty_cache()
        log(f"Loading {model_name}...")
        
        base_path = os.path.join(folder_paths.models_dir, "HeartMuLa")
        model_path = os.path.join(base_path, model_name)
        codec_path = os.path.join(base_path, "HeartCodec-oss")
        
        gen_config_path = os.path.join(base_path, "gen_config.json")
        gen_config = HeartMuLaGenConfig(**json.load(open(gen_config_path, encoding="utf-8"))) if os.path.exists(gen_config_path) else HeartMuLaGenConfig()

        vocab_path = os.path.join(model_path, "tokenizer.json")
        if not os.path.exists(vocab_path): vocab_path = os.path.join(base_path, "tokenizer.json")
        tokenizer = Tokenizer.from_file(vocab_path)

        codec = HeartCodec.from_pretrained(codec_path).to(device)
        if quantization != "none" and device == "cuda": codec = codec.half()

        load_dtype = torch.float16 if precision == "fp16" or (precision=="auto" and device=="cuda") else torch.float32
        if precision == "bf16": load_dtype = torch.bfloat16

        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=load_dtype) if quantization == "4bit" else None
        model = HeartMuLa.from_pretrained(model_path, quantization_config=bnb_config, torch_dtype=load_dtype, device_map={"": 0} if device == "cuda" else "cpu")
        
        if compile_model and device == "cuda":
            log("🚀 Compiling model...")
            model = torch.compile(model, mode="reduce-overhead")
        
        return (model, tokenizer, codec, gen_config)

class HeartMuLaInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("HEARTMULA_MODEL",), "codec": ("HEART_CODEC",), "gen_config": ("HEART_GEN_CONFIG",)}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_info"
    CATEGORY = "HeartMuLa/Utils"
    def get_info(self, model, codec, gen_config):
        return (f"Model: {type(model).__name__}\nPrecision: {model.dtype}\nDevice: {model.device}",)