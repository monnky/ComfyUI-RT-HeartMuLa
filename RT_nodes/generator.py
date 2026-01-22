import torch
import time
import numpy as np
import comfy.model_management as mm
from tqdm import tqdm
from datetime import datetime
from .utils import log, print_generation_dashboard, LAST_LOADER_INFO, ORANGE, YELLOW, RESET

class HeartMuLaGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("HEARTMULA_MODEL",), "tokenizer": ("HEARTMULA_TOKENIZER",),
                "codec": ("HEART_CODEC",), "gen_config": ("HEART_GEN_CONFIG",),
                "lyrics": ("STRING", {"multiline": True, "default": "[Verse]"}),
                "auto_clear_kv_cache": ("BOOLEAN", {"default": True}),
                "tags": ("STRING", {"multiline": True, "forceInput": True}),
                "duration_seconds": ("INT", {"default": 240, "min": 10, "max": 600, "step": 10}), 
                "cfg_scale": ("FLOAT", {"default": 1.7, "min": 1.0, "max": 5.0}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 200}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "HeartMuLa/Generation"

    def generate(self, model, tokenizer, codec, gen_config, lyrics, auto_clear_kv_cache, tags, duration_seconds, cfg_scale, temperature, top_k, seed):
        start_time_all = time.time()
        
        # 1. FULL TECHNICAL DASHBOARD
        print_generation_dashboard([
            ("MODEL", LAST_LOADER_INFO.get('model_name', 'Active')),
            ("QUANT", LAST_LOADER_INFO.get('quant', 'N/A')),
            ("PRECISION", LAST_LOADER_INFO.get('prec', 'N/A')),
            ("COMPILE", LAST_LOADER_INFO.get('compiled', 'DISABLED')),
            ("DEVICE", str(model.device)),
            ("FRAME RATE", "12.5 Hz (HeartCodec Spec)"),
            ("ARCHITECTURE", "Hierarchical (Global/Local)"),
            ("DURATION", f"{duration_seconds}s (Max)"),
            ("CFG SCALE", str(cfg_scale)),
            ("TEMP", str(temperature)),
            ("TOP_K", str(top_k)),
            ("SEED", str(seed)),
            ("TAGS", tags.replace("\n", " ").strip()[:60] + "...")
        ])

        mm.soft_empty_cache()
        device, dtype = model.device, model.dtype
        torch.manual_seed(seed)

        # 2. Preprocessing aligned with Paper Specs
        tags_processed = f"<tag>{tags.lower().strip()}</tag>"
        def _encode(txt): return tokenizer.encode(txt).ids if hasattr(tokenizer, "encode") and hasattr(tokenizer.encode(txt), "ids") else tokenizer.encode(txt, add_special_tokens=False)
        tags_ids, lyrics_ids = _encode(tags_processed), _encode(lyrics.lower().strip())

        for ids in [tags_ids, lyrics_ids]:
            if ids[0] != gen_config.text_bos_id: ids.insert(0, gen_config.text_bos_id)
            if ids[-1] != gen_config.text_eos_id: ids.append(gen_config.text_eos_id)

        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)
        parallel_number = 9 # HeartCodec Spec: 8 RVQ + 1 semantic
        
        # 3. Model & Cache Initialization
        bs_size = 2 if cfg_scale != 1.0 else 1
        if hasattr(model, "setup_caches"): model.setup_caches(bs_size)
        if auto_clear_kv_cache and hasattr(model, "reset_caches"):
            try: model.reset_caches()
            except: pass

        def _cfg_cat(t): return torch.cat([t.unsqueeze(0)] * bs_size, dim=0)
        
        tokens = torch.zeros([prompt_len, parallel_number], dtype=torch.long)
        tokens[:len(tags_ids), -1] = torch.tensor(tags_ids)
        tokens[len(tags_ids) + 1:, -1] = torch.tensor(lyrics_ids)
        
        # FIX: Removed semicolon and separated statements (E702)
        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens_mask[:, -1] = True
        
        prompt_tokens, prompt_tokens_mask = _cfg_cat(tokens).to(device), _cfg_cat(tokens_mask).to(device)
        muq_dim = 512 if not hasattr(model.config, "muq_dim") else model.config.muq_dim
        continuous_segment = _cfg_cat(torch.zeros([muq_dim], dtype=dtype)).to(device)
        prompt_pos = _cfg_cat(torch.arange(prompt_len, dtype=torch.long)).to(device)
        
        # 4. Sampling Loop with KV-Cache Alignment
        log(f"Sampling {duration_seconds}s at 12.5Hz...")
        frames = []
        max_frames = int(duration_seconds * 12.5)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
            curr_token = model.generate_frame(
                tokens=prompt_tokens, tokens_mask=prompt_tokens_mask, input_pos=prompt_pos,
                temperature=temperature, topk=top_k, cfg_scale=cfg_scale,
                continuous_segments=continuous_segment, starts=[len(tags_ids)]*bs_size,
            )
            frames.append(curr_token[0:1,])

            for i in tqdm(range(max_frames - 1), desc="HeartMuLa Generation"):
                if i % 8 == 0: mm.throw_exception_if_processing_interrupted()
                if i % 500 == 0: mm.soft_empty_cache() 

                padded = torch.ones((curr_token.shape[0], parallel_number), device=device, dtype=torch.long) * gen_config.empty_id
                
                # FIX: Removed semicolons and separated statements (E702)
                padded[:, :-1] = curr_token
                padded = padded.unsqueeze(1)
                
                mask = torch.ones_like(padded, device=device, dtype=torch.bool)
                mask[..., -1] = False
                
                curr_token = model.generate_frame(
                    tokens=padded, tokens_mask=mask,
                    input_pos=prompt_pos[..., -1:] + i + 1,
                    temperature=temperature, topk=top_k, cfg_scale=cfg_scale
                )
                
                if torch.any(curr_token[0:1, :] >= gen_config.audio_eos_id):
                    log("🎵 EOS Detected. Song concluded.")
                    break
                frames.append(curr_token[0:1,])

        # 5. Decoding and Fade-Out
        frames_tensor = torch.stack(frames).permute(1, 2, 0).squeeze(0).to(device)
        actual_duration = frames_tensor.shape[1] / 12.5 

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
            codec.to(device)
            wav = codec.detokenize(frames_tensor, duration=actual_duration, device=device)
            wav = wav.cpu().float()

            sample_rate = 48000
            
            # Start/End Fades for Paper-compliant noise reduction
            start_fade = int(0.1 * sample_rate)
            if wav.shape[-1] > start_fade:
                wav[..., :start_fade] *= torch.linspace(0.0, 1.0, start_fade)

            end_fade = int(2.5 * sample_rate)
            if wav.shape[-1] > end_fade:
                wav[..., -end_fade:] *= torch.linspace(1.0, 0.0, end_fade)

            mx = torch.max(torch.abs(wav))
            if mx > 0: wav = wav / mx
            
        if wav.ndim == 1: wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.ndim == 2: wav = wav.unsqueeze(0)
        
        total_time_seconds = time.time() - start_time_all
        total_minutes, total_seconds = divmod(int(total_time_seconds), 60)
        end_clock = datetime.now().strftime("%I:%M:%S %p")
        
        print(f"{ORANGE}--------------------processs completed [{end_clock}]--------------------{RESET}")
        print(f"{YELLOW}  Total Time: {total_minutes:02}:{total_seconds:02} minutes{RESET}\n")
        
        return ({"waveform": wav, "sample_rate": 48000},)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("NaN")