# main.py
# Vision LFM2 API Backend - AI Image Analysis with LiquidAI models
#
# --- Dependencies ---
# Install with: uv sync
#
# --- Run Server ---
# uv run uvicorn main:app --reload

import torch
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware # Importujemy middleware CORS
from pydantic import BaseModel, HttpUrl
from typing import Optional
import io
import logging
import re
import os
import time
import threading
import fnmatch
from pathlib import Path
from fastapi.responses import JSONResponse
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Step 1: Model Configuration ---
logger.info("Model configuration - models will be loaded on demand.")

MODEL_ID_450M = "LiquidAI/LFM2-VL-450M"  # 450M model
MODEL_ID_1_6B = "LiquidAI/LFM2-VL-1.6B"  # 1.6B model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Paths and caching configuration
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
HF_CACHE_DIR = MODELS_DIR / "huggingface"
TF_CACHE_DIR = MODELS_DIR / "transformers"

# Ensure local model directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Force libraries to use local caches inside repo (ignore user-level ~/.cache)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(TF_CACHE_DIR)

# Reasonable fallback chat template for ChatML-style multimodal prompts
DEFAULT_CHAT_TEMPLATE = (
    "{{ bos_token }}"  # optional BOS
    "{% for message in messages %}"
    "{% set role = message['role'] %}"
    "{% set content = message['content'] %}"
    "{% if role == 'user' %}"
    "<|im_start|>user\n"
    "{% for item in content %}"
    "{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}"
    "{% endfor %}"
    "<|im_end|>\n"
    "{% elif role == 'assistant' %}"
    "<|im_start|>assistant\n"
    "{% for item in content %}"
    "{% if item['type'] == 'text' %}{{ item['text'] }}{% endif %}"
    "{% endfor %}"
    "<|im_end|>\n"
    "{% elif role == 'system' %}"
    "<|im_start|>system\n"
    "{% for item in content %}"
    "{% if item['type'] == 'text' %}{{ item['text'] }}{% endif %}"
    "{% endfor %}"
    "<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "<|im_start|>assistant"
)

# Global variables for currently loaded model
current_model_id = None
model = None
processor = None
downloaded_models = set()  # Track downloaded models

# Simple download status for frontend polling
download_status = {
    "in_progress": False,
    "model_id": None,
    "model_type": None,
    "phase": "idle",
    "error": None,
    "progress": 0,
    # extra telemetry
    "total_files": 0,
    "current_index": 0,
    "current_file": None,
    "total_bytes": 0,
    "downloaded_bytes_completed": 0,
    "downloaded_bytes_estimate": 0
}
download_thread = None

def _monitor_progress_bytes(model_id: str, total_bytes: int):
    """Continuously estimate progress by summing bytes in HF cache 'blobs' (including partial files)."""
    try:
        model_dir = _cache_folder_for_model(model_id)
        blobs_dir = model_dir / "blobs"
        last_bytes = 0
        stall_count = 0
        
        logger.info(f"Started progress monitor for {model_id}, expected {total_bytes:,} bytes")
        
        while download_status.get("in_progress", False):
            cur_bytes = 0
            file_count = 0
            try:
                if blobs_dir.exists():
                    for p in blobs_dir.rglob("*"):
                        if p.is_file():
                            try:
                                size = p.stat().st_size
                                cur_bytes += size
                                file_count += 1
                            except Exception:
                                pass
            except Exception:
                pass
            
            # Calculate progress percentage
            frac = max(0.0, min(1.0, cur_bytes / max(1, total_bytes)))
            prog = 10 + int(frac * 85)
            
            # Detect stalled downloads
            if cur_bytes == last_bytes:
                stall_count += 1
            else:
                stall_count = 0
            last_bytes = cur_bytes
            
            # Update status with detailed info
            download_status.update({
                "downloaded_bytes_estimate": cur_bytes,
                "download_rate_mbps": round(cur_bytes / (1024 * 1024), 1),
                "cache_files_count": file_count,
                "stall_detected": stall_count > 10  # 2+ seconds without progress
            })
            
            # Only update progress if it increased
            if prog > download_status.get("progress", 10):
                download_status.update({"progress": prog})
                logger.debug(f"Progress update: {prog}% ({cur_bytes:,}/{total_bytes:,} bytes, {file_count} files)")
            
            # Faster monitoring for better responsiveness
            time.sleep(0.2)
            
        logger.info(f"Progress monitor stopped for {model_id}")
    except Exception as e:
        logger.error(f"Progress monitor error for {model_id}: {e}")


def _cache_folder_for_model(model_id: str) -> Path:
    """Return HF cache folder path for given model id inside repo models cache."""
    # e.g. LiquidAI/LFM2-VL-450M -> models--LiquidAI--LFM2-VL-450M
    folder_name = f"models--{model_id.replace('/', '--')}"
    return HF_CACHE_DIR / folder_name


def _get_model_validation_info(model_id: str) -> dict:
    """Get comprehensive model validation information including file presence and sizes."""
    model_dir = _cache_folder_for_model(model_id)
    snapshots_dir = model_dir / "snapshots"
    
    result = {
        "exists": False,
        "is_complete": False,
        "total_size_mb": 0,
        "issues": [],
        "snapshot_path": None,
        "critical_files": {
            "config.json": False,
            "model_weights": False,
            "tokenizer": False,
            "processor_config": False
        }
    }
    
    if not snapshots_dir.exists():
        result["issues"].append("No snapshots directory found")
        return result
    
    result["exists"] = True
    
    # Expected minimum sizes (in MB) based on previous analysis
    expected_min_sizes = {
        MODEL_ID_450M: 800,  # ~902MB expected, require at least 800MB
        MODEL_ID_1_6B: 2800  # ~3.17GB expected, require at least 2.8GB
    }
    
    min_size_mb = expected_min_sizes.get(model_id, 100)
    
    try:
        # Find the most recent snapshot with files
        snapshots = [s for s in snapshots_dir.iterdir() if s.is_dir()]
        if not snapshots:
            result["issues"].append("No snapshot directories found")
            return result
        
        # Check each snapshot for completeness
        best_snapshot = None
        best_score = 0
        
        for snapshot in snapshots:
            score = 0
            total_size = 0
            files_found = {
                "config.json": False,
                "model_weights": False,
                "tokenizer": False,
                "processor_config": False
            }
            
            # Check for critical files
            if (snapshot / "config.json").exists():
                files_found["config.json"] = True
                score += 1
                
            if (snapshot / "processor_config.json").exists():
                files_found["processor_config"] = True
                score += 1
            
            # Check for model weights (.safetensors or .bin files)
            weight_files = list(snapshot.glob("*.safetensors")) + list(snapshot.glob("*.bin"))
            if weight_files:
                files_found["model_weights"] = True
                score += 2  # Weight files are more important
                
            # Check for tokenizer files
            tokenizer_files = list(snapshot.glob("tokenizer*.json")) + list(snapshot.glob("tokenizer*.model"))
            if tokenizer_files:
                files_found["tokenizer"] = True
                score += 1
            
            # Calculate total size
            try:
                for file_path in snapshot.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
            except Exception:
                pass
            
            total_size_mb = total_size / (1024 * 1024)
            
            # Prefer snapshots with higher scores and reasonable size
            if score > best_score or (score == best_score and total_size_mb > result["total_size_mb"]):
                best_snapshot = snapshot
                best_score = score
                result["critical_files"] = files_found.copy()
                result["total_size_mb"] = total_size_mb
        
        if best_snapshot:
            result["snapshot_path"] = str(best_snapshot)
            
            # Check if model appears complete
            has_config = result["critical_files"]["config.json"]
            has_weights = result["critical_files"]["model_weights"]
            has_tokenizer = result["critical_files"]["tokenizer"]
            size_ok = result["total_size_mb"] >= min_size_mb
            
            if not has_config:
                result["issues"].append("Missing config.json")
            if not has_weights:
                result["issues"].append("Missing model weight files")
            if not has_tokenizer:
                result["issues"].append("Missing tokenizer files")
            if not size_ok:
                result["issues"].append(f"Size too small: {result['total_size_mb']:.1f}MB < {min_size_mb}MB expected")
            
            # Model is complete if it has all critical files and reasonable size
            result["is_complete"] = has_config and has_weights and has_tokenizer and size_ok
        
    except Exception as e:
        result["issues"].append(f"Validation error: {str(e)}")
    
    return result

def _is_model_cached(model_id: str) -> bool:
    """Detect if model files appear to be present and complete in local cache."""
    validation_info = _get_model_validation_info(model_id)
    return validation_info["is_complete"]

def _delete_model_cache(model_id: str) -> bool:
    """Delete all cached files for a model. Returns True if successful."""
    try:
        model_dir = _cache_folder_for_model(model_id)
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)
            logger.info(f"Deleted model cache for {model_id}: {model_dir}")
            return True
        return True  # Already deleted
    except Exception as e:
        logger.error(f"Error deleting model cache for {model_id}: {e}")
        return False


def _refresh_downloaded_models() -> None:
    """Recompute downloaded_models set based on local cache; idempotent and cheap."""
    global downloaded_models
    detected = set()
    if _is_model_cached(MODEL_ID_450M):
        detected.add(MODEL_ID_450M)
    if _is_model_cached(MODEL_ID_1_6B):
        detected.add(MODEL_ID_1_6B)
    downloaded_models = detected

logger.info(f"Computing device: {DEVICE}")
logger.info("Server ready - models will be loaded on user demand.")

def load_model(model_id: str):
    """Load specified model and processor"""
    global model, processor, current_model_id, downloaded_models
    
    if current_model_id == model_id and model is not None:
        logger.info(f"Model {model_id} is already loaded")
        return
    
    # Release previous model from memory
    if model is not None:
        del model
        model = None
    if processor is not None:
        del processor
        processor = None
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info(f"Loading model: {model_id}")
    
    try:
        if model_id == MODEL_ID_450M:
            # 450M model - use CPU for better stability
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                cache_dir=str(HF_CACHE_DIR),
                local_files_only=True
            ).to("cpu")
        else:
            # 1.6B model - use device_map auto for GPU optimization
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=str(HF_CACHE_DIR),
                local_files_only=True
            )
        
        # Load config first to extract chat_template if provided by the repo
        try:
            cfg = AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=str(HF_CACHE_DIR),
                local_files_only=True,
            )
            chat_template = getattr(cfg, "chat_template", None)
        except Exception:
            chat_template = None

        if chat_template is not None:
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=str(HF_CACHE_DIR),
                local_files_only=True,
                chat_template=chat_template,
            )
        else:
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=str(HF_CACHE_DIR),
                local_files_only=True,
                chat_template=DEFAULT_CHAT_TEMPLATE,
            )
        
        current_model_id = model_id
        downloaded_models.add(model_id)
        logger.info(f"Model {model_id} loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {e}")
        raise

def _perform_download(model_id: str, model_type: str):
    """Background download worker with simple retries and status updates."""
    global download_status
    download_status.update({
        "in_progress": True,
        "model_id": model_id,
        "model_type": model_type,
        "phase": "starting",
        "error": None,
        "progress": 0
    })
    
    logger.info(f"Starting download of {model_id} ({model_type})")
    logger.info("If downloads fail repeatedly, consider manual download:")
    logger.info(f"  git clone https://huggingface.co/{model_id} {HF_CACHE_DIR}/models--{model_id.replace('/', '--')}")
    logger.info(f"  cd {HF_CACHE_DIR}/models--{model_id.replace('/', '--')} && git lfs pull")

    try:
        # Validate availability (may hit network briefly)
        from transformers import AutoConfig
        download_status.update({"phase": "validating", "progress": 5})
        AutoConfig.from_pretrained(model_id, trust_remote_code=True, cache_dir=str(TF_CACHE_DIR))
        AutoProcessor.from_pretrained(model_id, trust_remote_code=True, cache_dir=str(TF_CACHE_DIR))

        # Determine file list and total size to provide real progress
        api = HfApi()
        info = api.repo_info(model_id, repo_type="model")
        allow_patterns = ["*.json", "*.safetensors", "*.bin", "tokenizer.*", "*.model", "*.txt", "*.py"]

        def allowed(path: str) -> bool:
            for pat in allow_patterns:
                if fnmatch.fnmatch(path, pat):
                    return True
            return False

        siblings = getattr(info, "siblings", [])
        files = [s for s in siblings if allowed(getattr(s, "rfilename", ""))]
        total_bytes = sum(getattr(s, "size", 0) or 0 for s in files) or 1
        download_status.update({"total_files": len(files), "total_bytes": int(total_bytes)})

        downloaded_bytes = 0
        attempts = 3
        
        # Sort files by size (largest first) for better progress indication
        files_sorted = sorted(files, key=lambda x: getattr(x, "size", 0) or 0, reverse=True)
        
        # Log file info for debugging
        logger.info(f"Downloading {len(files_sorted)} files for {model_id}:")
        for i, s in enumerate(files_sorted[:5]):  # Log first 5 files
            fname = getattr(s, "rfilename", "")
            fsize = getattr(s, "size", 0) or 0
            logger.info(f"  {i+1}. {fname} ({fsize:,} bytes = {fsize/(1024*1024):.1f} MB)")
        
        download_status.update({
            "phase": f"downloading (attempt 1/{attempts})", 
            "progress": 10,
            "files_info": [{
                "name": getattr(s, "rfilename", ""),
                "size_mb": round((getattr(s, "size", 0) or 0) / (1024 * 1024), 1)
            } for s in files_sorted[:3]]  # Show top 3 largest files
        })
        
        monitor = threading.Thread(target=_monitor_progress_bytes, args=(model_id, total_bytes), daemon=True)
        monitor.start()

        # Download each file; update progress after each completes
        for idx, s in enumerate(files_sorted, start=1):
            rfilename = getattr(s, "rfilename", "")
            fsize = (getattr(s, "size", 0) or 0)
            fsize_mb = fsize / (1024 * 1024)
            
            logger.info(f"Starting download {idx}/{len(files_sorted)}: {rfilename} ({fsize_mb:.1f} MB)")

            for attempt in range(1, attempts + 1):
                try:
                    download_status.update({
                        "current_file": rfilename, 
                        "current_index": idx,
                        "current_file_size_mb": round(fsize_mb, 1),
                        "file_progress_pct": 0
                    })
                    download_status.update({
                        "phase": f"downloading {rfilename} ({idx}/{len(files_sorted)}, {fsize_mb:.1f}MB, attempt {attempt}/{attempts})",
                    })
                    hf_hub_download(
                        repo_id=model_id,
                        filename=rfilename,
                        cache_dir=str(HF_CACHE_DIR),
                        local_files_only=False,
                        resume_download=True,
                        # Better stability for poor connections
                        etag_timeout=120,  # Increase timeout for large files
                        user_agent="lfm2-vision/1.0 (stable-download)"
                    )
                    break
                except Exception as inner:
                    logger.warning(f"File {rfilename} attempt {attempt} failed: {inner}")
                    if attempt == attempts:
                        raise
                    # Exponential backoff with jitter for network issues
                    wait_time = min(30, 5 * (2 ** (attempt - 1)))  # Cap at 30 seconds
                    logger.info(f"Waiting {wait_time}s before retry {attempt + 1}/{attempts}")
                    time.sleep(wait_time)

            downloaded_bytes += fsize
            logger.info(f"Completed {rfilename} ({fsize_mb:.1f} MB) - {downloaded_bytes:,}/{total_bytes:,} bytes total")
            
            download_status.update({
                "downloaded_bytes_completed": int(downloaded_bytes),
                "file_progress_pct": 100,
                "completed_files": idx
            })
            
            # Calculate more accurate progress mapping
            frac_c = max(0.0, min(1.0, downloaded_bytes / total_bytes))
            prog_c = 10 + int(frac_c * 85)  # Map to 10-95% range
            
            # Consider estimate from monitor (real-time cache size)
            est = download_status.get("downloaded_bytes_estimate", 0)
            frac_e = max(0.0, min(1.0, est / total_bytes))
            prog_e = 10 + int(frac_e * 85)
            
            # Use the higher of the two estimates for better UX
            prog = max(prog_c, prog_e)
            
            # Add file-based milestone progress boost for large files
            if fsize > total_bytes * 0.5:  # If this file is >50% of total
                milestone_boost = min(5, int((fsize / total_bytes) * 10))
                prog = min(95, prog + milestone_boost)
                
            download_status.update({"progress": prog})
            logger.debug(f"Progress: {prog}% (completed: {prog_c}%, estimated: {prog_e}%)")

        downloaded_models.add(model_id)
        download_status.update({"phase": "finalizing", "progress": 95})
        
        # Final verification
        final_cache_size = sum(p.stat().st_size for p in _cache_folder_for_model(model_id).rglob("*") if p.is_file())
        logger.info(f"Model {model_id} downloaded successfully to {HF_CACHE_DIR}")
        logger.info(f"Total cache size: {final_cache_size:,} bytes ({final_cache_size/(1024*1024):.1f} MB)")
        
        download_status.update({
            "phase": "done", 
            "progress": 100,
            "final_size_mb": round(final_cache_size / (1024 * 1024), 1)
        })
    except Exception as e:
        logger.error(f"Error downloading model {model_id}: {e}", exc_info=True)
        error_msg = str(e)
        if "IncompleteMessage" in error_msg or "ReqwestMiddlewareError" in error_msg:
            error_msg = f"Network connection issue: {error_msg}. Try again or use manual download with git clone."
        elif "timeout" in error_msg.lower():
            error_msg = f"Download timeout: {error_msg}. Check your internet connection."
        elif "transfer.xethub.hf.co" in error_msg:
            error_msg = f"CDN issue: {error_msg}. HuggingFace CDN may be temporarily unavailable."
        download_status.update({"error": error_msg})
    finally:
        download_status["in_progress"] = False

# --- Step 2: FastAPI Application Initialization ---
app = FastAPI(
    title="Vision Model API",
    description="API for image analysis using LiquidAI/LFM2-VL models (450M or 1.6B)",
    version="1.0.0"
)

def clean_answer(text: str) -> str:
    """Remove chat role markers like user/assistant and common chat tags."""
    if not isinstance(text, str):
        return text
    cleaned = text
    # Remove common special tokens
    cleaned = re.sub(r"<\|\/?(system|user|assistant|observation)[^>]*\|>", "", cleaned, flags=re.IGNORECASE)
    # Remove lines that are just role labels or with trailing colon
    cleaned = re.sub(r"(?mi)^(system|user|assistant)\s*:?[\t ]*$", "", cleaned)
    # Remove leading role label with colon at start of text
    cleaned = re.sub(r"^(system|user|assistant)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    # Collapse excessive newlines/spaces
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip().strip('"')
    return cleaned

# --- Step 3: CORS Configuration ---
# This is a crucial step. We allow requests from any origin ('*').
# In production environment, it's recommended to limit this to specific frontend domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# --- Step 4: Data Models Definition (Pydantic) ---
class AnalysisRequest(BaseModel):
    image_url: HttpUrl
    question: str
    # Optional generation params
    max_new_tokens: Optional[int] = None
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None

class AnalysisResponse(BaseModel):
    question: str
    answer: str

class ModelSwitchRequest(BaseModel):
    model_type: str  # "450M", "1.6B"

class ModelStatusResponse(BaseModel):
    current_model: str
    available_models: list[str]
    model_loaded: bool
    downloaded_models: list[str]
    model_validation: dict  # Detailed validation info for each model

class ModelDownloadRequest(BaseModel):
    model_type: str  # "450M", "1.6B"

class ModelLoadRequest(BaseModel):
    model_type: str  # "450M", "1.6B"

class ModelValidationRequest(BaseModel):
    model_type: str  # "450M", "1.6B"

class ModelDeleteRequest(BaseModel):
    model_type: str  # "450M", "1.6B"

# --- Step 5: API Endpoints Definition ---
@app.post("/analyze/", response_model=AnalysisResponse)
async def analyze_image(request: AnalysisRequest):
    """
    Analyze image from given URL and answer the question.
    """
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model is unavailable or not properly loaded.")

    logger.info(f"Received request for image: {request.image_url}")

    try:
        response = requests.get(request.image_url, stream=True, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        logger.info(f"Image downloaded successfully, size: {image.size}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Image download error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {e}")
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Error during image processing: {e}")

    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": request.question}]}
    ]

    try:
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(image, text, return_tensors="pt")

        # Move inputs to appropriate device based on model
        if current_model_id == MODEL_ID_450M:
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
        else:
            # For 1.6B model, let device_map handle placement
            pass

        # Build generation parameters with sane defaults
        gen_kwargs = {
            "max_new_tokens": request.max_new_tokens if request.max_new_tokens is not None else 300,
            "do_sample": request.do_sample if request.do_sample is not None else False,
        }
        if gen_kwargs["do_sample"]:
            if request.temperature is not None:
                gen_kwargs["temperature"] = request.temperature
            if request.top_p is not None:
                gen_kwargs["top_p"] = request.top_p
        if request.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = request.repetition_penalty

        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)

        answer = processor.decode(output[0], skip_special_tokens=True).strip()
        answer = clean_answer(answer)
        logger.info(f"Generated answer: {answer}")
        return AnalysisResponse(question=request.question, answer=answer)
    except Exception as e:
        logger.error(f"Model response generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error occurred during model response generation: {e}")


@app.post("/analyze/file", response_model=AnalysisResponse)
async def analyze_image_file(
    file: UploadFile = File(...),
    question: str = Form(...),
    max_new_tokens: Optional[int] = Form(None),
    do_sample: Optional[bool] = Form(None),
    temperature: Optional[float] = Form(None),
    top_p: Optional[float] = Form(None),
    repetition_penalty: Optional[float] = Form(None),
):
    """Analyze uploaded image file and answer the question."""
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model is unavailable or not properly loaded.")

    logger.info(f"Received request with file: {file.filename}")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        logger.info(f"Image from file loaded successfully, size: {image.size}")
    except Exception as e:
        logger.error(f"Error loading image from file: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to load image from file: {e}")

    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]

    try:
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(image, text, return_tensors="pt")

        if current_model_id == MODEL_ID_450M:
            inputs = {k: v.to("cpu") for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_new_tokens if max_new_tokens is not None else 300,
            "do_sample": do_sample if do_sample is not None else False,
        }
        if gen_kwargs["do_sample"]:
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty

        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)

        answer = processor.decode(output[0], skip_special_tokens=True).strip()
        answer = clean_answer(answer)
        logger.info(f"Generated answer (file): {answer}")
        return AnalysisResponse(question=question, answer=answer)
    except Exception as e:
        logger.error(f"Model response generation error (file): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error occurred during model response generation: {e}")

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Vision Model API server is running. Frontend is available in a separate HTML file."}

@app.get("/model/status", response_model=ModelStatusResponse)
async def get_model_status():
    """Return current model status with detailed validation information"""
    # Refresh local cache detection each time
    _refresh_downloaded_models()
    current = "450M" if current_model_id == MODEL_ID_450M else ("1.6B" if current_model_id == MODEL_ID_1_6B else "None")
    downloaded_list = ["450M" if m == MODEL_ID_450M else "1.6B" for m in downloaded_models]
    
    # Get detailed validation info for each model
    validation_info = {
        "450M": _get_model_validation_info(MODEL_ID_450M),
        "1.6B": _get_model_validation_info(MODEL_ID_1_6B)
    }
    
    return ModelStatusResponse(
        current_model=current,
        available_models=["450M", "1.6B"],
        model_loaded=model is not None and processor is not None,
        downloaded_models=downloaded_list,
        model_validation=validation_info
    )

@app.post("/model/download")
async def download_model_endpoint(request: ModelDownloadRequest):
    """Download model without loading it into memory"""
    # Model type mapping
    model_mapping = {
        "450M": MODEL_ID_450M,
        "1.6B": MODEL_ID_1_6B
    }
    
    if request.model_type not in model_mapping:
        raise HTTPException(status_code=400, detail=f"Invalid model type. Available: {list(model_mapping.keys())}")
    
    model_id = model_mapping[request.model_type]
    
    if model_id in downloaded_models:
        return {"message": f"Model {request.model_type} is already downloaded"}
    
    # If a download is already running, return its status
    if download_status.get("in_progress"):
        return JSONResponse(status_code=202, content={"message": "Download already in progress", "status": download_status})
    
    try:
        # Start background download thread
        global download_thread
        download_thread = threading.Thread(target=_perform_download, args=(model_id, request.model_type), daemon=True)
        download_thread.start()
        return JSONResponse(status_code=202, content={"message": f"Started download for model {request.model_type}", "status": download_status})
        
    except Exception as e:
        logger.error(f"Error during model download: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to download model: {e}")


@app.get("/model/download/status")
async def download_status_endpoint():
    """Return current background download status."""
    return download_status

@app.post("/model/load")
async def load_model_endpoint(request: ModelLoadRequest):
    """Load model into memory (requires previous download)"""
    # Model type mapping
    model_mapping = {
        "450M": MODEL_ID_450M,
        "1.6B": MODEL_ID_1_6B
    }
    
    if request.model_type not in model_mapping:
        raise HTTPException(status_code=400, detail=f"Invalid model type. Available: {list(model_mapping.keys())}")
    
    model_id = model_mapping[request.model_type]
    
    if current_model_id == model_id and model is not None:
        return {"message": f"Model {request.model_type} is already loaded"}
    
    try:
        # Ensure cache detection is up-to-date
        _refresh_downloaded_models()
        if model_id not in downloaded_models:
            raise HTTPException(status_code=409, detail=f"Model {request.model_type} is not downloaded yet")
        load_model(model_id)
        return {"message": f"Model {request.model_type} loaded successfully", "model_id": model_id}
        
    except Exception as e:
        logger.error(f"Error during model loading: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

@app.post("/model/switch")
async def switch_model(request: ModelSwitchRequest):
    """Switch to selected model type (downloads and loads automatically)"""
    # Model type mapping
    model_mapping = {
        "450M": MODEL_ID_450M,
        "1.6B": MODEL_ID_1_6B
    }
    
    if request.model_type not in model_mapping:
        raise HTTPException(status_code=400, detail=f"Invalid model type. Available: {list(model_mapping.keys())}")
    
    model_id = model_mapping[request.model_type]
    
    if current_model_id == model_id and model is not None:
        return {"message": f"Model {request.model_type} is already loaded"}
    
    try:
        # If model is not downloaded, trigger download synchronously for simplicity
        if model_id not in downloaded_models:
            # Run the download worker synchronously here to surface errors immediately
            _perform_download(model_id, request.model_type)
            if download_status.get("error"):
                raise RuntimeError(download_status["error"]) 
        
        # Load model
        load_model(model_id)
        return {"message": f"Model switched to {request.model_type}", "model_id": model_id}
        
    except Exception as e:
        logger.error(f"Error during model switching: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {e}")

@app.post("/model/validate")
async def validate_model_endpoint(request: ModelValidationRequest):
    """Get detailed validation information for a specific model"""
    # Model type mapping
    model_mapping = {
        "450M": MODEL_ID_450M,
        "1.6B": MODEL_ID_1_6B
    }
    
    if request.model_type not in model_mapping:
        raise HTTPException(status_code=400, detail=f"Invalid model type. Available: {list(model_mapping.keys())}")
    
    model_id = model_mapping[request.model_type]
    
    try:
        validation_info = _get_model_validation_info(model_id)
        return {
            "model_type": request.model_type,
            "model_id": model_id,
            "validation": validation_info
        }
    except Exception as e:
        logger.error(f"Error during model validation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to validate model: {e}")

@app.post("/model/delete")
async def delete_model_endpoint(request: ModelDeleteRequest):
    """Delete a model from local cache"""
    # Model type mapping
    model_mapping = {
        "450M": MODEL_ID_450M,
        "1.6B": MODEL_ID_1_6B
    }
    
    if request.model_type not in model_mapping:
        raise HTTPException(status_code=400, detail=f"Invalid model type. Available: {list(model_mapping.keys())}")
    
    model_id = model_mapping[request.model_type]
    
    # Don't allow deleting currently loaded model
    if current_model_id == model_id and model is not None:
        raise HTTPException(status_code=409, detail=f"Cannot delete currently loaded model {request.model_type}. Switch to another model first.")
    
    try:
        success = _delete_model_cache(model_id)
        if success:
            # Update downloaded models set
            _refresh_downloaded_models()
            return {"message": f"Model {request.model_type} deleted successfully", "model_id": model_id}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to delete model cache")
    except Exception as e:
        logger.error(f"Error during model deletion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {e}")

