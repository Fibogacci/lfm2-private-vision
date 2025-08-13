# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered image analysis application with a FastAPI backend and HTML/JavaScript frontend. The backend supports multiple LiquidAI vision models (LFM2-VL-450M and LFM2-VL-1.6B) to analyze images from URLs or uploaded files and answer questions about them. The application includes model download, management, and switching capabilities.

## Development Environment

- **Python Version**: 3.11+ (managed by uv)
- **Package Manager**: uv (modern Python package manager)
- **Project Structure**:
  - `backend/` - FastAPI server with vision model integration
  - `frontend/` - Single HTML page with Tailwind CSS and model management UI
  - `models/` - Local cache for downloaded models
    - `huggingface/` - HuggingFace Hub cache
    - `transformers/` - Transformers library cache
  - `install.sh` - Automated installation script
  - `start.sh` - Bash startup script (Linux/Unix)
  - `start_server.py` - Python startup script with port detection
  - `backend/run.py` - Advanced startup script with process management

## Installation and Startup

### Quick Start (Recommended)
```bash
# Install all dependencies and setup environment
./install.sh

# Start both backend and frontend servers
./start.sh
```

### Alternative Startup Methods

```bash
# Python startup with automatic port detection
python start_server.py

# Advanced process manager (supports --backend, --frontend, --browser flags)
python backend/run.py

# Manual backend startup
cd backend
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Common Commands

### Backend Development

```bash
cd backend

# Install dependencies
uv sync

# Run development server
uv run uvicorn main:app --reload

# Run server on specific host/port
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Install new dependencies
uv add package-name

# Enter virtual environment shell
uv shell
```

### Frontend Development

The frontend is a standalone HTML file that can be opened directly in a browser or served via HTTP server. It expects the backend to be running (default: `http://127.0.0.1:8000`).

## Architecture

### Backend (`backend/main.py`)
- **FastAPI application** with CORS middleware for cross-origin requests
- **Multi-model support**: LiquidAI/LFM2-VL-450M (CPU) and LFM2-VL-1.6B (GPU/auto device mapping)
- **Local model cache** management with fallback chat templates
- **Dynamic model downloading** with progress tracking and telemetry
- **GPU support** when available, intelligent device placement per model
- **Key endpoints**:
  - `POST /analyze/` - Analyzes image from URL with generation parameters
  - `POST /analyze/file` - Analyzes uploaded image file
  - `GET /model/status` - Returns current model status and downloaded models
  - `POST /model/download` - Downloads model without loading to memory
  - `GET /model/download/status` - Returns download progress with detailed telemetry
  - `POST /model/load` - Loads downloaded model into memory
  - `POST /model/switch` - Automatically downloads and loads specified model
  - `GET /` - Health check endpoint

### Frontend (`frontend/index.html`)
- **Tailwind CSS** for styling with dark mode support
- **Model management UI** with download progress and status tracking
- **Dual input support** for image URLs and file uploads
- **Advanced settings modal** with generation parameters (temperature, sampling, etc.)
- **Real-time progress tracking** for model downloads with byte-level telemetry
- **Error handling** and loading states for all operations

## Key Dependencies

- `fastapi` - Web framework with multipart support
- `torch` - PyTorch for model inference
- `transformers` - HuggingFace model integration with AutoProcessor
- `accelerate` - GPU acceleration and device mapping
- `huggingface-hub` - Model downloading and cache management
- `pillow` - Image processing
- `requests` - HTTP client for image downloads
- `uvicorn[standard]` - ASGI server for FastAPI

## Model Details

The application supports two LiquidAI vision-language models that can be downloaded and switched on-demand:

### LFM2-VL-450M
- **Size**: ~1GB
- **Inference**: CPU-optimized (torch.float32)
- **Use case**: Fast inference, lower memory usage
- **Device placement**: CPU only for stability

### LFM2-VL-1.6B  
- **Size**: ~3GB
- **Inference**: GPU-optimized with device_map="auto"
- **Use case**: Better accuracy, higher quality responses
- **Device placement**: Automatic GPU/CPU distribution

### Common Features
- **Prompt format**: ChatML-style with `<|im_start|>user\n<image>\n{question}<|im_end|><|im_start|>assistant\n`
- **Fallback chat template**: Built-in DEFAULT_CHAT_TEMPLATE when model config lacks template
- **Generation parameters**: Configurable max_new_tokens (default 300), sampling, temperature
- **Local caching**: Models stored in `models/huggingface/` with `local_files_only=True` loading

## Development Notes

### Application Behavior
- **Lazy model loading**: Models are loaded on-demand, not at startup
- **Background downloading**: Models download in background threads with progress tracking
- **Graceful degradation**: Application starts without models; download them via frontend UI
- **Memory management**: Only one model loaded at a time; switching models clears previous from memory
- **Cache-first loading**: Uses `local_files_only=True` after initial download

### Configuration
- **Local model storage**: All models cached in repository `models/` directory
- **Environment variables**: `HUGGINGFACE_HUB_CACHE` and `TRANSFORMERS_CACHE` point to local dirs
- **Port flexibility**: Startup scripts automatically find available ports if defaults are busy
- **Frontend API URL**: Automatically updated by startup scripts to match backend port

### Manual Model Download (Network Issues Fallback)
If automatic downloads fail due to unstable internet or CDN issues, use manual git clone:

```bash
# For LFM2-VL-450M
cd models/huggingface
git clone https://huggingface.co/LiquidAI/LFM2-VL-450M models--LiquidAI--LFM2-VL-450M
cd models--LiquidAI--LFM2-VL-450M && git lfs pull

# For LFM2-VL-1.6B
cd models/huggingface  
git clone https://huggingface.co/LiquidAI/LFM2-VL-1.6B models--LiquidAI--LFM2-VL-1.6B
cd models--LiquidAI--LFM2-VL-1.6B && git lfs pull
```

After manual download, restart the application. Models will be automatically detected and available for loading.

### Development Environment
- **CORS**: Configured to allow all origins (suitable for development)
- **Hot reload**: Backend supports `--reload` for development
- **Multi-language**: Polish language used in some UI elements and code comments
- **Error handling**: Comprehensive error handling with detailed logging for debugging

### Telemetry and Monitoring
- **Download progress**: Real-time progress tracking with file-level and byte-level metrics
- **Model status**: Frontend polls backend for model availability and loading status
- **Process monitoring**: Startup scripts monitor backend/frontend process health