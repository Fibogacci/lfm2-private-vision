#!/bin/bash

# LFM2 Private Vision - Installation Script for Linux/Raspberry Pi/Termux
# This script installs all dependencies needed to run the application

set -e  # Exit on any error

# Check for Termux flag
TERMUX_MODE=false
if [[ "$1" == "--termux" ]]; then
    TERMUX_MODE=true
    echo "🤖 Termux mode enabled - skipping system package installation"
fi

echo "🚀 LFM2 Private Vision - Installation Script"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Linux (skip check for Termux)
if [[ "$OSTYPE" != "linux-gnu"* && "$TERMUX_MODE" != "true" ]]; then
    print_error "This script is designed for Linux systems only"
    print_status "For Termux on Android, use: ./install.sh --termux"
    exit 1
fi

print_status "Checking system requirements..."

# Python management approach
if [[ "$TERMUX_MODE" == "true" ]]; then
    print_status "Termux mode: will use system Python with uv-managed dependencies"
else
    print_status "uv will download and manage Python 3.11 automatically"
    print_success "System Python check skipped - uv will handle Python installation"
fi

# Check for curl (needed for uv installation)
if ! command -v curl &> /dev/null; then
    if [[ "$TERMUX_MODE" == "true" ]]; then
        print_error "curl not found. Please install it with: pkg install curl"
        exit 1
    else
        print_status "Installing curl..."
        if command -v apt &> /dev/null; then
            sudo apt update && sudo apt install -y curl
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y curl
        elif command -v pacman &> /dev/null; then
            sudo pacman -S --noconfirm curl
        else
            print_error "Cannot install curl. Please install it manually"
            exit 1
        fi
    fi
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    print_status "Installing uv (modern Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    # Check if uv was installed successfully
    if ! command -v uv &> /dev/null; then
        print_error "Failed to install uv. Please try manual installation:"
        print_status "Visit: https://docs.astral.sh/uv/"
        exit 1
    fi
    
    print_success "uv installed successfully"
else
    print_success "uv is already installed"
fi

# Install backend dependencies
print_status "Installing Python dependencies..."
cd backend

if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found in backend directory"
    exit 1
fi

# Run uv sync to install dependencies
if [[ "$TERMUX_MODE" == "true" ]]; then
    # On Termux, use system Python but let uv manage venv
    if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
        print_error "Python not found. Please install with: pkg install python"
        exit 1
    fi
    print_status "Installing dependencies via uv (using Termux Python)..."
    uv sync
else
    # On other platforms, let uv download Python 3.11
    print_status "Installing Python 3.11 and dependencies via uv..."
    uv sync --python 3.11
fi
print_success "Backend dependencies installed"

cd ..

# Check for GPU support (optional)
print_status "Checking for GPU support..."
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected - CUDA acceleration will be available"
elif lspci | grep -i amd | grep -i vga &> /dev/null; then
    print_warning "AMD GPU detected - Limited GPU support may be available"
else
    print_warning "No dedicated GPU detected - Will use CPU inference (slower but works)"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating configuration file..."
    cat > .env << EOF
# Vision LFM2 Configuration
# Note: start.sh will automatically find available ports if these are busy
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
FRONTEND_PORT=3000
EOF
    print_success "Configuration file created (.env)"
fi

# Make scripts executable
chmod +x start.sh 2>/dev/null || true
chmod +x start_server.py 2>/dev/null || true

print_success "Installation completed successfully!"
echo ""
echo "🎉 Ready to use!"
echo "==============="
echo ""
echo "To start the application, run:"
echo "  ./start.sh"
echo ""
echo "Then open your browser to:"
echo "  http://localhost:3000"
echo ""
echo "The first time you use it, you'll need to download AI models"
echo "through the web interface (this will take a few minutes)."
echo ""