#!/bin/bash

# Vision LFM2 - Startup Script
# Starts both backend and frontend servers

set -e  # Exit on any error

# Create main log file with timestamp
LOG_FILE="vision-lfm2-$(date +"%Y%m%d-%H%M%S").log"
STARTUP_TIME=$(date '+%Y-%m-%d %H:%M:%S')

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to log without color but with timestamp
log_status() {
    local message="$1"
    log_with_timestamp "[INFO] $message"
    print_status "$message"
}

log_success() {
    local message="$1"
    log_with_timestamp "[SUCCESS] $message"
    print_success "$message"
}

log_warning() {
    local message="$1"
    log_with_timestamp "[WARNING] $message"
    print_warning "$message"
}

log_error() {
    local message="$1"
    log_with_timestamp "[ERROR] $message"
    print_error "$message"
}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Load configuration
if [ -f ".env" ]; then
    source .env
fi

BACKEND_HOST=${BACKEND_HOST:-127.0.0.1}
BACKEND_PORT=${BACKEND_PORT:-8000}
FRONTEND_PORT=${FRONTEND_PORT:-3000}

# Check if port is available
check_port() {
	local port=$1
	# Prefer ss if available
	if command -v ss >/dev/null 2>&1; then
		if ss -ltn "( sport = :$port )" 2>/dev/null | grep -q LISTEN; then
			return 1
		else
			return 0
		fi
	fi

	# Fallback to lsof
	if command -v lsof >/dev/null 2>&1; then
		if lsof -Pi :"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
			return 1
		else
			return 0
		fi
	fi

	# Fallback to netstat
	if command -v netstat >/dev/null 2>&1; then
		if netstat -tuln 2>/dev/null | awk '{print $4}' | grep -E "(:|\.)$port$" >/dev/null 2>&1; then
			return 1
		else
			return 0
		fi
	fi

	# Fallback to nc
	if command -v nc >/dev/null 2>&1; then
		if nc -z 127.0.0.1 "$port" >/dev/null 2>&1; then
			return 1
		else
			return 0
		fi
	fi

	# Final fallback: bash /dev/tcp connect test
	if (echo >/dev/tcp/127.0.0.1/$port) >/dev/null 2>&1; then
		return 1
	else
		return 0
	fi
}

# Find available port starting from given port
find_available_port() {
    local start_port=$1
    local port=$start_port
    while ! check_port $port; do
        port=$((port + 1))
        if [ $port -gt $((start_port + 100)) ]; then
            print_error "No available ports found in range $start_port-$((start_port + 100))"
            exit 1
        fi
    done
    echo $port
}

# Cleanup function
cleanup() {
    log_status "Shutting down servers..."
    log_with_timestamp "Cleanup initiated"
    
    if [ ! -z "$BACKEND_PID" ]; then
        log_with_timestamp "Killing backend PID: $BACKEND_PID"
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        log_with_timestamp "Killing frontend PID: $FRONTEND_PID"
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    # Wait a moment for graceful shutdown
    sleep 1
    
    log_success "Servers stopped"
    log_with_timestamp "Application shutdown completed"
    log_with_timestamp "=== End of session ==="
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo "🚀 Vision LFM2 - Starting Application"
echo "===================================="

# Initialize log file
log_with_timestamp "=== Vision LFM2 Application Startup ===="
log_with_timestamp "Started at: $STARTUP_TIME"
log_with_timestamp "Log file: $LOG_FILE"
log_with_timestamp "Working directory: $(pwd)"
log_with_timestamp "User: $(whoami)"
log_with_timestamp "System: $(uname -a)"
echo

# Check if installation was completed
if [ ! -f "backend/pyproject.toml" ]; then
    log_error "Backend not found. Did you run ./install.sh ?"
    exit 1
fi

if ! command -v uv &> /dev/null; then
    log_error "uv not found. Please run ./install.sh first"
    exit 1
fi

# Log system information
log_with_timestamp "Python version: $(python3 --version 2>/dev/null || echo 'Not found')"
log_with_timestamp "UV version: $(uv --version 2>/dev/null || echo 'Not found')"
if command -v nvidia-smi &>/dev/null; then
    log_with_timestamp "NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)"
else
    log_with_timestamp "NVIDIA GPU: Not detected"
fi

# Find available ports
log_status "Finding available ports..."
ACTUAL_BACKEND_PORT=$(find_available_port $BACKEND_PORT)
ACTUAL_FRONTEND_PORT=$(find_available_port $FRONTEND_PORT)

log_with_timestamp "Port configuration: Backend $BACKEND_PORT -> $ACTUAL_BACKEND_PORT, Frontend $FRONTEND_PORT -> $ACTUAL_FRONTEND_PORT"

if [ "$ACTUAL_BACKEND_PORT" != "$BACKEND_PORT" ]; then
    log_warning "Backend port $BACKEND_PORT is busy, using port $ACTUAL_BACKEND_PORT"
fi

if [ "$ACTUAL_FRONTEND_PORT" != "$FRONTEND_PORT" ]; then
    log_warning "Frontend port $FRONTEND_PORT is busy, using port $ACTUAL_FRONTEND_PORT"
fi

# Update frontend API URL
log_status "Configuring frontend to connect to backend..."
if [ -f "frontend/index.html" ]; then
    # Create backup
    cp frontend/index.html frontend/index.html.backup
    log_with_timestamp "Created frontend backup: frontend/index.html.backup"
    
    # Update API_BASE URL in frontend
    sed -i "s|const API_BASE = 'http://[^']*'|const API_BASE = 'http://$BACKEND_HOST:$ACTUAL_BACKEND_PORT'|g" frontend/index.html
    log_success "Frontend configured to use backend at http://$BACKEND_HOST:$ACTUAL_BACKEND_PORT"
else
    log_error "Frontend file not found!"
    exit 1
fi

# Start backend
log_status "Starting backend server on port $ACTUAL_BACKEND_PORT..."
log_with_timestamp "Backend command: PYTHONPATH=$(pwd)/backend uv run uvicorn main:app --host $BACKEND_HOST --port $ACTUAL_BACKEND_PORT --reload"
cd backend
PYTHONPATH=$(pwd) uv run uvicorn main:app --host $BACKEND_HOST --port $ACTUAL_BACKEND_PORT --reload > ../backend.log 2>&1 &
BACKEND_PID=$!
log_with_timestamp "Backend PID: $BACKEND_PID"
cd ..

# Wait a moment for backend to start
sleep 3

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    log_error "Backend failed to start. Check backend.log for details."
    log_with_timestamp "=== Backend Log Contents ==="
    cat backend.log | tee -a "$LOG_FILE"
    log_with_timestamp "=== End Backend Log ==="
    exit 1
fi

log_success "Backend server started (PID: $BACKEND_PID)"
log_with_timestamp "Backend URL: http://$BACKEND_HOST:$ACTUAL_BACKEND_PORT"

# Start frontend server
log_status "Starting frontend server on port $ACTUAL_FRONTEND_PORT..."
log_with_timestamp "Frontend command: uv run python -m http.server $ACTUAL_FRONTEND_PORT -d ../frontend"
cd backend
uv run python -m http.server $ACTUAL_FRONTEND_PORT -d ../frontend > ../frontend.log 2>&1 &
FRONTEND_PID=$!
log_with_timestamp "Frontend PID: $FRONTEND_PID"
cd ..

# Wait a moment for frontend to start
sleep 2

# Check if frontend started successfully
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    log_error "Frontend failed to start. Check frontend.log for details."
    log_with_timestamp "=== Frontend Log Contents ==="
    cat frontend.log | tee -a "$LOG_FILE"
    log_with_timestamp "=== End Frontend Log ==="
    cleanup
    exit 1
fi

log_success "Frontend server started (PID: $FRONTEND_PID)"
log_with_timestamp "Frontend URL: http://localhost:$ACTUAL_FRONTEND_PORT"

echo ""
print_success "🎉 Application is running!"
echo "=========================="
echo ""
echo "📱 Frontend: http://localhost:$ACTUAL_FRONTEND_PORT"
echo "🔧 Backend:  http://$BACKEND_HOST:$ACTUAL_BACKEND_PORT"
echo ""
echo "📖 Usage:"
echo "   1. Open http://localhost:$ACTUAL_FRONTEND_PORT in your browser"
echo "   2. Download a model using the 'Download' buttons"
echo "   3. Load the downloaded model using the 'Load' button"
echo "   4. Start analyzing images!"
echo ""
echo "💡 First time setup:"
echo "   - Download the 450M model (faster, ~1GB)"
echo "   - Or download the 1.6B model (better quality, ~3GB)"
echo ""
echo "🛑 Press Ctrl+C to stop both servers"
echo ""
echo "📋 Main log file: $LOG_FILE"
echo "🔧 Backend log: backend.log"
echo "🌐 Frontend log: frontend.log"
echo ""

log_with_timestamp "Application fully started and ready"

# Keep the script running and monitor the processes
monitor_start_time=$(date +%s)
while true; do
    current_time=$(date +%s)
    uptime_minutes=$(( (current_time - monitor_start_time) / 60 ))
    
    # Check if backend is still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        log_error "Backend process died unexpectedly after ${uptime_minutes} minutes"
        log_with_timestamp "Last backend log entries:"
        tail -n 10 backend.log | tee -a "$LOG_FILE"
        cleanup
        exit 1
    fi
    
    # Check if frontend is still running
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        log_error "Frontend process died unexpectedly after ${uptime_minutes} minutes"
        log_with_timestamp "Last frontend log entries:"
        tail -n 10 frontend.log | tee -a "$LOG_FILE"
        cleanup
        exit 1
    fi
    
    # Log periodic status every 5 minutes
    if [ $((uptime_minutes % 5)) -eq 0 ] && [ $uptime_minutes -gt 0 ] && [ $((current_time % 60)) -lt 5 ]; then
        log_with_timestamp "Status: Running for ${uptime_minutes} minutes (Backend PID: $BACKEND_PID, Frontend PID: $FRONTEND_PID)"
    fi
    
    sleep 5
done