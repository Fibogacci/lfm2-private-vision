#!/usr/bin/env python3
"""
Startup script for LFM2-VL-450M Image Analysis Application

This script starts both the FastAPI backend and frontend HTTP server.
Requires uv package manager for dependency management.

Usage:
    python run.py              # Start both backend and frontend
    python run.py --backend     # Start only backend
    python run.py --frontend    # Start only frontend
    python run.py --help        # Show help
"""

import argparse
import subprocess
import sys
import time
import signal
import os
from pathlib import Path
import webbrowser
from typing import List, Optional

class Colors:
    """ANSI color codes for terminal output"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

class ServerManager:
    """Manages backend and frontend server processes"""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.backend_port = 8000
        self.frontend_port = 3000
        
    def print_colored(self, message: str, color: str = Colors.BLUE) -> None:
        """Print colored message to terminal"""
        print(f"{color}{message}{Colors.END}")
        
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available"""
        try:
            # Check uv
            result = subprocess.run(['uv', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                self.print_colored("❌ Error: 'uv' not found. Please install uv package manager.", Colors.RED)
                self.print_colored("   Install: curl -LsSf https://astral.sh/uv/install.sh | sh", Colors.YELLOW)
                return False
                
            # Check Python 3
            result = subprocess.run(['python3', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                self.print_colored("❌ Error: 'python3' not found.", Colors.RED)
                return False
                
            return True
        except FileNotFoundError:
            return False
    
    def check_model_files(self) -> bool:
        """Check if model files exist locally"""
        model_path = Path("models/LFM2-VL-450M/model/model.safetensors")
        processor_path = Path("models/LFM2-VL-450M/processor/tokenizer.json")
        
        if not model_path.exists():
            self.print_colored("⚠️  Warning: Model files not found at models/LFM2-VL-450M/", Colors.YELLOW)
            self.print_colored("   Run the download script or check the model path.", Colors.YELLOW)
            return False
            
        if not processor_path.exists():
            self.print_colored("⚠️  Warning: Processor files not found.", Colors.YELLOW)
            return False
            
        return True
    
    def install_dependencies(self) -> bool:
        """Install backend dependencies using uv"""
        self.print_colored("📦 Installing backend dependencies...", Colors.BLUE)
        
        try:
            os.chdir("backend")
            result = subprocess.run(['uv', 'sync'], check=True)
            os.chdir("..")
            self.print_colored("✅ Dependencies installed successfully", Colors.GREEN)
            return True
        except subprocess.CalledProcessError as e:
            self.print_colored(f"❌ Failed to install dependencies: {e}", Colors.RED)
            return False
        except Exception as e:
            self.print_colored(f"❌ Error: {e}", Colors.RED)
            return False
    
    def start_backend(self) -> Optional[subprocess.Popen]:
        """Start FastAPI backend server"""
        self.print_colored(f"🚀 Starting backend server on port {self.backend_port}...", Colors.BLUE)
        
        try:
            os.chdir("backend")
            process = subprocess.Popen([
                'uv', 'run', 'uvicorn', 'main:app',
                '--host', '0.0.0.0',
                '--port', str(self.backend_port),
                '--reload'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            os.chdir("..")
            
            # Wait a moment to check if process started successfully
            time.sleep(2)
            if process.poll() is None:
                self.print_colored(f"✅ Backend running at http://localhost:{self.backend_port}", Colors.GREEN)
                return process
            else:
                self.print_colored("❌ Backend failed to start", Colors.RED)
                return None
                
        except Exception as e:
            self.print_colored(f"❌ Error starting backend: {e}", Colors.RED)
            return None
    
    def start_frontend(self) -> Optional[subprocess.Popen]:
        """Start frontend HTTP server"""
        self.print_colored(f"🌐 Starting frontend server on port {self.frontend_port}...", Colors.BLUE)
        
        try:
            os.chdir("frontend")
            process = subprocess.Popen([
                'python3', '-m', 'http.server', str(self.frontend_port)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            os.chdir("..")
            
            # Wait a moment to check if process started successfully
            time.sleep(2)
            if process.poll() is None:
                self.print_colored(f"✅ Frontend running at http://localhost:{self.frontend_port}", Colors.GREEN)
                return process
            else:
                self.print_colored("❌ Frontend failed to start", Colors.RED)
                return None
                
        except Exception as e:
            self.print_colored(f"❌ Error starting frontend: {e}", Colors.RED)
            return None
    
    def open_browser(self) -> None:
        """Open browser to frontend URL"""
        try:
            url = f"http://localhost:{self.frontend_port}"
            self.print_colored(f"🌍 Opening browser at {url}...", Colors.BLUE)
            
            # Try to open with specific browsers, avoid text browsers
            import shutil
            browsers_to_try = ['google-chrome', 'chromium', 'firefox', 'brave-browser']
            
            browser_found = False
            for browser in browsers_to_try:
                if shutil.which(browser):
                    subprocess.Popen([browser, url], 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
                    browser_found = True
                    break
            
            if not browser_found:
                self.print_colored(f"⚠️  No GUI browser found. Open manually: {url}", Colors.YELLOW)
                
        except Exception as e:
            self.print_colored(f"⚠️  Could not open browser: {e}", Colors.YELLOW)
    
    def wait_for_interrupt(self) -> None:
        """Wait for Ctrl+C and handle graceful shutdown"""
        try:
            self.print_colored("\\n🎯 Application is running!", Colors.GREEN)
            self.print_colored("   • Backend:  http://localhost:8000", Colors.BLUE)
            self.print_colored("   • Frontend: http://localhost:3000", Colors.BLUE)
            self.print_colored("\\n📝 Press Ctrl+C to stop all servers", Colors.YELLOW)
            
            while True:
                time.sleep(1)
                # Check if any process died
                for i, process in enumerate(self.processes):
                    if process.poll() is not None:
                        self.print_colored(f"⚠️  Process {i+1} stopped unexpectedly", Colors.YELLOW)
                        
        except KeyboardInterrupt:
            self.print_colored("\\n🛑 Shutting down servers...", Colors.YELLOW)
    
    def cleanup(self) -> None:
        """Stop all running processes"""
        for i, process in enumerate(self.processes):
            if process.poll() is None:
                self.print_colored(f"🛑 Stopping process {i+1}...", Colors.YELLOW)
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.print_colored(f"⚡ Force killing process {i+1}...", Colors.RED)
                    process.kill()
        
        self.print_colored("✅ All servers stopped", Colors.GREEN)
    
    def run(self, backend_only: bool = False, frontend_only: bool = False, 
            open_browser: bool = False) -> None:
        """Main run method"""
        
        # Setup signal handler for graceful shutdown
        def signal_handler(sig, frame):
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Header
            self.print_colored("="*60, Colors.BOLD)
            self.print_colored("🤖 LFM2-VL-450M Image Analysis Application", Colors.BOLD)
            self.print_colored("="*60, Colors.BOLD)
            
            # Check dependencies
            if not self.check_dependencies():
                sys.exit(1)
            
            # Check model files
            model_available = self.check_model_files()
            if not model_available:
                self.print_colored("⚠️  Model files missing - backend may run in demo mode", Colors.YELLOW)
            
            # Install dependencies (always ensure env is synced)
            if not self.install_dependencies():
                sys.exit(1)
            
            # Start services
            if not frontend_only:
                backend_process = self.start_backend()
                if backend_process:
                    self.processes.append(backend_process)
                else:
                    sys.exit(1)
            
            if not backend_only:
                time.sleep(1)  # Give backend time to start
                frontend_process = self.start_frontend()
                if frontend_process:
                    self.processes.append(frontend_process)
                else:
                    if not frontend_only:
                        self.cleanup()
                    sys.exit(1)
            
            # Open browser
            if open_browser and not backend_only:
                time.sleep(2)
                self.open_browser()
            
            # Wait for interrupt
            self.wait_for_interrupt()
            
        except Exception as e:
            self.print_colored(f"❌ Unexpected error: {e}", Colors.RED)
            self.cleanup()
            sys.exit(1)
        finally:
            self.cleanup()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Start LFM2-VL-450M Image Analysis Application"
    )
    parser.add_argument(
        '--backend', 
        action='store_true', 
        help='Start only the backend server'
    )
    parser.add_argument(
        '--frontend', 
        action='store_true', 
        help='Start only the frontend server'
    )
    parser.add_argument(
        '--browser', 
        action='store_true', 
        help='Automatically open browser (disabled by default)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.backend and args.frontend:
        print("❌ Cannot specify both --backend and --frontend")
        sys.exit(1)
    
    manager = ServerManager()
    manager.run(
        backend_only=args.backend,
        frontend_only=args.frontend,
        open_browser=args.browser
    )

if __name__ == "__main__":
    main()