#!/usr/bin/env python3
"""
Starter script that finds available ports and launches the Vision LFM2 server
"""
import socket
import subprocess
import sys
import os
from pathlib import Path

def is_port_available(port):
    """Check if a port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
            return True
    except OSError:
        return False

def find_available_port(start_port=8000):
    """Find the first available port starting from start_port"""
    port = start_port
    while not is_port_available(port):
        port += 1
        if port > start_port + 100:  # Safety limit
            raise RuntimeError(f"No available ports found in range {start_port}-{start_port + 100}")
    return port

def update_frontend_api_url(port):
    """Update the frontend HTML file with the correct API URL"""
    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    
    if not frontend_path.exists():
        print(f"Warning: Frontend file not found at {frontend_path}")
        return
    
    # Read the file
    content = frontend_path.read_text(encoding='utf-8')
    
    # Update the API_BASE URL
    import re
    pattern = r"const API_BASE = '[^']+'"
    new_api_base = f"const API_BASE = 'http://127.0.0.1:{port}'"
    
    updated_content = re.sub(pattern, new_api_base, content)
    
    # Write back if changed
    if updated_content != content:
        frontend_path.write_text(updated_content, encoding='utf-8')
        print(f"Updated frontend API URL to use port {port}")

def main():
    """Main function"""
    print("🚀 Starting Vision LFM2 Server...")
    
    # Find available port
    try:
        port = find_available_port(8000)
        print(f"📡 Using port {port}")
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Update frontend
    try:
        update_frontend_api_url(port)
    except Exception as e:
        print(f"⚠️  Warning: Could not update frontend: {e}")
    
    # Change to backend directory
    backend_dir = Path(__file__).parent / "backend"
    
    print(f"🏃 Starting server on http://127.0.0.1:{port}")
    print("📄 Open frontend/index.html in your browser to use the application")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 60)
    
    # Start the server
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(backend_dir)
        
        subprocess.run([
            "uv", "run", "uvicorn", "main:app", 
            "--host", "127.0.0.1", 
            "--port", str(port), 
            "--reload"
        ], cwd=backend_dir, env=env)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except FileNotFoundError:
        print("❌ Error: 'uv' not found. Please install uv: https://docs.astral.sh/uv/")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()