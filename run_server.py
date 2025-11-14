#!/usr/bin/env python3
"""
Simple server runner for Crop Analysis AI
"""
import uvicorn
import os

def get_current_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "Unable to get IP"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    current_ip = get_current_ip()
    print("Starting Crop Analysis AI Server...")
    print("=" * 50)
    print(f"Local API: http://localhost:{port}/analyze")
    print(f"Network API: http://{current_ip}:{port}/analyze")
    print(f"Share this URL: http://{current_ip}:{port}/analyze")
    print("=" * 50)
    
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=False)