#!/usr/bin/env python3
"""
Simple server runner for Crop Analysis AI
"""
import uvicorn

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
    current_ip = get_current_ip()
    print("Starting Crop Analysis AI Server...")
    print("=" * 50)
    print(f"Local API: http://localhost:8000/analyze")
    print(f"Network API: http://{current_ip}:8000/analyze")
    print(f"Share this URL: http://{current_ip}:8000/analyze")
    print("=" * 50)
    
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)