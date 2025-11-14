#!/usr/bin/env python3
"""
Get your computer's IP address for API access
"""
import socket

def get_local_ip():
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "Unable to determine IP"

if __name__ == "__main__":
    ip = get_local_ip()
    print("=" * 50)
    print("CROP ANALYSIS API - NETWORK ACCESS")
    print("=" * 50)
    print(f"Your IP Address: {ip}")
    print(f"API URL: http://{ip}:8000/analyze")
    print(f"For websites to use: http://{ip}:8000/analyze")
    print("=" * 50)