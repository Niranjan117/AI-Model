#!/usr/bin/env python3
"""
Network connectivity solutions for crop analysis API
"""
import socket
import subprocess
import platform

def get_all_network_interfaces():
    """Get all available network interfaces and IPs"""
    interfaces = []
    
    try:
        if platform.system() == "Windows":
            result = subprocess.run(['ipconfig'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            current_adapter = ""
            
            for line in lines:
                if "adapter" in line.lower():
                    current_adapter = line.strip()
                elif "IPv4 Address" in line:
                    ip = line.split(':')[1].strip()
                    if not ip.startswith('127.'):
                        interfaces.append({
                            'adapter': current_adapter,
                            'ip': ip,
                            'type': 'WiFi' if 'wireless' in current_adapter.lower() else 'Ethernet'
                        })
        else:
            # Linux/Mac
            result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
            ips = result.stdout.strip().split()
            for ip in ips:
                if not ip.startswith('127.'):
                    interfaces.append({'ip': ip, 'type': 'Network'})
                    
    except Exception as e:
        print(f"Error getting interfaces: {e}")
    
    return interfaces

def create_ngrok_tunnel():
    """Instructions for ngrok tunnel (public internet access)"""
    return {
        "solution": "ngrok",
        "description": "Create public tunnel for internet access",
        "steps": [
            "1. Download ngrok from https://ngrok.com/",
            "2. Install and authenticate: ngrok authtoken YOUR_TOKEN",
            "3. Run: ngrok http 8000",
            "4. Use the https://xxx.ngrok.io URL for global access"
        ],
        "benefits": ["Works from anywhere", "No network restrictions", "HTTPS secure"]
    }

def create_hotspot_solution():
    """Mobile hotspot solution"""
    return {
        "solution": "mobile_hotspot",
        "description": "Use mobile hotspot for consistent network",
        "steps": [
            "1. Enable mobile hotspot on your phone",
            "2. Connect your computer to the hotspot",
            "3. Connect website user's device to same hotspot",
            "4. Both will be on same network with consistent IPs"
        ],
        "benefits": ["Same network guaranteed", "Consistent IP", "No external dependencies"]
    }

def create_dynamic_dns_solution():
    """Dynamic DNS solution"""
    return {
        "solution": "dynamic_dns",
        "description": "Use domain name instead of IP",
        "steps": [
            "1. Sign up for free DDNS service (No-IP, DuckDNS)",
            "2. Create domain like: yourapi.ddns.net",
            "3. Install DDNS client to auto-update IP",
            "4. Use domain name instead of IP in website"
        ],
        "benefits": ["Fixed domain name", "Auto IP updates", "Professional looking"]
    }

def show_all_solutions():
    """Display all network solutions"""
    print("NETWORK CONNECTIVITY SOLUTIONS")
    print("=" * 50)
    
    # Current network info
    interfaces = get_all_network_interfaces()
    print("Current Network Interfaces:")
    for i, interface in enumerate(interfaces, 1):
        print(f"{i}. {interface.get('type', 'Network')}: {interface['ip']}")
    
    print("\n" + "=" * 50)
    
    # Solution 1: ngrok
    ngrok = create_ngrok_tunnel()
    print(f"\nSOLUTION 1: {ngrok['solution'].upper()}")
    print(ngrok['description'])
    for step in ngrok['steps']:
        print(f"  {step}")
    print(f"Benefits: {', '.join(ngrok['benefits'])}")
    
    # Solution 2: Mobile hotspot
    hotspot = create_hotspot_solution()
    print(f"\nSOLUTION 2: {hotspot['solution'].upper()}")
    print(hotspot['description'])
    for step in hotspot['steps']:
        print(f"  {step}")
    print(f"Benefits: {', '.join(hotspot['benefits'])}")
    
    # Solution 3: Dynamic DNS
    ddns = create_dynamic_dns_solution()
    print(f"\nSOLUTION 3: {ddns['solution'].upper()}")
    print(ddns['description'])
    for step in ddns['steps']:
        print(f"  {step}")
    print(f"Benefits: {', '.join(ddns['benefits'])}")
    
    print("\n" + "=" * 50)
    print("RECOMMENDED: Use ngrok for easiest setup")
    print("URL will be: https://abc123.ngrok.io/analyze")

if __name__ == "__main__":
    show_all_solutions()