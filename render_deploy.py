#!/usr/bin/env python3
"""
Render deployment setup for crop analysis API
"""

def create_render_files():
    """Create files needed for Render deployment"""
    
    # Create Dockerfile
    dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "run_server.py"]"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    # Create render.yaml
    render_yaml = """services:
  - type: web
    name: crop-analysis-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python run_server.py
    envVars:
      - key: PORT
        value: 8000"""
    
    with open('render.yaml', 'w') as f:
        f.write(render_yaml)
    
    print("Created Dockerfile")
    print("Created render.yaml")
    print("\nDEPLOYMENT STEPS:")
    print("1. Push code to GitHub")
    print("2. Connect GitHub to Render.com")
    print("3. Deploy - takes 5-10 minutes")
    print("4. Get URL: https://your-app.onrender.com/analyze")

if __name__ == "__main__":
    create_render_files()