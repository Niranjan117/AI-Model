@echo off
echo Pushing Crop Analysis API to GitHub
echo ===================================

echo.
echo Only essential files will be pushed:
echo - ai_model.py
echo - api_server.py  
echo - run_server.py
echo - requirements.txt
echo - README.md
echo - Dockerfile
echo - render.yaml
echo.

echo Large files (images, models) are ignored via .gitignore
echo.

echo Commands to run:
echo git init
echo git add .
echo git commit -m "Crop Analysis API"
echo git remote add origin https://github.com/YOUR_USERNAME/crop-analysis-api.git
echo git push -u origin main
echo.

echo Create GitHub repo first at: https://github.com/new
echo Then run the commands above
echo.

pause