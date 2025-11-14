@echo off
echo Setting up ngrok for global API access
echo =======================================

echo.
echo Step 1: Download ngrok
echo Go to: https://ngrok.com/download
echo Download and extract ngrok.exe to this folder
echo.

echo Step 2: Get auth token
echo Go to: https://dashboard.ngrok.com/get-started/your-authtoken
echo Copy your authtoken
echo.

echo Step 3: Authenticate (run this once)
echo ngrok authtoken YOUR_TOKEN_HERE
echo.

echo Step 4: Start tunnel
echo ngrok http 8000
echo.

echo Your API will be available at: https://xxx.ngrok.io/analyze
echo This URL works from anywhere in the world!
echo.

pause