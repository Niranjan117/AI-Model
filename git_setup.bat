@echo off
echo Git Setup and Push
echo ==================

echo Step 1: Configure Git
git config --global user.name "Niranjan117"
git config --global user.email "niranjan.kandpal100@gmail.com"

echo Step 2: Add all files
git add .

echo Step 3: Commit
git commit -m "Crop Analysis API - Initial commit"

echo Step 4: Push to GitHub
git branch -M main
git push -u origin main

echo Done!
pause