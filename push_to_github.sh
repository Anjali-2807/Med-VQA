#!/bin/bash
# Shell helper script to push your upgraded KG-CMI project to your personal GitHub repository

if [ -z "$1" ]; then
  echo "Usage: ./push_to_github.sh <YOUR_GITHUB_REPO_URL>"
  echo "Example: ./push_to_github.sh https://github.com/anjali-tiwari/KG-CMI-MedVQA.git"
  exit 1
fi

REPO_URL=$1

echo "1. Adding all project files (including upgraded R-GCN module and Colab notebook)..."
git add .

echo "2. Committing changes..."
git commit -m "Minor Project: Integrated R-GCN KGE module and Google Colab training notebook"

echo "3. Updating git remote origin to $REPO_URL..."
git remote set-url origin "$REPO_URL"

echo "4. Pushing code to your GitHub repository..."
git push -u origin main || git push -u origin master

echo "✅ Successfully pushed to your GitHub repository!"
