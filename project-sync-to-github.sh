#!/usr/bin/env bash
set -euo pipefail

# Go to your project folder
cd /home/ubuntu/alpha12_24   # <-- adjust to your actual path

# Stage all changes
git add -A

# Commit with timestamp (skip error if nothing to commit)
git commit -m "Server sync $(date +%F_%T)" || true

# Push to GitHub (force AWS as canonical)
git push origin main --force-with-lease
