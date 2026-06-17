#!/usr/bin/env python3
"""Quick download of Llama 3.2 1B Instruct Q4 GGUF"""
import requests
import os
import sys

url = 'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf'
headers = {'user-agent': 'Mozilla/5.0'}
dest = './models/Llama-3.2-1B-Instruct-Q4_K_M.gguf'

print(f"Downloading from {url}")
print(f"To: {dest}")

r = requests.get(url, headers=headers, stream=True)
print(f"Status: {r.status_code}")

if r.status_code != 200:
    print(f"Error: {r.text[:500]}")
    sys.exit(1)

total = int(r.headers.get('content-length', 0))
print(f"Size: {total / 1024 / 1024:.1f} MB")

with open(dest, 'wb') as f:
    downloaded = 0
    for chunk in r.iter_content(chunk_size=1024*1024):
        f.write(chunk)
        downloaded += len(chunk)
        print(f"\rDownloaded: {downloaded / 1024 / 1024:.1f} MB", end='', flush=True)

print(f"\nDone! File size: {os.path.getsize(dest) / 1024 / 1024:.1f} MB")
