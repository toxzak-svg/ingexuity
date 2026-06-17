"""
GGUF Re-Quantization Script using llama.cpp

This script uses llama.cpp's tools to re-quantize your existing GGUF model
to a smaller format (Q2_K or Q3_K) which can significantly reduce size.

Requirements:
    pip install llama-cpp-python

Usage:
    python quantize_gguf_requantize.py
"""

import subprocess
import os
import shutil

# Configuration
INPUT_GGUF = "./Llama-3.2-1B-Instruct-Q4_K_M.gguf"
OUTPUT_DIR = "./requantized"

# llama.cpp quantize options:
# Q2_K - 2-bit (smallest, quality loss)
# Q3_K - 3-bit
# Q4_K_M - 4-bit medium (your current)
# Q5_K - 5-bit
# Q6_K - 6-bit (nearly lossless)

TARGET_QUANT = "Q2_K"  # Most aggressive


def re_quantize_gguf():
    print(f"Re-quantizing {INPUT_GGUF} to {TARGET_QUANT}...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"Llama-3.2-1B_{TARGET_QUANT}.gguf")
    
    # Get the llama.cpp quantize binary path
    try:
        from llama_cpp import Llama GGUFReader
        
        print("Using llama-cpp-python for re-quantization...")
        print("This script shows the approach - actual re-quantization requires llama.cpp CLI")
        print("\nAlternative: Use llama.cpp CLI directly:")
        print(f"  ./quantize {INPUT_GGUF} {output_path} {TARGET_QUANT}")
        
    except ImportError:
        print("llama-cpp-python not installed.")
        print("\nTo install: pip install llama-cpp-python")
        print("\nOr use llama.cpp CLI (build from https://github.com/ggerganov/llama.cpp):")
        print(f"  # Build llama.cpp")
        print(f"  git clone https://github.com/ggerganov/llama.cpp.git")
        print(f"  cd llama.cpp")
        print(f"  make")
        print(f"  # Run quantize tool")
        print(f"  ./quantize {INPUT_GGUF} {output_path} {TARGET_QUANT}")


def estimate_size_reduction():
    """Estimate file sizes for different quant levels"""
    import math
    
    # Llama 3.2 1B base size ~2.2GB FP16
    sizes = {
        "Q6_K": 1.0 * 0.75,
        "Q5_K_M": 1.0 * 0.6,
        "Q4_K_M": 1.0 * 0.7,
        "Q3_K": 1.0 * 0.45,
        "Q2_K": 1.0 * 0.35,
    }
    
    print("\nEstimated sizes for Llama 3.2 1B:")
    for quant, size_gb in sizes.items():
        print(f"  {quant}: ~{size_gb:.2f} GB")


def main():
    print("GGUF Re-Quantization Tool")
    print("="*40)
    estimate_size_reduction()
    print()
    re_quantize_gguf()


if __name__ == "__main__":
    main()
