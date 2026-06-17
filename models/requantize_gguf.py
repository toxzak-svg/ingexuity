"""
GGUF Re-Quantization Script using llama.cpp

llama.cpp provides tools to convert and requantize GGUF models.
This script shows how to use the llama.cpp CLI for re-quantization.

Requirements:
    # Install llama.cpp Python bindings
    pip install llama-cpp-python
    
    # Or build llama.cpp CLI from source
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp && mkdir build && cd build
    cmake .. && cmake --build . --config Release
    # Binary will be in ./build/bin/quantize

Usage:
    python requantize_gguf.py
"""

import subprocess
import os
import shutil
import sys

# Configuration
INPUT_GGUF = os.path.abspath("./Llama-3.2-1B-Instruct-Q4_K_M.gguf")
OUTPUT_DIR = "./requantized"

TARGET_QUANTS = ["Q2_K", "Q3_K", "Q4_K_M"]  # Options to try


def find_llama_cpp_build():
    possible_paths = [
        "../llama.cpp/build/bin/quantize",
        "../llama.cpp/build/quantize",
        "llama.cpp/build/bin/quantize",
        "llama.cpp/build/quantize",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def re_quantize_with_llama_cpp(input_path, output_path, quant_type):
    quantize_bin = find_llama_cpp_build()
    
    if not quantize_bin:
        print("llama.cpp quantize tool not found!")
        print("\nTo build llama.cpp:")
        print("  git clone https://github.com/ggerganov/llama.cpp.git")
        print("  cd llama.cpp")
        print("  mkdir build && cd build")
        print("  cmake .. && cmake --build . --config Release")
        print("\nThe binary will be at: llama.cpp/build/bin/quantize")
        print(f"\nManual command to re-quantize:")
        print(f"  ./quantize {input_path} {output_path} {quant_type}")
        return False
    
    cmd = [quantize_bin, input_path, output_path, quant_type]
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Success! Output: {output_path}")
        return True
    else:
        print(f"Error: {result.stderr}")
        return False


def estimate_size_reduction():
    print("\n" + "="*50)
    print("Estimated sizes for Llama 3.2 1B:")
    print("-"*50)
    
    sizes = {
        "FP16": 2.2,
        "Q6_K": 1.0 * 0.75,
        "Q5_K_M": 1.0 * 0.6,
        "Q4_K_M": 1.0 * 0.7,  # ~700MB
        "Q3_K": 1.0 * 0.45,   # ~450MB
        "Q2_K": 1.0 * 0.35,   # ~350MB
    }
    
    for quant, size_gb in sizes.items():
        print(f"  {quant:>10}: ~{size_gb:.2f} GB ({int(size_gb*1024)} MB)")
    
    print("-"*50)
    print("Q2_K is smallest but may have quality loss")


def requantize_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_GGUF):
        print(f"ERROR: Input file not found: {INPUT_GGUF}")
        return
    
    print(f"Input: {INPUT_GGUF}")
    print(f"Output dir: {OUTPUT_DIR}")
    print()
    
    for quant in TARGET_QUANTS:
        output_path = os.path.join(OUTPUT_DIR, f"Llama-3.2-1B-{quant}.gguf")
        print(f"\n>>> Quantizing to {quant}...")
        re_quantize_with_llama_cpp(INPUT_GGUF, output_path, quant)


def main():
    print("GGUF Re-Quantization Tool")
    print("="*50)
    estimate_size_reduction()
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        requantize_all()
    else:
        print("Run with --all to requantize all configurations:")
        print("  python requantize_gguf.py --all")
        print()
        
        if os.path.exists(INPUT_GGUF):
            input_size = os.path.getsize(INPUT_GGUF) / (1024**2)
            print(f"Current model: {INPUT_GGUF}")
            print(f"Current size: {input_size:.1f} MB")


if __name__ == "__main__":
    main()
