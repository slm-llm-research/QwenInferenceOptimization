"""
Week 1: Model Download Script

This script downloads the Qwen2.5-7B-Instruct model from HuggingFace Hub.
The model will be cached locally for faster subsequent loads.

Model Details:
- Name: Qwen/Qwen2.5-7B-Instruct
- Size: ~15GB
- License: Apache 2.0 (open source)
- Context Length: Up to 131k tokens (default 32k)
"""

import os
import sys
from pathlib import Path


def download_model():
    """Download Qwen2.5-7B-Instruct model from HuggingFace Hub"""
    
    print("="*60)
    print("Week 1: Downloading Qwen2.5-7B-Instruct Model")
    print("="*60)
    print("")
    
    # Check if huggingface_hub is installed
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ Error: huggingface_hub is not installed")
        print("Run: pip install huggingface-hub")
        sys.exit(1)
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    print(f"📦 Model: {model_name}")
    print(f"📏 Expected size: ~15 GB")
    print(f"📁 Cache location: {os.path.expanduser('~/.cache/huggingface')}")
    print("")
    print("⏳ This may take 10-30 minutes depending on your internet speed...")
    print("    You can cancel and run again later - progress is saved.")
    print("")
    
    try:
        # Download model with progress bar
        cache_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=None,  # Use default cache directory
            resume_download=True,  # Resume if interrupted
        )
        
        print("")
        print("="*60)
        print("✅ Model downloaded successfully!")
        print("="*60)
        print(f"📁 Cache location: {cache_dir}")
        print("")
        print("Next steps:")
        print("1. Run: python baseline_inference.py")
        print("")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user.")
        print("Progress has been saved. Run this script again to resume.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space (~20GB free)")
        print("3. Try again - downloads are resumable")
        print("4. If issues persist, the model will auto-download on first use")
        sys.exit(1)


def check_model_cache():
    """Check if model is already in cache"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    if not cache_dir.exists():
        return False
    
    # Look for Qwen model directories
    for item in cache_dir.iterdir():
        if "qwen2.5-7b-instruct" in item.name.lower():
            return True
    
    return False


def main():
    # Check if model might already be cached
    if check_model_cache():
        print("ℹ️  Model appears to be already cached.")
        response = input("Do you want to re-download/update? (y/N): ")
        if response.lower() != 'y':
            print("Skipping download. Model will be loaded from cache.")
            return
    
    download_model()


if __name__ == "__main__":
    main()

