"""
Week 1: Environment Verification Script

This script checks that all prerequisites are properly installed:
- Python version
- PyTorch and CUDA
- vLLM
- GPU availability

Run this after installing dependencies to ensure everything is set up correctly.
"""

import sys
import subprocess


def check_python_version():
    """Check Python version is 3.9+"""
    version = sys.version_info
    print(f"🔍 Checking Python version...")
    print(f"   Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 9:
        print("   ✓ Python version is compatible (3.9+)")
        return True
    else:
        print("   ✗ Python version must be 3.9 or higher")
        return False


def check_pytorch():
    """Check PyTorch installation"""
    print(f"\n🔍 Checking PyTorch...")
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print("   ✓ PyTorch is installed")
        return True
    except ImportError:
        print("   ✗ PyTorch is not installed")
        print("   Run: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return False


def check_cuda():
    """Check CUDA availability"""
    print(f"\n🔍 Checking CUDA...")
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"   CUDA available: True")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")
            
            # Get GPU details
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            print("   ✓ CUDA is available and working")
            return True
        else:
            print("   ✗ CUDA is not available")
            print("   Note: You can still run on CPU but inference will be very slow")
            print("   Verify: nvidia-smi shows your GPU")
            return False
            
    except Exception as e:
        print(f"   ✗ Error checking CUDA: {e}")
        return False


def check_vllm():
    """Check vLLM installation"""
    print(f"\n🔍 Checking vLLM...")
    try:
        import vllm
        print(f"   vLLM version: {vllm.__version__}")
        print("   ✓ vLLM is installed")
        return True
    except ImportError:
        print("   ✗ vLLM is not installed")
        print("   Run: pip install vllm")
        return False


def check_huggingface_hub():
    """Check HuggingFace Hub"""
    print(f"\n🔍 Checking HuggingFace Hub...")
    try:
        import huggingface_hub
        print(f"   HuggingFace Hub version: {huggingface_hub.__version__}")
        print("   ✓ HuggingFace Hub is installed")
        return True
    except ImportError:
        print("   ✗ HuggingFace Hub is not installed")
        print("   Run: pip install huggingface-hub")
        return False


def print_summary(results):
    """Print summary of all checks"""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All checks passed! Your environment is ready.")
        print("\nNext steps:")
        print("1. Run: python download_model.py")
        print("2. Run: python baseline_inference.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("See README.md for detailed installation instructions.")
    
    print("")
    return all_passed


def main():
    """Run all environment checks"""
    print("="*60)
    print("Week 1: Environment Verification")
    print("="*60)
    
    results = {
        "Python 3.9+": check_python_version(),
        "PyTorch": check_pytorch(),
        "CUDA": check_cuda(),
        "vLLM": check_vllm(),
        "HuggingFace Hub": check_huggingface_hub(),
    }
    
    success = print_summary(results)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

