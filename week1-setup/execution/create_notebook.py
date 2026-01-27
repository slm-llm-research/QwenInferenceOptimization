"""
Script to create the Week 1 Learning Guide Jupyter Notebook
This creates a comprehensive learning notebook with step-by-step instructions
"""

import json

def create_learning_notebook():
    """Create a comprehensive learning notebook for Week 1"""
    
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    def add_markdown(text):
        """Add a markdown cell"""
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": text.split("\n")
        })
    
    def add_code(text):
        """Add a code cell"""
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.split("\n")
        })
    
    # Title and Introduction
    add_markdown("""# Week 1: vLLM & Qwen2.5 - Complete Learning Guide

## 🎯 Learning Objectives

By completing this notebook, you will:
1. **Set up and verify** your vLLM environment with GPU support
2. **Download and cache** the Qwen2.5-7B-Instruct model from HuggingFace
3. **Perform basic inference** using vLLM's LLM class
4. **Understand sampling parameters** and how they affect text generation
5. **Experiment systematically** with different parameter combinations
6. **Analyze results** to determine best practices for different use cases

## 📚 Structure

This notebook covers 6 main sections (matching the week1-setup scripts):
1. **Environment Verification** (`verify_environment.py`)
2. **Model Download** (`download_model.py`)
3. **Baseline Inference** (`baseline_inference.py`)
4. **Sampling Parameters** (`experiment_sampling_params.py`)
5. **Results Comparison** (`compare_results.py`)
6. **Custom Testing** (`test_custom_params.py`)

## 🚨 Learning Approach

- **Learn by implementing**: Don't copy from the original scripts!
- **Follow step-by-step guides**: Each function has detailed instructions
- **Test incrementally**: Run cells as you complete them
- **Check expected outputs**: Verify your implementation
- **Learn from pitfalls**: Common mistakes are highlighted

---""")

    # Part 1: Environment Verification
    add_markdown("""# Part 1: Environment Verification

## 🎓 What You'll Learn
- Programmatically check Python version
- Verify PyTorch and CUDA installation
- Inspect GPU properties using PyTorch
- Check required libraries (vLLM, HuggingFace Hub)
- Create comprehensive validation reports

## 📖 Background
Before running LLM inference, ensure:
- Python 3.9+ is installed
- PyTorch with CUDA support
- CUDA can access your GPU(s)
- vLLM and HuggingFace Hub are available

---""")

    add_markdown("""## Function 1.1: `check_python_version()`

### 🎯 Goal
Create a function that checks if Python version is 3.9 or higher.

### 📋 Step-by-Step Instructions

**Step 1:** Import the `sys` module at the top of your notebook
- Hint: `sys` is a built-in module, no installation needed

**Step 2:** Inside the function, access the Python version information
- Use `sys.version_info` to get version details
- This returns an object with attributes: `major`, `minor`, `micro`

**Step 3:** Print a header message
- Use an emoji like 🔍 to make it visually clear
- Print "Checking Python version..."

**Step 4:** Print the current Python version
- Format: "Python version: {major}.{minor}.{micro}"
- Use f-strings for formatting

**Step 5:** Check if version meets requirements
- Requirement: Python 3.9 or higher
- Check: `version.major == 3 and version.minor >= 9`
- Why both conditions? Python 4.x would also pass if it existed!

**Step 6:** Print success or failure message
- If compatible: Print "✓ Python version is compatible (3.9+)"
- If not: Print "✗ Python version must be 3.9 or higher"
- Use proper indentation (3 spaces) to align with other output

**Step 7:** Return the result
- Return `True` if compatible
- Return `False` if not compatible

### ✅ Expected Output
```
🔍 Checking Python version...
   Python version: 3.10.12
   ✓ Python version is compatible (3.9+)
```

### ⚠️ Common Pitfalls
1. **Forgetting to check major version**: Just checking `minor >= 9` would fail for Python 2.9 (if it existed)
2. **Wrong attribute names**: It's `version_info.major`, not `version_info.version`
3. **Not returning a boolean**: The function should return True/False for later use

### 🧪 Test Your Function
After implementing, run:
```python
result = check_python_version()
assert isinstance(result, bool), "Function should return a boolean"
print(f"\\nFunction returned: {result}")
```""")

    add_code("""# TODO: Import necessary modules
import sys

# TODO: Implement check_python_version() function
def check_python_version():
    \"\"\"Check Python version is 3.9+\"\"\"
    # Step 1: Access version info
    
    # Step 2: Print header
    
    # Step 3: Print current version
    
    # Step 4: Check requirements
    
    # Step 5: Print result message
    
    # Step 6: Return boolean
    pass

# TODO: Test your function
# result = check_python_version()
# assert isinstance(result, bool)
# print(f"\\nTest passed! Result: {result}")""")

    # Add remaining environment check functions
    add_markdown("""## Function 1.2: `check_pytorch()`

### 🎯 Goal
Check if PyTorch is installed and display its version.

### 📋 Step-by-Step Instructions

**Step 1:** Print a header with proper spacing
- Use `\\n` before the emoji for spacing
- Print "🔍 Checking PyTorch..."

**Step 2:** Try to import PyTorch
- Use a try-except block to handle ImportError
- Try: `import torch`
- This is the safe way to check if a library is installed

**Step 3:** If import succeeds, get version info
- Access `torch.__version__` to get version string
- Print: "PyTorch version: {version}"
- Print: "✓ PyTorch is installed"
- Return `True`

**Step 4:** If import fails, provide helpful guidance
- Catch `ImportError`
- Print: "✗ PyTorch is not installed"
- Print installation command: "Run: pip install torch --index-url https://download.pytorch.org/whl/cu121"
- Return `False`

### ✅ Expected Output (if PyTorch installed)
```
🔍 Checking PyTorch...
   PyTorch version: 2.1.0+cu121
   ✓ PyTorch is installed
```

### ⚠️ Common Pitfalls
1. **Importing at module level**: Import inside the function so ImportError can be caught
2. **Not providing installation instructions**: Users need to know how to fix the issue
3. **Forgetting indentation**: Error messages should be indented with 3 spaces for alignment""")

    add_code("""# TODO: Implement check_pytorch() function
def check_pytorch():
    \"\"\"Check PyTorch installation\"\"\"
    # Step 1: Print header
    
    # Step 2: Try to import torch
    
    # Step 3: If success, get and print version
    
    # Step 4: If fail, print error and instructions
    
    pass

# TODO: Test your function
# result = check_pytorch()
# print(f"\\nPyTorch available: {result}")""")

    # Add check_cuda function
    add_markdown("""## Function 1.3: `check_cuda()`

### 🎯 Goal
Check CUDA availability and display GPU information.

### 📋 Step-by-Step Instructions

**Step 1:** Print the header
- Print "\\n🔍 Checking CUDA..."

**Step 2:** Wrap everything in a try-except block
- Catch general `Exception` to handle any unexpected errors

**Step 3:** Import torch (inside the try block)
- `import torch`

**Step 4:** Check if CUDA is available
- Use `torch.cuda.is_available()` - returns True/False
- Store result in a variable

**Step 5:** If CUDA is available, gather detailed information
- Print: "CUDA available: True"
- Get CUDA version: `torch.version.cuda`
- Print: "CUDA version: {version}"
- Get number of GPUs: `torch.cuda.device_count()`
- Print: "Number of GPUs: {count}"

**Step 6:** Loop through each GPU and get details
- Use `for i in range(torch.cuda.device_count()):`
- Get GPU name: `torch.cuda.get_device_name(i)`
- Get GPU memory: `torch.cuda.get_device_properties(i).total_memory`
- Convert memory from bytes to GB: divide by `1024**3`
- Print: "GPU {i}: {name} ({memory:.1f} GB)"

**Step 7:** Print success message and return True
- Print: "✓ CUDA is available and working"
- Return `True`

**Step 8:** If CUDA is not available (in the else block)
- Print: "✗ CUDA is not available"
- Print helpful note about CPU fallback
- Return `False`

**Step 9:** Handle exceptions in the except block
- Print: "✗ Error checking CUDA: {error_message}"
- Return `False`

### ✅ Expected Output (with GPU)
```
🔍 Checking CUDA...
   CUDA available: True
   CUDA version: 12.1
   Number of GPUs: 1
   GPU 0: NVIDIA A100-SXM4-40GB (40.0 GB)
   ✓ CUDA is available and working
```

### ⚠️ Common Pitfalls
1. **Memory unit conversion**: `total_memory` is in bytes, need to divide by `1024**3` for GB
2. **Forgetting the .1f format**: Shows memory with 1 decimal place
3. **Not handling the case when device_count is 0**: The for loop handles this automatically""")

    add_code("""# TODO: Implement check_cuda() function
def check_cuda():
    \"\"\"Check CUDA availability\"\"\"
    # Step 1: Print header
    
    # Step 2: Start try-except block
    
    # Step 3: Import torch
    
    # Step 4: Check CUDA availability
    
    # Step 5: If available, get details
    
    # Step 6: Loop through GPUs
    
    # Step 7: Print success and return True
    
    # Step 8: Handle not available case
    
    # Step 9: Handle exceptions
    
    pass

# TODO: Test your function
# result = check_cuda()
# print(f"\\nCUDA available: {result}")""")

    # Continue with remaining check functions
    for lib_name, import_name, install_cmd in [
        ("vLLM", "vllm", "pip install vllm"),
        ("HuggingFace Hub", "huggingface_hub", "pip install huggingface-hub")
    ]:
        add_markdown(f"""## Function 1.{4 if lib_name == "vLLM" else 5}: `check_{import_name.replace('_', '')}()`

### 🎯 Goal
Check if {lib_name} library is installed.

### 📋 Step-by-Step Instructions

**Step 1:** Follow the same pattern as `check_pytorch()`
- Print header: "\\n🔍 Checking {lib_name}..."

**Step 2:** Use try-except for import
- Try to `import {import_name}`
- Get version: `{import_name}.__version__`
- Print version and success message
- Return `True`

**Step 3:** Handle ImportError
- Print error and installation command: "Run: {install_cmd}"
- Return `False`

### ✅ Expected Output
```
🔍 Checking {lib_name}...
   {lib_name} version: X.X.X
   ✓ {lib_name} is installed
```""")

        add_code(f"""# TODO: Implement check_{import_name.replace('_', '')}() function
def check_{import_name.replace('_', '')}():
    \"\"\"Check {lib_name} installation\"\"\"
    # Your code here
    pass

# TODO: Test your function
# result = check_{import_name.replace('_', '')}()""")

    # Add print_summary function
    add_markdown("""## Function 1.6: `print_summary(results)`

### 🎯 Goal
Create a summary of all environment checks.

### 📝 Function Signature
```python
def print_summary(results):
    \"\"\"Print summary of all checks
    
    Args:
        results: Dictionary with check names as keys and boolean results as values
        
    Returns:
        bool: True if all checks passed, False otherwise
    \"\"\"
    pass
```

### 📋 Step-by-Step Instructions

**Step 1:** Print a formatted header
- Print a blank line
- Print "="*60 (creates a line of 60 equal signs)
- Print "SUMMARY"
- Print "="*60 again

**Step 2:** Determine if all checks passed
- Use `all(results.values())` to check if all values are True
- Store in variable `all_passed`

**Step 3:** Loop through results and print each one
- Use `for check, passed in results.items():`
- Set status emoji: "✓" if passed, else "✗"
- Use ternary operator: `status = "✓" if passed else "✗"`
- Print: "{status} {check}"

**Step 4:** Print footer
- Print "="*60

**Step 5:** Print conditional final message
- If `all_passed`:
  - Print "\\n🎉 All checks passed! Your environment is ready."
  - Print "\\nNext steps:"
  - Print "1. Continue to Part 2: Model Download"
- Else:
  - Print "\\n⚠️  Some checks failed. Please fix the issues above."

**Step 6:** Print blank line and return result
- Print ""
- Return `all_passed`

### ✅ Expected Output (all passing)
```
============================================================
SUMMARY
============================================================
✓ Python 3.9+
✓ PyTorch
✓ CUDA
✓ vLLM
✓ HuggingFace Hub
============================================================

🎉 All checks passed! Your environment is ready.

Next steps:
1. Continue to Part 2: Model Download

```

### ⚠️ Common Pitfalls
1. **Using `any()` instead of `all()`**: We need ALL checks to pass
2. **Forgetting `.values()`**: `results` is a dict, need to get just the boolean values
3. **Wrong number of `=` signs**: Should be exactly 60 for proper formatting""")

    add_code("""# TODO: Implement print_summary() function
def print_summary(results):
    \"\"\"Print summary of all checks\"\"\"
    # Step 1: Print header
    
    # Step 2: Check if all passed
    
    # Step 3: Loop through and print each result
    
    # Step 4: Print footer
    
    # Step 5: Print conditional message
    
    # Step 6: Return result
    
    pass

# TODO: Test your function with sample data
# sample_results = {
#     "Python 3.9+": True,
#     "PyTorch": True,
#     "CUDA": True,
#     "vLLM": True,
#     "HuggingFace Hub": True,
# }
# print_summary(sample_results)""")

    # Add complete verification runner
    add_markdown("""## 🧪 Run Complete Environment Verification

Now let's put it all together and run a complete environment check!""")

    add_code("""# TODO: Run all checks and create summary
def run_environment_verification():
    \"\"\"Run all environment checks\"\"\"
    print("="*60)
    print("Week 1: Environment Verification")
    print("="*60)
    
    # TODO: Create a dictionary with check names and results
    # Call each check function and store the result
    results = {
        "Python 3.9+": check_python_version(),
        "PyTorch": check_pytorch(),
        "CUDA": check_cuda(),
        "vLLM": check_vllm(),
        "HuggingFace Hub": check_huggingfacehub(),
    }
    
    # TODO: Print summary
    success = print_summary(results)
    return success

# TODO: Run the verification
# run_environment_verification()""")

    # Part 2: Model Download
    add_markdown("""---

# Part 2: Model Download

## 🎓 What You'll Learn
- How to download models from HuggingFace Hub programmatically
- How to use `snapshot_download` for large models
- How to check if a model is already cached
- How to handle download interruptions gracefully
- How to work with Path objects for file system operations

## 📖 Background
The Qwen2.5-7B-Instruct model is approximately 15GB. HuggingFace Hub provides `snapshot_download` which:
- Downloads all model files to a local cache
- Supports resumable downloads (can be interrupted and continued)
- Caches files in `~/.cache/huggingface/hub/` by default
- Allows models to be loaded quickly on subsequent uses

---""")

    add_markdown("""## Function 2.1: `check_model_cache()`

### 🎯 Goal
Check if the Qwen2.5-7B-Instruct model is already cached locally.

### 📋 Step-by-Step Instructions

**Step 1:** Import required modules at the top of the notebook
- `from pathlib import Path`
- `import os`

**Step 2:** Construct the cache directory path
- HuggingFace cache location: `~/.cache/huggingface/hub`
- Use `Path.home()` to get the user's home directory
- Chain with: `/ ".cache" / "huggingface" / "hub"`
- Store in variable `cache_dir`

**Step 3:** Check if cache directory exists
- Use `cache_dir.exists()` method
- If it doesn't exist, return `False` immediately
- No cache directory = no cached models

**Step 4:** Look for Qwen model directories
- Use `cache_dir.iterdir()` to iterate through all items
- For each item, check if "qwen2.5-7b-instruct" is in the name (lowercase)
- Use `item.name.lower()` to make the search case-insensitive
- If found, return `True`

**Step 5:** If no Qwen directory found
- Return `False`

### ✅ Expected Behavior
- Returns `True` if a directory containing "qwen2.5-7b-instruct" exists in the cache
- Returns `False` otherwise

### ⚠️ Common Pitfalls
1. **Case sensitivity**: Model directory names might have different casing, always use `.lower()`
2. **Not checking if directory exists first**: `iterdir()` will fail if directory doesn't exist
3. **Using string paths instead of Path objects**: Path objects are more robust and cross-platform

### 💡 Why This Matters
Checking the cache before downloading:
- Saves time (no need to re-download 15GB)
- Saves bandwidth
- Provides better user experience (can prompt before re-downloading)""")

    add_code("""# TODO: Import required modules
from pathlib import Path
import os

# TODO: Implement check_model_cache() function
def check_model_cache():
    \"\"\"Check if model is already in cache\"\"\"
    # Step 1: Construct cache directory path
    
    # Step 2: Check if directory exists
    
    # Step 3: Look for Qwen model directories
    
    # Step 4: Return result
    
    pass

# TODO: Test your function
# is_cached = check_model_cache()
# print(f"Model cached: {is_cached}")""")

    # Add download_model function
    add_markdown("""## Function 2.2: `download_model()`

### 🎯 Goal
Download the Qwen2.5-7B-Instruct model from HuggingFace Hub with proper error handling.

### 📋 Step-by-Step Instructions

**Step 1:** Print a formatted header
- Print "="*60
- Print "Week 1: Downloading Qwen2.5-7B-Instruct Model"
- Print "="*60
- Print blank line

**Step 2:** Try to import snapshot_download
- Wrap in try-except to catch ImportError
- `from huggingface_hub import snapshot_download`
- If ImportError: print error message and return (don't use sys.exit in notebooks!)

**Step 3:** Define model name and print info
- `model_name = "Qwen/Qwen2.5-7B-Instruct"`
- Print: "📦 Model: {model_name}"
- Print: "📏 Expected size: ~15 GB"
- Print: "📁 Cache location: {os.path.expanduser('~/.cache/huggingface')}"
- Print blank line
- Print: "⏳ This may take 10-30 minutes depending on your internet speed..."
- Print: "    You can cancel and run again later - progress is saved."
- Print blank line

**Step 4:** Download the model with proper error handling
- Wrap in try-except block with TWO exception types:
  1. `KeyboardInterrupt` (user presses Ctrl+C)
  2. `Exception` (any other error)

**Step 5:** In the try block, call snapshot_download
- Use these parameters:
  - `repo_id=model_name`
  - `cache_dir=None` (use default)
  - `resume_download=True` (enable resumable downloads)
- Store returned path in `cache_dir` variable

**Step 6:** Print success message
- Print blank line
- Print "="*60
- Print "✅ Model downloaded successfully!"
- Print "="*60
- Print: "📁 Cache location: {cache_dir}"
- Print blank line
- Print "Next steps:"
- Print "1. Continue to Part 3: Baseline Inference"
- Print blank line

**Step 7:** Handle KeyboardInterrupt (in except block)
- Print: "\\n\\n⚠️  Download interrupted by user."
- Print: "Progress has been saved. Run this cell again to resume."
- Return from function (in notebooks, don't use sys.exit)

**Step 8:** Handle general exceptions (in except block)
- Catch `Exception as e`
- Print: "\\n❌ Error downloading model: {e}"
- Print "\\nTroubleshooting:"
- Print "1. Check your internet connection"
- Print "2. Ensure you have enough disk space (~20GB free)"
- Print "3. Try again - downloads are resumable"
- Print "4. If issues persist, the model will auto-download on first use"
- Return from function

### ⚠️ Common Pitfalls
1. **Using sys.exit() in notebooks**: This will crash the kernel. Use `return` instead.
2. **Not handling KeyboardInterrupt separately**: Users should know their progress is saved.
3. **Forgetting resume_download=True**: Downloads should be resumable for large models.
4. **Not providing troubleshooting steps**: Users need guidance when errors occur.""")

    add_code("""# TODO: Implement download_model() function
def download_model():
    \"\"\"Download Qwen2.5-7B-Instruct model from HuggingFace Hub\"\"\"
    # Step 1: Print header
    
    # Step 2: Try to import snapshot_download
    
    # Step 3: Define model name and print info
    
    # Step 4-6: Download with error handling
    
    # Step 7-8: Handle exceptions
    
    pass

# TODO: Uncomment to run the download (WARNING: Large download!)
# First, check if model is already cached
# if not check_model_cache():
#     download_model()
# else:
#     print("✅ Model is already cached!")""")

    # Save the notebook
    with open("week1_learning_guide.ipynb", "w") as f:
        json.dump(notebook, f, indent=2)
    
    print("✅ Notebook created successfully!")
    print("📄 File: week1_learning_guide.ipynb")
    print(f"📊 Total cells: {len(notebook['cells'])}")

if __name__ == "__main__":
    create_learning_notebook()

