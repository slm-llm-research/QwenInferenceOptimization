# Week 1 Execution - Learning Guide

## 📚 Overview

This folder contains a **comprehensive Jupyter notebook** that guides you through implementing all the Week 1 scripts **from scratch** using detailed step-by-step instructions.

## 🎯 Purpose

Instead of just running pre-written code, this learning guide helps you:
- **Understand** how each function works by building it yourself
- **Learn** programming patterns and best practices
- **Develop** problem-solving skills through guided implementation
- **Master** vLLM and Qwen2.5 inference fundamentals

## 📁 Contents

### `week1_learning_guide.ipynb` - Main Learning Notebook

**35 comprehensive cells** covering:

1. **Part 1: Environment Verification** (Functions 1.1-1.6)
   - `check_python_version()` - Verify Python 3.9+
   - `check_pytorch()` - Check PyTorch installation
   - `check_cuda()` - Verify CUDA and GPU availability
   - `check_vllm()` - Check vLLM installation
   - `check_huggingfacehub()` - Check HuggingFace Hub
   - `print_summary()` - Create comprehensive validation report

2. **Part 2: Model Download** (Functions 2.1-2.2)
   - `check_model_cache()` - Check if model is cached
   - `download_model()` - Download Qwen2.5-7B-Instruct

3. **Part 3: Baseline Inference** (Function 3.1)
   - `run_baseline_inference()` - Perform first inference test
   - Understanding vLLM's LLM class and SamplingParams

4. **Part 4: Sampling Parameters Experiments** (Functions 4.1+)
   - `create_results_dir()` - Set up results directory
   - `test_parameter()` - Test individual parameter values
   - `experiment_max_tokens()` - Test output length effects
   - `save_all_results()` - Save experiment data

5. **Part 5: Results Comparison & Analysis** (Functions 5.1+)
   - `load_latest_results()` - Load experiment data
   - `compare_max_tokens()` - Analyze results
   - `show_recommendations()` - Best practices guide

6. **Part 6: Custom Parameter Testing** (Function 6.1)
   - `test_generation()` - Interactive parameter testing
   - Quick experimentation tool

### `create_notebook.py` - Generator Script

Python script used to generate the learning notebook. You can:
- Modify it to add more exercises
- Regenerate the notebook with updates
- Customize for your learning style

## 🚀 How to Use

### Step 1: Open the Notebook

```bash
cd week1-setup/execution
jupyter notebook week1_learning_guide.ipynb
# or
jupyter lab week1_learning_guide.ipynb
```

### Step 2: Follow the Structure

Each function has:
- **🎯 Goal**: What you're building
- **📋 Step-by-Step Instructions**: Detailed implementation guide
- **✅ Expected Output**: Verify your implementation
- **⚠️ Common Pitfalls**: Avoid common mistakes
- **💡 Why This Matters**: Understand the concepts
- **🧪 Test Cases**: Verify functionality

### Step 3: Implement, Don't Copy

- **Read** all instructions for a function before coding
- **Implement** the function step-by-step
- **Test** your implementation immediately
- **Compare** your output with expected output
- **Debug** if outputs don't match
- **Learn** from mistakes!

### Step 4: Progress Through Parts

Work through the parts in order:
1. Environment Verification (ensure setup is correct)
2. Model Download (get the model)
3. Baseline Inference (your first generation!)
4. Experiments (systematic parameter testing)
5. Analysis (understand the results)
6. Custom Testing (apply your knowledge)

## 📊 Learning Objectives

By the end of this notebook, you will be able to:

✅ Set up and verify a complete vLLM environment  
✅ Download and manage large language models  
✅ Perform inference with vLLM  
✅ Understand and tune sampling parameters  
✅ Design and run systematic experiments  
✅ Analyze results and extract insights  
✅ Apply best practices for different use cases  

## 💡 Tips for Success

### 1. Don't Skip Steps
Every step builds on the previous one. If you skip ahead, you might miss important concepts.

### 2. Test Incrementally
Run and test each function as soon as you implement it. Don't wait until the end!

### 3. Read Error Messages
Error messages are your friends. They tell you exactly what's wrong.

### 4. Compare with Original Scripts
After implementing a function, you can look at the original script in `week1-setup/` to see how it was done. But try to implement it yourself first!

### 5. Experiment Beyond the Guide
Once you understand a concept, try variations:
- Different prompts
- Different parameter values
- Edge cases

### 6. Take Notes
Document your observations:
- What parameter values work best for what tasks?
- What errors did you encounter?
- What insights did you gain?

## 🎓 Educational Approach

This notebook uses **guided discovery learning**:

- **Scaffolding**: Detailed steps provide support
- **Active Learning**: You implement, not just read
- **Immediate Feedback**: Test cases verify understanding
- **Conceptual Understanding**: Explanations of "why" not just "how"
- **Real-World Application**: Practical, usable code

## 📖 Mapping to Original Scripts

| Notebook Part | Original Script(s) |
|---------------|-------------------|
| Part 1 | `verify_environment.py` |
| Part 2 | `download_model.py` |
| Part 3 | `baseline_inference.py` |
| Part 4 | `experiment_sampling_params.py` |
| Part 5 | `compare_results.py` |
| Part 6 | `test_custom_params.py` |

## ⚠️ Important Notes

### Large Model Download
- Part 2 downloads a 15GB model
- Ensure you have sufficient disk space (~20GB free)
- Download can take 10-30 minutes depending on internet speed
- Downloads are resumable if interrupted

### GPU Requirements
- Most experiments require a GPU
- Minimum 16GB GPU memory recommended
- Can run on CPU but will be very slow

### Time Investment
- Complete notebook: 3-5 hours
- Can be done in multiple sessions
- Save your work frequently!

## 🔄 Workflow

```
1. Read Instructions → 2. Implement Function → 3. Test Function
                ↓                                       ↓
            Understand                            Verify Output
                ↓                                       ↓
         Add Comments ← ─ ─ ─ ─ ─ ─ ─ ─ ─  If Error: Debug
                ↓
        Move to Next Function
```

## 📝 After Completion

Once you finish the notebook:

1. **Compare**: Check your implementations against original scripts
2. **Experiment**: Try your own use cases
3. **Document**: Note which parameter combinations work best for you
4. **Share**: Help others learn by sharing your insights
5. **Progress**: Move on to Week 2!

## 🆘 Getting Help

If you get stuck:

1. **Re-read instructions**: Often the answer is in the details
2. **Check expected output**: Does your output match?
3. **Review common pitfalls**: Did you fall into a known trap?
4. **Check original script**: See how it was implemented
5. **Debug systematically**: Add print statements to understand flow
6. **Ask for help**: Reach out to instructors or peers

## 🌟 Success Criteria

You've successfully completed this when you can:

- ✅ Run all cells without errors
- ✅ Explain what each function does
- ✅ Modify parameters and predict outcomes
- ✅ Debug issues independently
- ✅ Apply concepts to new problems

---

## 🎉 Happy Learning!

Remember: **The goal is understanding, not just working code.** Take your time, experiment, and enjoy the learning process!

**Pro Tip**: Keep this README open in a browser tab while working through the notebook. It provides helpful context and guidance.

---

*Created as part of the Week 1: Introduction and Environment Preparation module of the Qwen2.5 Inference Optimization & Deployment Plan.*

