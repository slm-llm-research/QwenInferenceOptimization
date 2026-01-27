# Quick Start Guide

## 🚀 Get Started in 5 Minutes

This course teaches you LLM inference optimization and deployment from scratch. Here's how to begin:

### Step 1: Prerequisites Check

You'll need:
- **GPU**: NVIDIA GPU with CUDA 12.1+ (16GB+ VRAM recommended)
- **OS**: Linux (Ubuntu 20.04+) or macOS for development
- **Python**: 3.9, 3.10, or 3.11
- **Accounts** (for deployment weeks):
  - Runpod account (Week 6)
  - AWS account (Week 6-7)

### Step 2: Clone or Navigate to This Repository

```bash
cd /path/to/InferenceOptimization
```

### Step 3: Start with Week 1

```bash
cd week1-setup
cat README.md  # Read the detailed instructions
```

Each week's README contains:
- ✅ Clear learning objectives
- 📚 Background concepts explained
- 🚀 Step-by-step instructions
- 🐛 Troubleshooting guide
- ✅ Completion checklist

### Step 4: Follow Week by Week

**Week-by-week progression:**

1. **Week 1** (2-3 hours): Environment setup and first inference
2. **Week 2** (3-4 hours): Performance profiling and baseline metrics
3. **Week 3** (4-5 hours): GPU optimization and tuning
4. **Week 4** (1-2 hours): Integration week - learning and documentation
5. **Week 5** (3-4 hours): Multi-GPU distributed inference
6. **Week 6** (4-6 hours): Cloud deployment (Runpod + AWS EKS)
7. **Week 7** (2-3 hours): Load testing and final teardown

**Total time**: ~20-30 hours

## 💡 Key Features

### Educational Focus
- **No prior vLLM or Kubernetes knowledge required**
- Clear explanations of every concept
- Hands-on code for every week
- Real-world deployment patterns

### Cost-Conscious
- ⚠️ Detailed cost estimates for cloud resources
- 🛑 Comprehensive teardown instructions
- ✅ Verification scripts to avoid lingering charges
- 💰 Cost management best practices

### Production-Ready
- Docker containerization
- Kubernetes deployment
- Load testing at scale
- Monitoring and optimization

## 📁 Project Structure

```
InferenceOptimization/
├── README.md                    # Course overview
├── QUICKSTART.md               # This file
├── requirements.txt            # All Python dependencies
│
├── week1-setup/                # Environment preparation
│   ├── README.md              # Detailed week 1 guide
│   ├── baseline_inference.py  # First inference test
│   └── ...
│
├── week2-profiling/            # Performance benchmarking
│   ├── README.md
│   ├── benchmark_latency.py
│   ├── benchmark_throughput.py
│   └── ...
│
├── week3-optimization/         # GPU tuning
│   ├── README.md
│   ├── optimize_memory_utilization.py
│   ├── optimize_max_num_seqs.py
│   └── ...
│
├── week4-integration/          # Learning week
│   ├── README.md
│   └── ...
│
├── week5-distributed/          # Multi-GPU inference
│   ├── README.md
│   ├── test_tensor_parallel.py
│   └── ...
│
├── week6-deployment/           # Cloud deployment
│   ├── README.md
│   ├── deploy_eks.sh          # EKS deployment script
│   ├── teardown_eks.sh        # EKS teardown script ⚠️
│   ├── docker/                # Container files
│   └── kubernetes/            # K8s manifests
│
└── week7-load-testing/         # Production testing
    ├── README.md
    ├── run_load_test.py       # Load testing
    ├── verify_cleanup.sh      # Cleanup verification ⚠️
    └── ...
```

## ⚠️ Important Notes

### About Cloud Costs

**Weeks 6-7 use cloud resources that cost money!**

- Always read the cost warnings in each week's README
- Follow teardown instructions immediately after testing
- Use the verification scripts to ensure cleanup
- Set up billing alerts on AWS

### About Multi-GPU (Week 5)

Week 5 requires multi-GPU access. Options:
1. Rent multi-GPU instance (Runpod, AWS)
2. Skip and proceed to Week 6 (single-GPU deployment works)
3. Review concepts without running code

### Recommended Hardware

**For local development (Week 1-4)**:
- GPU: RTX 3090/4090, A100, A10G, or T4
- VRAM: 16GB minimum, 24GB+ recommended
- RAM: 16GB+

**For deployment (Week 6-7)**:
- Use cloud GPUs (cheaper than buying hardware)
- AWS g5.xlarge or Runpod A10G recommended

## 🆘 Getting Help

### If You Get Stuck

1. **Check the week's README** - Most issues are covered in troubleshooting sections
2. **Review prerequisites** - Ensure all tools are installed correctly
3. **Check the specific error** - Most errors have clear solutions in the READMEs

### Common Issues

**"CUDA out of memory"**
- Expected during optimization experiments
- Helps find the limits
- Reduce batch size or max_tokens

**"No module named 'vllm'"**
- Virtual environment not activated
- Run: `source vllm-env/bin/activate`

**"Cannot connect to cluster"**
- Update kubeconfig: `aws eks update-kubeconfig --name CLUSTER --region REGION`

## 📚 What You'll Learn

By the end of this course:

✅ LLM inference optimization techniques
✅ GPU performance profiling and tuning
✅ Distributed inference patterns
✅ Docker and Kubernetes for ML
✅ Cloud deployment (AWS EKS)
✅ Production load testing
✅ Cost optimization strategies

## 🎯 Success Criteria

You'll know you've succeeded when you can:

1. Deploy a production-ready LLM inference endpoint
2. Optimize GPU utilization from ~60% to 90%+
3. Scale inference across multiple GPUs
4. Handle 100+ concurrent users
5. Properly manage cloud resources and costs

## 🔜 Ready to Start?

```bash
cd week1-setup
cat README.md
```

Good luck on your LLM inference optimization journey! 🚀

---

**Remember**: This is a hands-on learning course. Don't just read - run the code, experiment with parameters, and learn by doing!

