# Qwen2.5 Inference Optimization & Deployment Course

Welcome to this comprehensive 7-week course on LLM inference optimization and deployment! This project will guide you through optimizing and deploying the Qwen2.5-7B-Instruct model using vLLM, from basic setup to production-ready deployment on cloud platforms.

## 🎯 Course Overview

This course is structured as a hands-on learning journey, with each week building upon the previous one:

- **Week 1**: Environment Setup and Model Loading
- **Week 2**: Baseline Performance Profiling
- **Week 3**: GPU-Level Optimization
- **Week 4**: Integration Week (Guest Lectures & Exploration)
- **Week 5**: Multi-GPU & Distributed Inference
- **Week 6**: Production Deployment (Runpod & AWS EKS)
- **Week 7**: Load Testing & Final Teardown

## 📁 Project Structure

```
InferenceOptimization/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── week1-setup/                 # Environment preparation
├── week2-profiling/             # Baseline benchmarking
├── week3-optimization/          # GPU tuning experiments
├── week4-integration/           # Documentation and exploration
├── week5-distributed/           # Multi-GPU scaling
├── week6-deployment/            # Cloud deployment (Runpod, EKS)
└── week7-load-testing/          # Production load tests
```

## 🚀 Getting Started

### Prerequisites

- **Hardware**: GPU with CUDA Compute Capability ≥7.0 (e.g., T4, A10G, A100)
- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS for development
- **Python**: 3.9 or higher
- **CUDA**: 12.1+ (for GPU instances)
- **Cloud Accounts** (for deployment weeks):
  - Runpod account (Week 6)
  - AWS account with EKS access (Week 6-7)

### Quick Start

Each week is self-contained with its own README and scripts. Start with Week 1:

```bash
cd week1-setup
cat README.md
```

## 💡 Learning Philosophy

This course emphasizes:
- **Hands-on Learning**: Every concept is accompanied by runnable code
- **Progressive Complexity**: Each week builds on previous knowledge
- **Cost Awareness**: Detailed teardown instructions to avoid unexpected charges
- **Production Readiness**: Real-world deployment patterns and best practices

## 💰 Cost Management

**IMPORTANT**: Weeks 6-7 involve cloud resources that incur costs. Each README includes:
- ✅ Setup instructions
- 🧪 Testing procedures
- ❌ **Teardown instructions** (follow these carefully!)

Estimated costs (as of 2026):
- Runpod: ~$0.50-2.00/hour depending on GPU
- AWS EKS: ~$0.10/hour (control plane) + instance costs
- **Always tear down resources immediately after experimentation**

## 📚 Learning Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [AWS EKS User Guide](https://docs.aws.amazon.com/eks/)
- [Runpod Documentation](https://docs.runpod.io/)

## 🤝 How to Use This Course

1. **Sequential Learning**: Follow weeks in order - each builds on the previous
2. **Read First**: Always read the week's README completely before running code
3. **Experiment**: Modify parameters and observe the effects
4. **Document**: Keep notes on your observations and results
5. **Clean Up**: Always run teardown scripts after cloud deployments

## 📝 Notes

- Some weeks require specific hardware (GPUs). Development can be done locally, but deployment requires cloud GPUs.
- The course assumes no prior knowledge of vLLM or Kubernetes, but basic Python and command-line familiarity is expected.
- All scripts include comments and documentation for educational purposes.

## 🆘 Troubleshooting

If you encounter issues:
1. Check the specific week's README for known issues
2. Verify your environment matches the prerequisites
3. Ensure CUDA and GPU drivers are properly installed
4. For cloud issues, verify credentials and permissions

## 📊 Expected Outcomes

By the end of this course, you will:
- Understand LLM inference optimization techniques
- Know how to profile and benchmark inference performance
- Master vLLM configuration for maximum throughput
- Deploy production-ready inference systems on cloud platforms
- Implement proper monitoring and scaling strategies
- Manage cloud resources cost-effectively

---

**Ready to begin? Start with [Week 1: Setup](week1-setup/README.md)**

