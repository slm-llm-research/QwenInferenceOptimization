# Week 7: Demo Day – Load Test and Final Teardown

## 🎯 Goals

By the end of this week, you will:
- Perform production-scale load testing (100+ concurrent users)
- Analyze system performance under stress
- Validate production readiness
- Execute complete infrastructure teardown
- Document final recommendations

## 📚 What You'll Learn

- Load testing methodologies for LLM services
- Performance analysis under concurrent load
- Identifying bottlenecks and limits
- Production capacity planning
- Complete cost cleanup verification

## 🔬 Key Concepts

### Load Testing Metrics

1. **Throughput**: Requests per second (RPS) or tokens per second
2. **Latency**: Response time (p50, p95, p99)
3. **Error Rate**: Failed requests / total requests
4. **Saturation Point**: When adding load no longer increases throughput
5. **Resource Utilization**: GPU/CPU/memory under load

### Why Load Test?

- Verify system handles expected traffic
- Find breaking points before production
- Validate autoscaling and failover
- Measure real-world user experience
- Justify infrastructure costs

## 🚀 Running Load Tests

### Prerequisites

```bash
pip install locust requests
```

### Test 1: Baseline Load Test

Test with moderate concurrent users:

```bash
python run_load_test.py --users 50 --duration 300
```

**What it does**:
- Simulates 50 concurrent users
- Runs for 5 minutes
- Measures latency and throughput
- Reports errors and failures

**Expected results** (Qwen2.5-7B on 1x A10G):
- RPS: 2-5 requests/second
- P95 latency: 2-5 seconds
- Error rate: <1%

### Test 2: Stress Test

Push to limits:

```bash
python run_load_test.py --users 100 --duration 300
```

**What it does**:
- 100 concurrent users
- Finds saturation point
- Identifies when system degrades

**Watch for**:
- Increasing latency
- Rising error rates
- GPU memory/utilization limits

### Test 3: Spike Test

Sudden traffic spike:

```bash
python spike_test.py --initial 10 --spike 100 --duration 60
```

**What it does**:
- Starts with 10 users
- Suddenly jumps to 100
- Tests system responsiveness

### Test 4: Locust Interactive

For detailed analysis:

```bash
locust -f locustfile.py --host http://YOUR_ENDPOINT
```

**Then**:
- Open browser: http://localhost:8089
- Set users and spawn rate
- Watch real-time dashboard
- Download detailed reports

## 📂 Files in This Week

| File | Purpose |
|------|---------|
| `README.md` | This guide |
| `run_load_test.py` | Automated load testing script |
| `locustfile.py` | Locust configuration for interactive testing |
| `spike_test.py` | Sudden load spike testing |
| `analyze_results.py` | Results analysis and visualization |
| `final_teardown_checklist.md` | **Complete teardown checklist** |
| `verify_cleanup.sh` | **Automated cleanup verification** |

## 📊 Understanding Results

### Good Performance Indicators

✅ **Healthy system**:
- P95 latency < 5s for 50-token generation
- Error rate < 1%
- Throughput scales linearly until saturation
- GPU utilization 80-95%

⚠️ **Warning signs**:
- Latency increases exponentially with load
- Error rate > 5%
- Throughput plateaus early
- GPU underutilized (<50%)

### Example Output

```
Load Test Results (100 users, 5 minutes)
========================================
Requests:
  Total: 1,234
  Successful: 1,220
  Failed: 14
  Error rate: 1.13%

Latency (seconds):
  P50: 2.3
  P95: 4.7
  P99: 6.2
  Max: 8.9

Throughput:
  Requests/sec: 4.1
  Tokens/sec: 205

Recommendations:
  ✅ System handles 100 concurrent users
  ⚠️  P99 latency is high - consider scaling
```

## 🧪 Load Testing Best Practices

### 1. Start Small

- Begin with 10-20 users
- Gradually increase
- Find natural saturation point

### 2. Realistic Workload

- Use varied prompt lengths
- Mix short and long requests
- Simulate real user behavior

### 3. Monitor Everything

- GPU utilization (nvidia-smi)
- Container logs (kubectl logs)
- System metrics (CloudWatch/Prometheus)
- Network throughput

### 4. Document Findings

- Record all metrics
- Note when degradation starts
- Identify bottlenecks
- Plan for scaling

## 💰 Cost Analysis

After load testing, calculate production costs:

### Example Calculation

**Assumptions**:
- 1000 requests/day
- Average 50 tokens/request
- 50,000 tokens/day

**AWS EKS (g5.xlarge)**:
- Instance: $1.00/hour × 24 = $24/day
- EKS control: $0.10/hour × 24 = $2.40/day
- LoadBalancer: $0.02/hour × 24 = $0.48/day
- **Total: ~$27/day or ~$810/month**

**Runpod (A10G)**:
- On-demand: $0.79/hour × 24 = $18.96/day
- Spot pricing: ~$0.40/hour × 24 = $9.60/day
- **Total: ~$10-19/day or ~$300-570/month**

**Per-request cost**:
- EKS: $27 ÷ 1000 = $0.027/request
- Runpod: $10 ÷ 1000 = $0.01/request

## 🛑 FINAL TEARDOWN CHECKLIST

**CRITICAL**: Before finishing this course, ensure ALL resources are deleted!

### Automated Verification

```bash
./verify_cleanup.sh
```

This checks:
- [ ] No Runpod pods running
- [ ] No EKS clusters active
- [ ] No EC2 instances running
- [ ] No LoadBalancers active
- [ ] No EBS volumes available
- [ ] No ECR repositories (optional)

### Manual Verification

#### Runpod
1. Go to https://runpod.io
2. Check "Pods" tab → Should be empty
3. Check "Serverless" tab → Should be empty
4. Check billing to ensure no active charges

#### AWS
1. **EKS**: Console → EKS → No clusters
2. **EC2**: Console → EC2 → Instances → All terminated
3. **ELB**: Console → EC2 → Load Balancers → None
4. **Volumes**: Console → EC2 → Volumes → No "available" volumes
5. **Billing**: Console → Billing Dashboard → Check costs stopped

### Final Cleanup Commands

```bash
# Runpod - already covered in Week 6

# AWS - if missed anything
cd ../week6-deployment
./teardown_eks.sh

# Verify
./verify_cleanup.sh
```

## 📝 Final Deliverables

### 1. Performance Report

Document your findings in `performance_report.md`:
- Baseline metrics (Week 2)
- Optimized metrics (Week 3)
- Multi-GPU scaling (Week 5)
- Load test results (Week 7)
- Cost analysis
- Production recommendations

### 2. Optimal Configuration

Save your best configuration:

```python
# production_config.py
optimal_config = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    
    # Single GPU (Week 3)
    "gpu_memory_utilization": 0.95,
    "max_num_seqs": 512,
    
    # Multi-GPU (Week 5)
    "tensor_parallel_size": 4,  # if using 4 GPUs
    
    # Production settings
    "max_model_len": 8192,      # context length
    "trust_remote_code": True,
}

# Expected performance:
# - Throughput: 500-800 tokens/sec (single GPU)
# - Latency (P95): 2-4 seconds
# - Concurrent users: 50-100
```

### 3. Lessons Learned

Key takeaways:
1. Most impactful optimization? _____________
2. Biggest challenge? _____________
3. Would you use in production? _____________
4. Alternative approaches considered? _____________

## ✅ Week 7 Checklist

Final tasks:

- [ ] Run baseline load test (50 users)
- [ ] Run stress test (100 users)
- [ ] Run spike test
- [ ] Analyzed results with `analyze_results.py`
- [ ] Documented performance in `performance_report.md`
- [ ] Calculated production costs
- [ ] **Ran `verify_cleanup.sh` successfully**
- [ ] **Verified Runpod shows no active pods**
- [ ] **Verified AWS shows no resources**
- [ ] **Checked billing dashboards**
- [ ] Created optimal configuration file
- [ ] Documented lessons learned

## 🎓 Course Completion

Congratulations! You've completed the LLM Inference Optimization course!

### What You've Learned

1. **Week 1**: Environment setup and vLLM basics
2. **Week 2**: Performance profiling and baseline metrics
3. **Week 3**: GPU optimization and parameter tuning
4. **Week 4**: Advanced features and production patterns
5. **Week 5**: Multi-GPU and distributed inference
6. **Week 6**: Cloud deployment (Runpod and EKS)
7. **Week 7**: Load testing and production validation

### Skills Acquired

✅ LLM inference optimization
✅ GPU performance tuning
✅ Distributed computing
✅ Container orchestration (Kubernetes)
✅ Cloud deployment and management
✅ Production monitoring and testing
✅ Cost optimization

### Next Steps

- Deploy a real application
- Experiment with other models (Llama, Mistral, etc.)
- Try quantization (4-bit, 8-bit)
- Implement autoscaling
- Add monitoring (Prometheus, Grafana)
- Optimize for specific use cases

## 📖 Additional Resources

- [vLLM Production Best Practices](https://docs.vllm.ai/en/latest/)
- [Kubernetes Production Patterns](https://kubernetes.io/docs/concepts/)
- [Load Testing Guide](https://locust.io/documentation)

## 🆘 Support

If you encounter issues:
1. Check teardown scripts ran successfully
2. Manually verify in cloud dashboards
3. Contact cloud support if unexpected charges
4. Review week-specific READMEs for troubleshooting

---

**🎉 Congratulations on completing the course!**

Remember: Always tear down cloud resources immediately after use!

