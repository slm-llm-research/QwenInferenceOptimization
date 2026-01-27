# Week 6: Deploying a Production-Ready Inference System

## 🎯 Goals

By the end of this week, you will:
- Containerize your vLLM inference system with Docker
- Deploy on Runpod as a serverless GPU endpoint
- Deploy on AWS EKS (Kubernetes) with GPU nodes
- Understand production deployment considerations
- **Master complete teardown procedures to avoid costs**

## ⚠️ COST WARNING

**This week involves cloud resources that WILL charge you money!**

- **Runpod**: ~$0.50-$2.00/hour depending on GPU
- **AWS EKS**: ~$0.10/hour (control plane) + EC2 instance costs (~$1-10/hour)
- **AWS ELB**: ~$0.02/hour + data transfer

**CRITICAL**: Always follow teardown procedures immediately after testing!

## 📚 What You'll Learn

- Docker containerization for ML models
- Serverless GPU deployment (Runpod)
- Kubernetes fundamentals for ML deployment
- AWS EKS cluster management
- Production monitoring and logging
- **Complete infrastructure teardown**

## 🐳 Part 1: Containerization

### Step 1: Build Docker Image

```bash
cd docker/
docker build -t vllm-qwen:latest .
```

**What's in the Dockerfile**:
- NVIDIA CUDA base image
- PyTorch and vLLM installation
- Model files (or downloads on startup)
- Startup script for vLLM server

### Step 2: Test Locally (Optional)

If you have Docker with GPU support:

```bash
docker run --gpus all -p 8000:8000 vllm-qwen:latest
```

### Step 3: Push to Registry

For deployment, push to Docker Hub or AWS ECR:

```bash
# Docker Hub
docker tag vllm-qwen:latest yourusername/vllm-qwen:latest
docker push yourusername/vllm-qwen:latest

# Or AWS ECR (see deploy_eks.sh for automated ECR setup)
```

## 🚀 Part 2: Runpod Deployment

### Setup

1. **Create Runpod account**: https://runpod.io
2. **Add payment method** (required for GPU pods)
3. **Get API key** (optional, for automation)

### Deploy via UI

Run the deployment helper:

```bash
python deploy_runpod.py
```

This guides you through:
1. Model selection
2. GPU type choice
3. Environment configuration
4. Endpoint creation

**Or manually via Runpod UI**:
1. Go to Runpod → Serverless → Deploy
2. Select "vLLM" template
3. Configure:
   - Model: `Qwen/Qwen2.5-7B-Instruct`
   - GPU: A10G or L4 (cheapest)
   - Max workers: 1-3
4. Deploy

### Test Deployment

```bash
python test_runpod_endpoint.py --endpoint-id YOUR_ENDPOINT_ID
```

### 🛑 TEARDOWN RUNPOD

**IMMEDIATELY after testing**:

```bash
python teardown_runpod.py --endpoint-id YOUR_ENDPOINT_ID
```

**Or manually**:
1. Runpod Dashboard → Serverless
2. Find your endpoint
3. Click "Terminate"
4. **Confirm termination**
5. **Verify it shows "Terminated" status**

**Verification**:
- No active endpoints in dashboard
- Check billing to ensure no ongoing charges

## ☸️ Part 3: AWS EKS Deployment

### Prerequisites

Install required tools:

```bash
# AWS CLI
pip install awscli
aws configure  # Enter your credentials

# eksctl
brew install eksctl  # macOS
# Or: curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
# sudo mv /tmp/eksctl /usr/local/bin

# kubectl
brew install kubectl  # macOS
# Or: curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# Verify installations
aws --version
eksctl version
kubectl version --client
```

### Deploy EKS Cluster

**Option 1: Automated (Recommended)**

```bash
./deploy_eks.sh
```

This script:
1. Creates ECR repository for Docker image
2. Builds and pushes Docker image
3. Creates EKS cluster with GPU nodes
4. Installs NVIDIA device plugin
5. Deploys vLLM service
6. Exposes via LoadBalancer

**Option 2: Manual**

Follow the detailed steps in `MANUAL_EKS_SETUP.md`

### Deployment Configuration

The script creates:
- **Cluster**: `llm-inference-cluster`
- **Region**: `us-west-2` (configurable)
- **Node type**: `g5.xlarge` (1x A10G GPU)
- **Nodes**: 1 (min: 1, max: 3 for autoscaling)

### Test EKS Deployment

```bash
python test_eks_endpoint.py
```

This will:
1. Get the LoadBalancer URL
2. Send test requests
3. Verify responses
4. Measure latency

### Monitor Deployment

```bash
# Check pods
kubectl get pods -o wide

# Check service
kubectl get svc

# View logs
kubectl logs -l app=qwen-vllm --follow

# Check GPU usage (from within pod)
kubectl exec -it <pod-name> -- nvidia-smi
```

## 🛑 COMPLETE TEARDOWN PROCEDURES

### AWS EKS Teardown

**CRITICAL**: This is the most important section to avoid costs!

#### Option 1: Automated Teardown

```bash
./teardown_eks.sh
```

This removes:
1. Kubernetes deployment and service
2. LoadBalancer (ELB)
3. EKS cluster
4. EC2 instances (nodes)
5. Security groups
6. VPC (if created by eksctl)
7. ECR repository (optional)

**Verification steps are included in the script!**

#### Option 2: Manual Teardown

**Step 1: Delete Kubernetes resources**

```bash
# Delete deployment
kubectl delete deployment qwen-vllm

# Delete service (removes LoadBalancer)
kubectl delete service qwen-vllm-service

# Wait for LoadBalancer to be deleted
kubectl get svc
# Should show no LoadBalancer or service deleted
```

**Step 2: Delete EKS cluster**

```bash
eksctl delete cluster --name llm-inference-cluster --region us-west-2
```

This takes 10-15 minutes and removes:
- All nodes (EC2 instances)
- Node groups
- EKS control plane
- Associated VPC and subnets (if created by eksctl)

**Step 3: Verify deletion**

```bash
# Check no clusters remain
eksctl get cluster

# Check no EC2 instances
aws ec2 describe-instances --region us-west-2 \
  --filters "Name=tag:alpha.eksctl.io/cluster-name,Values=llm-inference-cluster" \
  --query "Reservations[*].Instances[*].[InstanceId,State.Name]"
  
# Should return empty or all "terminated"

# Check no LoadBalancers
aws elbv2 describe-load-balancers --region us-west-2
# Look for any related to your cluster

# Check no volumes
aws ec2 describe-volumes --region us-west-2 \
  --filters "Name=status,Values=available"
# Delete any orphaned volumes:
# aws ec2 delete-volume --volume-id vol-xxxxx
```

**Step 4: Clean up ECR (optional)**

```bash
# List repositories
aws ecr describe-repositories --region us-west-2

# Delete your repository
aws ecr delete-repository --repository-name vllm-qwen --region us-west-2 --force
```

**Step 5: Check billing**

- Go to AWS Console → Billing Dashboard
- Check "Bills" for current charges
- Check "Cost Explorer" for daily costs
- Set up billing alerts!

### Complete Teardown Checklist

Use this checklist after each deployment session:

```bash
./verify_complete_teardown.sh
```

**Manual checklist**:

- [ ] Runpod endpoints terminated
- [ ] Runpod shows no active pods
- [ ] Kubernetes deployment deleted (`kubectl get deploy`)
- [ ] Kubernetes service deleted (`kubectl get svc`)
- [ ] LoadBalancer deleted (AWS Console → EC2 → Load Balancers)
- [ ] EKS cluster deleted (`eksctl get cluster` returns empty)
- [ ] EC2 instances terminated (AWS Console → EC2 → Instances)
- [ ] No "available" EBS volumes (AWS Console → EC2 → Volumes)
- [ ] ECR repository deleted (if no longer needed)
- [ ] Billing dashboard checked
- [ ] Cost Explorer shows costs stopped

## 📂 Files in This Week

| File | Purpose |
|------|---------|
| `README.md` | This guide (you are here) |
| `docker/Dockerfile` | Container definition |
| `docker/entrypoint.sh` | Container startup script |
| `deploy_runpod.py` | Runpod deployment helper |
| `test_runpod_endpoint.py` | Test Runpod endpoint |
| `teardown_runpod.py` | Runpod teardown script |
| `deploy_eks.sh` | Automated EKS deployment |
| `teardown_eks.sh` | **Automated EKS teardown** |
| `verify_complete_teardown.sh` | Verify all resources deleted |
| `test_eks_endpoint.py` | Test EKS deployment |
| `kubernetes/deployment.yaml` | Kubernetes manifests |
| `MANUAL_EKS_SETUP.md` | Step-by-step manual EKS guide |
| `TEARDOWN_GUIDE.md` | **Detailed teardown instructions** |

## 💰 Cost Estimates (as of 2026)

### Runpod
- **A10G**: ~$0.79/hour
- **L4**: ~$0.59/hour  
- **A100 (40GB)**: ~$1.89/hour

### AWS EKS
- **Control Plane**: $0.10/hour (~$73/month)
- **g5.xlarge** (A10G): ~$1.00/hour
- **LoadBalancer**: ~$0.02/hour + data transfer
- **Total**: ~$1.12/hour (plus storage and data transfer)

**For 1 hour testing**: ~$1-2
**For 1 day**: ~$25-50 (DON'T leave running!)

## 🐛 Troubleshooting

### Docker Issues

**Issue**: "CUDA not available in container"
**Solution**: Ensure `--gpus all` flag and NVIDIA Docker runtime installed

**Issue**: "Model download fails in container"
**Solution**: Check internet connectivity, or pre-download model into image

### Runpod Issues

**Issue**: "Endpoint stays in 'Initializing' state"
**Solution**: Check logs, may need to increase timeout or use smaller model

**Issue**: "Cold start is very slow"
**Solution**: Enable "Workers Always Running" (increases cost)

### EKS Issues

**Issue**: "`kubectl` commands fail"
**Solution**: Update kubeconfig: `aws eks update-kubeconfig --name llm-inference-cluster --region us-west-2`

**Issue**: "Pods stuck in Pending"
**Solution**: Check node status: `kubectl describe node`. Ensure GPU nodes are Ready.

**Issue**: "NVIDIA device plugin not working"
**Solution**: Reinstall: `kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml`

**Issue**: "LoadBalancer stuck in Pending"
**Solution**: Check security groups and VPC configuration. May take 2-3 minutes.

### Teardown Issues

**Issue**: "eksctl delete cluster hangs"
**Solution**: 
- Cancel with Ctrl+C
- Manually delete from AWS Console → EKS
- Then delete CloudFormation stacks

**Issue**: "Load Balancer not deleted"
**Solution**: AWS Console → EC2 → Load Balancers → Manual deletion

**Issue**: "Volumes remain after cluster deletion"
**Solution**: AWS Console → EC2 → Volumes → Delete orphaned volumes

## 📖 Additional Resources

- [vLLM Docker Images](https://docs.vllm.ai/en/latest/deployment/docker.html)
- [Runpod Documentation](https://docs.runpod.io/)
- [AWS EKS User Guide](https://docs.aws.amazon.com/eks/latest/userguide/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

## ✅ Week 6 Checklist

Before moving to Week 7:

- [ ] Built Docker image successfully
- [ ] Deployed to Runpod and tested
- [ ] **Torn down Runpod endpoint**
- [ ] Installed AWS CLI, eksctl, kubectl
- [ ] Deployed EKS cluster with GPU nodes
- [ ] Tested EKS endpoint
- [ ] **Completely torn down EKS cluster**
- [ ] Verified no resources remain
- [ ] Understood cost implications
- [ ] Checked AWS billing dashboard

## 🔜 Next Steps

Ready for **Week 7: Load Testing & Final Teardown**!

Week 7 covers:
- Load testing with 100+ concurrent users
- Performance analysis under load
- Final production recommendations
- Complete teardown verification

---

**Remember**: Always tear down resources immediately after experimentation!

Set billing alerts: AWS Console → Billing → Budgets → Create Budget

