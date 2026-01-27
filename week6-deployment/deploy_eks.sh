#!/bin/bash

#######################################################################
# Week 6: Automated EKS Deployment Script
#
# This script automates the deployment of vLLM on AWS EKS with GPU support.
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - eksctl installed
#   - kubectl installed
#   - Docker installed (for building image)
#
# Usage: ./deploy_eks.sh
#######################################################################

set -e  # Exit on error

# Configuration
CLUSTER_NAME="llm-inference-cluster"
REGION="us-west-2"
NODE_TYPE="g5.xlarge"  # 1x A10G GPU, ~$1/hour
MIN_NODES=1
MAX_NODES=3
ECR_REPO="vllm-qwen"
IMAGE_TAG="latest"

echo "======================================================================"
echo "Week 6: Automated EKS Deployment"
echo "======================================================================"
echo ""
echo "This script will:"
echo "  1. Create ECR repository"
echo "  2. Build and push Docker image"
echo "  3. Create EKS cluster with GPU nodes"
echo "  4. Install NVIDIA device plugin"
echo "  5. Deploy vLLM service"
echo "  6. Expose via LoadBalancer"
echo ""
echo "Configuration:"
echo "  Cluster: $CLUSTER_NAME"
echo "  Region: $REGION"
echo "  Node type: $NODE_TYPE"
echo "  Nodes: $MIN_NODES-$MAX_NODES"
echo ""
echo "⚠️  Estimated cost: ~$1-2/hour while running"
echo "    REMEMBER TO RUN teardown_eks.sh WHEN DONE!"
echo ""

read -p "Continue? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Deployment cancelled."
    exit 0
fi

echo ""
echo "======================================================================"
echo "Step 1: Create ECR Repository"
echo "======================================================================"
echo ""

# Create ECR repository if it doesn't exist
if aws ecr describe-repositories --repository-names $ECR_REPO --region $REGION &> /dev/null; then
    echo "✅ ECR repository already exists: $ECR_REPO"
else
    echo "📦 Creating ECR repository: $ECR_REPO"
    aws ecr create-repository --repository-name $ECR_REPO --region $REGION
    echo "✅ ECR repository created"
fi

# Get ECR login
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$REGION.amazonaws.com
echo "✅ ECR login successful"

echo ""
echo "======================================================================"
echo "Step 2: Build and Push Docker Image"
echo "======================================================================"
echo ""

ECR_URI=$(aws ecr describe-repositories --repository-names $ECR_REPO --region $REGION --query 'repositories[0].repositoryUri' --output text)
IMAGE_URI="$ECR_URI:$IMAGE_TAG"

echo "🐳 Building Docker image..."
echo "   Image: $IMAGE_URI"
cd docker/
docker build -t $IMAGE_URI .
echo "✅ Docker image built"

echo ""
echo "📤 Pushing image to ECR..."
docker push $IMAGE_URI
echo "✅ Image pushed"

cd ..

echo ""
echo "======================================================================"
echo "Step 3: Create EKS Cluster"
echo "======================================================================"
echo ""

# Check if cluster already exists
if eksctl get cluster --name $CLUSTER_NAME --region $REGION &> /dev/null; then
    echo "✅ Cluster already exists: $CLUSTER_NAME"
    echo "   Skipping cluster creation"
else
    echo "🚀 Creating EKS cluster: $CLUSTER_NAME"
    echo "   This will take 15-20 minutes..."
    echo ""
    
    eksctl create cluster \
        --name $CLUSTER_NAME \
        --region $REGION \
        --node-type $NODE_TYPE \
        --nodes $MIN_NODES \
        --nodes-min $MIN_NODES \
        --nodes-max $MAX_NODES \
        --with-oidc \
        --managed
    
    echo "✅ EKS cluster created"
fi

# Update kubeconfig
echo ""
echo "🔧 Updating kubeconfig..."
aws eks update-kubeconfig --name $CLUSTER_NAME --region $REGION
echo "✅ kubeconfig updated"

echo ""
echo "======================================================================"
echo "Step 4: Install NVIDIA Device Plugin"
echo "======================================================================"
echo ""

echo "🎮 Installing NVIDIA device plugin for GPU support..."
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

echo "⏳ Waiting for device plugin to be ready..."
sleep 10
kubectl wait --for=condition=ready pod -l name=nvidia-device-plugin-ds -n kube-system --timeout=120s

echo "✅ NVIDIA device plugin installed"

echo ""
echo "======================================================================"
echo "Step 5: Deploy vLLM Service"
echo "======================================================================"
echo ""

# Create namespace if needed
kubectl create namespace vllm --dry-run=client -o yaml | kubectl apply -f -

# Update deployment manifest with ECR image
echo "📝 Creating deployment manifest..."
cat > kubernetes/deployment-generated.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen-vllm
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: qwen-vllm
  template:
    metadata:
      labels:
        app: qwen-vllm
    spec:
      containers:
      - name: vllm-server
        image: $IMAGE_URI
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: MODEL_NAME
          value: "Qwen/Qwen2.5-7B-Instruct"
---
apiVersion: v1
kind: Service
metadata:
  name: qwen-vllm-service
  namespace: default
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  selector:
    app: qwen-vllm
EOF

echo "🚀 Deploying vLLM service..."
kubectl apply -f kubernetes/deployment-generated.yaml

echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available deployment/qwen-vllm --timeout=600s

echo "✅ vLLM deployment is ready"

echo ""
echo "======================================================================"
echo "Step 6: Get Service Endpoint"
echo "======================================================================"
echo ""

echo "⏳ Waiting for LoadBalancer to be provisioned..."
echo "   This can take 2-3 minutes..."

# Wait for LoadBalancer to get external IP
for i in {1..60}; do
    EXTERNAL_IP=$(kubectl get svc qwen-vllm-service -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    if [ ! -z "$EXTERNAL_IP" ]; then
        break
    fi
    echo "   Attempt $i/60: Still waiting..."
    sleep 5
done

if [ -z "$EXTERNAL_IP" ]; then
    echo "⚠️  LoadBalancer provisioning is taking longer than expected"
    echo "   Check status with: kubectl get svc qwen-vllm-service"
else
    echo "✅ LoadBalancer ready!"
    echo ""
    echo "======================================================================"
    echo "✅ DEPLOYMENT COMPLETE!"
    echo "======================================================================"
    echo ""
    echo "🌐 Service Endpoint: http://$EXTERNAL_IP"
    echo ""
    echo "Test the endpoint:"
    echo "  curl http://$EXTERNAL_IP/v1/completions \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{"
    echo '      "model": "Qwen/Qwen2.5-7B-Instruct",'
    echo '      "prompt": "Hello, world!",'
    echo '      "max_tokens": 20'
    echo "    }'"
    echo ""
    echo "Or run: python test_eks_endpoint.py"
    echo ""
fi

echo "📊 Cluster Information:"
kubectl get nodes -o wide
echo ""
kubectl get pods -o wide
echo ""
kubectl get svc

echo ""
echo "======================================================================"
echo "⚠️  IMPORTANT: Cost Management"
echo "======================================================================"
echo ""
echo "Your cluster is now running and ACCRUING COSTS!"
echo ""
echo "Estimated costs:"
echo "  • EKS control plane: \$0.10/hour"
echo "  • $NODE_TYPE node: ~\$1.00/hour"
echo "  • LoadBalancer: ~\$0.02/hour"
echo "  • Total: ~\$1.12/hour"
echo ""
echo "WHEN YOU'RE DONE TESTING:"
echo "  ./teardown_eks.sh"
echo ""
echo "This will delete ALL resources and stop charges."
echo ""
echo "Set up billing alerts:"
echo "  https://console.aws.amazon.com/billing/home#/budgets"
echo ""

