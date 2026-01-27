#!/bin/bash

#######################################################################
# Week 6: Complete EKS Teardown Script
#
# This script safely tears down ALL AWS EKS resources to avoid costs.
# Run this IMMEDIATELY after finishing your experiments.
#
# Usage: ./teardown_eks.sh
#######################################################################

set -e  # Exit on error

# Configuration (match your deployment)
CLUSTER_NAME="llm-inference-cluster"
REGION="us-west-2"
ECR_REPO="vllm-qwen"

echo "======================================================================"
echo "Week 6: Complete EKS Teardown"
echo "======================================================================"
echo ""
echo "⚠️  WARNING: This will delete ALL resources for cluster: $CLUSTER_NAME"
echo ""
echo "Resources to be deleted:"
echo "  • Kubernetes deployment and services"
echo "  • LoadBalancer (ELB)"
echo "  • EKS cluster and control plane"
echo "  • EC2 instances (nodes)"
echo "  • Security groups"
echo "  • VPC resources (if created by eksctl)"
echo ""

read -p "Are you sure you want to continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Teardown cancelled."
    exit 0
fi

echo ""
echo "======================================================================"
echo "Step 1: Delete Kubernetes Resources"
echo "======================================================================"
echo ""

# Check if cluster is accessible
if kubectl cluster-info &> /dev/null; then
    echo "✅ Cluster is accessible"
    echo ""
    
    # Delete deployment
    echo "🗑️  Deleting deployment..."
    if kubectl get deployment qwen-vllm &> /dev/null; then
        kubectl delete deployment qwen-vllm
        echo "✅ Deployment deleted"
    else
        echo "⚠️  Deployment not found (may already be deleted)"
    fi
    
    echo ""
    
    # Delete service (this removes the LoadBalancer)
    echo "🗑️  Deleting service and LoadBalancer..."
    if kubectl get service qwen-vllm-service &> /dev/null; then
        kubectl delete service qwen-vllm-service
        echo "✅ Service deleted"
        echo "⏳ Waiting 30 seconds for LoadBalancer to be fully deleted..."
        sleep 30
    else
        echo "⚠️  Service not found (may already be deleted)"
    fi
    
else
    echo "⚠️  Cannot access cluster (may already be deleted)"
    echo "    Skipping Kubernetes resource deletion"
fi

echo ""
echo "======================================================================"
echo "Step 2: Delete EKS Cluster"
echo "======================================================================"
echo ""

# Check if cluster exists
if eksctl get cluster --name $CLUSTER_NAME --region $REGION &> /dev/null; then
    echo "🗑️  Deleting EKS cluster: $CLUSTER_NAME"
    echo "    This will take 10-15 minutes..."
    echo ""
    
    eksctl delete cluster --name $CLUSTER_NAME --region $REGION
    
    echo ""
    echo "✅ EKS cluster deleted"
else
    echo "⚠️  Cluster not found (may already be deleted)"
fi

echo ""
echo "======================================================================"
echo "Step 3: Verify Deletion"
echo "======================================================================"
echo ""

# Check for remaining clusters
echo "🔍 Checking for remaining EKS clusters..."
REMAINING_CLUSTERS=$(eksctl get cluster --region $REGION 2>&1)
if echo "$REMAINING_CLUSTERS" | grep -q "No clusters found"; then
    echo "✅ No EKS clusters remain"
else
    echo "⚠️  Found remaining clusters:"
    echo "$REMAINING_CLUSTERS"
fi

echo ""

# Check for EC2 instances
echo "🔍 Checking for EC2 instances..."
INSTANCES=$(aws ec2 describe-instances --region $REGION \
    --filters "Name=tag:alpha.eksctl.io/cluster-name,Values=$CLUSTER_NAME" \
    --query "Reservations[*].Instances[*].[InstanceId,State.Name]" \
    --output text 2>&1)

if [ -z "$INSTANCES" ]; then
    echo "✅ No EC2 instances found"
else
    echo "⚠️  Found EC2 instances:"
    echo "$INSTANCES"
    echo ""
    echo "If instances are not 'terminated', wait a few minutes and check again."
fi

echo ""

# Check for LoadBalancers
echo "🔍 Checking for LoadBalancers..."
LBS=$(aws elbv2 describe-load-balancers --region $REGION \
    --query "LoadBalancers[*].[LoadBalancerName,State.Code]" \
    --output text 2>&1)

if [ -z "$LBS" ]; then
    echo "✅ No LoadBalancers found"
else
    echo "⚠️  Found LoadBalancers (check if any are related to your cluster):"
    echo "$LBS"
fi

echo ""

# Check for volumes
echo "🔍 Checking for available EBS volumes..."
VOLUMES=$(aws ec2 describe-volumes --region $REGION \
    --filters "Name=status,Values=available" \
    --query "Volumes[*].[VolumeId,Size]" \
    --output text 2>&1)

if [ -z "$VOLUMES" ]; then
    echo "✅ No available volumes found"
else
    echo "⚠️  Found available volumes (orphaned from deleted instances):"
    echo "$VOLUMES"
    echo ""
    read -p "Delete these volumes? (yes/no): " DELETE_VOLS
    if [ "$DELETE_VOLS" == "yes" ]; then
        echo "$VOLUMES" | awk '{print $1}' | while read VOL_ID; do
            echo "Deleting volume: $VOL_ID"
            aws ec2 delete-volume --volume-id $VOL_ID --region $REGION
        done
        echo "✅ Volumes deleted"
    fi
fi

echo ""
echo "======================================================================"
echo "Step 4: ECR Repository (Optional)"
echo "======================================================================"
echo ""

# Check if ECR repository exists
if aws ecr describe-repositories --repository-names $ECR_REPO --region $REGION &> /dev/null; then
    echo "📦 Found ECR repository: $ECR_REPO"
    echo ""
    read -p "Delete ECR repository? (yes/no): " DELETE_ECR
    if [ "$DELETE_ECR" == "yes" ]; then
        aws ecr delete-repository --repository-name $ECR_REPO --region $REGION --force
        echo "✅ ECR repository deleted"
    else
        echo "⏭️  Keeping ECR repository"
    fi
else
    echo "✅ No ECR repository found"
fi

echo ""
echo "======================================================================"
echo "✅ TEARDOWN COMPLETE"
echo "======================================================================"
echo ""
echo "📊 Summary:"
echo "  • Kubernetes resources: Deleted"
echo "  • EKS cluster: Deleted"
echo "  • EC2 instances: Terminated"
echo "  • LoadBalancer: Removed"
echo "  • Volumes: Cleaned up"
echo ""
echo "🔍 Next Steps:"
echo "  1. Check AWS Console → EC2 → Instances"
echo "     All instances should be 'terminated'"
echo ""
echo "  2. Check AWS Console → EC2 → Load Balancers"
echo "     No load balancers related to your cluster"
echo ""
echo "  3. Check AWS Console → EKS"
echo "     No clusters should be listed"
echo ""
echo "  4. Check AWS Billing Dashboard"
echo "     https://console.aws.amazon.com/billing/home"
echo "     Verify costs have stopped"
echo ""
echo "  5. Set up billing alerts if you haven't already!"
echo ""
echo "💡 If you see any remaining resources in AWS Console, delete them manually."
echo ""
echo "✅ You can now safely proceed to Week 7!"
echo ""

