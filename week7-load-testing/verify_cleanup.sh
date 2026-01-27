#!/bin/bash

#######################################################################
# Week 7: Complete Cleanup Verification Script
#
# This script verifies that ALL cloud resources have been properly
# deleted to avoid unexpected charges.
#
# Run this after completing Week 7 (or anytime you want to verify cleanup)
#
# Usage: ./verify_cleanup.sh
#######################################################################

set -e

echo "======================================================================"
echo "Week 7: Complete Cleanup Verification"
echo "======================================================================"
echo ""
echo "This script will check for any remaining cloud resources that"
echo "could be incurring costs."
echo ""

ISSUES_FOUND=0

# Configuration
REGION="us-west-2"
CLUSTER_NAME="llm-inference-cluster"

echo "======================================================================"
echo "Checking AWS Resources"
echo "======================================================================"
echo ""

# Check if AWS CLI is configured
if ! command -v aws &> /dev/null; then
    echo "⚠️  AWS CLI not found - skipping AWS checks"
    echo "   Install: pip install awscli"
    echo ""
else
    echo "🔍 AWS CLI found - checking resources in $REGION..."
    echo ""
    
    # Check 1: EKS Clusters
    echo "1️⃣  Checking for EKS clusters..."
    if command -v eksctl &> /dev/null; then
        CLUSTERS=$(eksctl get cluster --region $REGION 2>&1)
        if echo "$CLUSTERS" | grep -q "No clusters found"; then
            echo "   ✅ No EKS clusters found"
        else
            echo "   ❌ FOUND EKS CLUSTERS:"
            echo "$CLUSTERS"
            ISSUES_FOUND=$((ISSUES_FOUND + 1))
            echo ""
            echo "   Action required:"
            echo "   cd ../week6-deployment && ./teardown_eks.sh"
        fi
    else
        echo "   ⚠️  eksctl not found - checking via AWS CLI..."
        CLUSTERS=$(aws eks list-clusters --region $REGION --query 'clusters' --output text 2>&1)
        if [ -z "$CLUSTERS" ]; then
            echo "   ✅ No EKS clusters found"
        else
            echo "   ❌ FOUND EKS CLUSTERS: $CLUSTERS"
            ISSUES_FOUND=$((ISSUES_FOUND + 1))
            echo "   Action: eksctl delete cluster --name $CLUSTER_NAME --region $REGION"
        fi
    fi
    echo ""
    
    # Check 2: EC2 Instances
    echo "2️⃣  Checking for running EC2 instances..."
    INSTANCES=$(aws ec2 describe-instances --region $REGION \
        --filters "Name=instance-state-name,Values=running,pending,stopping" \
        --query "Reservations[*].Instances[*].[InstanceId,InstanceType,State.Name,Tags[?Key=='Name'].Value|[0]]" \
        --output text 2>&1)
    
    if [ -z "$INSTANCES" ]; then
        echo "   ✅ No running EC2 instances"
    else
        echo "   ❌ FOUND RUNNING INSTANCES:"
        echo "$INSTANCES"
        ISSUES_FOUND=$((ISSUES_FOUND + 1))
        echo ""
        echo "   Action required:"
        echo "   Terminate instances in AWS Console → EC2 → Instances"
    fi
    echo ""
    
    # Check 3: Load Balancers
    echo "3️⃣  Checking for Load Balancers..."
    LBS=$(aws elbv2 describe-load-balancers --region $REGION \
        --query "LoadBalancers[*].[LoadBalancerName,State.Code]" \
        --output text 2>&1)
    
    if [ -z "$LBS" ]; then
        echo "   ✅ No Load Balancers found"
    else
        echo "   ❌ FOUND LOAD BALANCERS:"
        echo "$LBS"
        ISSUES_FOUND=$((ISSUES_FOUND + 1))
        echo ""
        echo "   Action required:"
        echo "   Delete in AWS Console → EC2 → Load Balancers"
    fi
    echo ""
    
    # Check 4: EBS Volumes
    echo "4️⃣  Checking for available EBS volumes..."
    VOLUMES=$(aws ec2 describe-volumes --region $REGION \
        --filters "Name=status,Values=available" \
        --query "Volumes[*].[VolumeId,Size,CreateTime]" \
        --output text 2>&1)
    
    if [ -z "$VOLUMES" ]; then
        echo "   ✅ No orphaned volumes"
    else
        echo "   ⚠️  FOUND AVAILABLE VOLUMES (may be orphaned):"
        echo "$VOLUMES"
        echo ""
        echo "   These may be leftover from deleted instances."
        echo "   Review and delete if not needed:"
        echo "   aws ec2 delete-volume --volume-id vol-xxxxx --region $REGION"
    fi
    echo ""
    
    # Check 5: ECR Repositories
    echo "5️⃣  Checking for ECR repositories..."
    REPOS=$(aws ecr describe-repositories --region $REGION \
        --query "repositories[*].[repositoryName,repositoryUri]" \
        --output text 2>&1)
    
    if [ -z "$REPOS" ]; then
        echo "   ✅ No ECR repositories"
    else
        echo "   ℹ️  Found ECR repositories:"
        echo "$REPOS"
        echo ""
        echo "   These are storage costs only (~\$0.10/GB/month)."
        echo "   Delete if not needed:"
        echo "   aws ecr delete-repository --repository-name REPO_NAME --region $REGION --force"
    fi
    echo ""
    
    # Check 6: Elastic IPs
    echo "6️⃣  Checking for unattached Elastic IPs..."
    EIPS=$(aws ec2 describe-addresses --region $REGION \
        --query "Addresses[?AssociationId==null].[PublicIp,AllocationId]" \
        --output text 2>&1)
    
    if [ -z "$EIPS" ]; then
        echo "   ✅ No unattached Elastic IPs"
    else
        echo "   ⚠️  FOUND UNATTACHED ELASTIC IPs:"
        echo "$EIPS"
        echo ""
        echo "   Unattached IPs cost ~\$0.005/hour."
        echo "   Release them in AWS Console → EC2 → Elastic IPs"
    fi
    echo ""
fi

echo "======================================================================"
echo "Checking Runpod Resources"
echo "======================================================================"
echo ""

echo "⚠️  Runpod verification requires manual check:"
echo ""
echo "   1. Go to: https://runpod.io"
echo "   2. Log in to your account"
echo "   3. Check 'Pods' tab → Should be empty"
echo "   4. Check 'Serverless' tab → Should be empty"
echo "   5. Check billing section → No active charges"
echo ""

read -p "Have you verified Runpod is clean? (yes/no): " RUNPOD_CLEAN

if [ "$RUNPOD_CLEAN" != "yes" ]; then
    echo "   ⚠️  Please verify Runpod manually"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
else
    echo "   ✅ Runpod verified clean"
fi

echo ""
echo "======================================================================"
echo "Billing Check Recommendations"
echo "======================================================================"
echo ""

echo "📊 Check your billing dashboards:"
echo ""
echo "AWS:"
echo "   • Billing Dashboard:"
echo "     https://console.aws.amazon.com/billing/home"
echo "   • Cost Explorer (last 7 days):"
echo "     https://console.aws.amazon.com/cost-management/home#/dashboard"
echo "   • Set up billing alerts:"
echo "     https://console.aws.amazon.com/billing/home#/budgets"
echo ""
echo "Runpod:"
echo "   • Billing: https://www.runpod.io/console/user/billing"
echo ""

echo "======================================================================"
echo "VERIFICATION SUMMARY"
echo "======================================================================"
echo ""

if [ $ISSUES_FOUND -eq 0 ]; then
    echo "✅ ✅ ✅  ALL CHECKS PASSED  ✅ ✅ ✅"
    echo ""
    echo "No cloud resources found that could incur costs."
    echo "You're all set! 🎉"
else
    echo "⚠️ ⚠️ ⚠️  ISSUES FOUND: $ISSUES_FOUND  ⚠️ ⚠️ ⚠️"
    echo ""
    echo "Please review the issues above and take action."
    echo "Resources left running will continue to incur costs!"
fi

echo ""
echo "======================================================================"
echo ""

# Save verification log
LOG_FILE="cleanup_verification_$(date +%Y%m%d_%H%M%S).log"
{
    echo "Cleanup Verification - $(date)"
    echo "Region checked: $REGION"
    echo "Issues found: $ISSUES_FOUND"
    echo ""
    echo "This log can be used as proof of cleanup if needed."
} > "$LOG_FILE"

echo "📝 Verification log saved to: $LOG_FILE"
echo ""

if [ $ISSUES_FOUND -eq 0 ]; then
    exit 0
else
    exit 1
fi

