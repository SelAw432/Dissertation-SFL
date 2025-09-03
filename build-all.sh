#!/bin/bash

set -e

# Configuration
AWS_ACCOUNT_ID="838814607172"
TARGET_REGIONS=("us-west-1" "us-west-2" "eu-west-1")  # Your federated learning regions
# TARGET_REGIONS=("us-west-1" "us-west-2")  # Your federated learning regions
IMAGE_TAG="${1:-latest}"  # Allow tag override via command line

# Define all Lambda functions to build
LAMBDA_FUNCTIONS=(
    # "orchestrator"
    "aggregator"
    # "model-copier"
    # "inference"
)

echo "=============================================="
echo "🌍 Federated Learning - Multi-Regional Build"
echo "=============================================="
echo "AWS Account: $AWS_ACCOUNT_ID"
echo "Target Regions: ${TARGET_REGIONS[@]}"
echo "Image Tag: $IMAGE_TAG"
echo "Functions to build: ${LAMBDA_FUNCTIONS[@]}"
echo "=============================================="
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed"
    exit 1
fi

if [ ! -d "lambdas" ]; then
    echo "❌ 'lambdas' directory not found. Run this script from project root."
    exit 1
fi
echo "   ✅ Prerequisites check passed"
echo ""

# Create buildx builder
echo "🛠️  Setting up Docker buildx..."
docker buildx create --name amd64_builder --platform linux/amd64 --use 2>/dev/null || docker buildx use amd64_builder
echo "   ✅ Docker buildx builder ready"
echo ""

# Build counters
total_success=0
total_operations=$((${#LAMBDA_FUNCTIONS[@]} * ${#TARGET_REGIONS[@]}))
failed_operations=()

for region in "${TARGET_REGIONS[@]}"; do
    BASE_ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${region}.amazonaws.com"
    FULL_IMAGE_URI="${BASE_ECR_URI}/${ECR_REPOSITORY_NAME}:${IMAGE_TAG}"
        
    echo "      🔐 Logging in to ECR in $region..."
    if aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${region}.amazonaws.com; then
        echo "      ✅ ECR login successful"
    else
        echo "      ❌ ECR login failed for $region"
        failed_operations+=("${function_name}@${region}")
        continue
    fi   
done

# Build and push each function to all regions
for function_name in "${LAMBDA_FUNCTIONS[@]}"; do
    echo "🏗️  Processing function: $function_name"
    
    # Check if function directory exists
    if [ ! -d "lambdas/$function_name" ]; then
        echo "   ⚠️  Directory 'lambdas/$function_name' not found, skipping..."
        # Mark all regions as failed for this function
        for region in "${TARGET_REGIONS[@]}"; do
            failed_operations+=("${function_name}@${region}")
        done
        continue
    fi
    
    # Check if Dockerfile exists
    if [ ! -f "lambdas/$function_name/Dockerfile" ]; then
        echo "   ⚠️  Dockerfile not found in 'lambdas/$function_name', skipping..."
        # Mark all regions as failed for this function
        for region in "${TARGET_REGIONS[@]}"; do
            failed_operations+=("${function_name}@${region}")
        done
        continue
    fi
    
    ECR_REPOSITORY_NAME="fedlearn-${function_name}"
    
    echo "   📦 Repository: $ECR_REPOSITORY_NAME"
    echo "   🔨 Building Docker image..."
    
    for region in "${TARGET_REGIONS[@]}"; do
        BASE_ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${region}.amazonaws.com"
        FULL_IMAGE_URI="${BASE_ECR_URI}/${ECR_REPOSITORY_NAME}:${IMAGE_TAG}"
        
        echo "      🔧 Ensuring ECR repository exists in $region..."
        aws ecr describe-repositories --repository-names ${ECR_REPOSITORY_NAME} --region ${region} >/dev/null 2>&1 || {
            echo "      📦 Creating ECR repository: $ECR_REPOSITORY_NAME"
            aws ecr create-repository --repository-name ${ECR_REPOSITORY_NAME} --region ${region} >/dev/null
        }
    done

    # Clean up local images for this function
    docker rmi ${ECR_REPOSITORY_NAME}:${IMAGE_TAG} 2>/dev/null || true
    for region in "${TARGET_REGIONS[@]}"; do
        docker rmi ${AWS_ACCOUNT_ID}.dkr.ecr.${region}.amazonaws.com/${ECR_REPOSITORY_NAME}:${IMAGE_TAG} 2>/dev/null || true
    done

    # Build image once (will be tagged for all regions)
    if docker buildx build \
        --builder amd64_builder \
        --platform linux/amd64 \
        --load \
        -t ${ECR_REPOSITORY_NAME}:${IMAGE_TAG} \
        lambdas/${function_name}; then
        echo "   ✅ Build successful for $function_name"
    else
        echo "   ❌ Build failed for $function_name"
        # Mark all regions as failed for this function
        for region in "${TARGET_REGIONS[@]}"; do
            failed_operations+=("${function_name}@${region}")
        done
        continue
    fi
    
    for region in "${TARGET_REGIONS[@]}"; do
        # Tag for this region
        echo "      🏷️  Tagging for ECR..."
        docker tag ${ECR_REPOSITORY_NAME}:${IMAGE_TAG} ${AWS_ACCOUNT_ID}.dkr.ecr.${region}.amazonaws.com/${ECR_REPOSITORY_NAME}:${IMAGE_TAG}

            
        # Push to this region
        echo "      📤 Pushing to ECR..."
        if docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${region}.amazonaws.com/${ECR_REPOSITORY_NAME}:${IMAGE_TAG}; then
            echo "      ✅ Successfully pushed ${function_name} to ${region}"
            echo "      🔗 Image URI: ${AWS_ACCOUNT_ID}.dkr.ecr.${region}.amazonaws.com/${ECR_REPOSITORY_NAME}:${IMAGE_TAG}"
        else
            echo "      ❌ Failed to push ${function_name} to ${region}"
            failed_operations+=("${function_name}@${region}")
        fi
    done
    
    echo ""
    echo "   ✅ Completed processing $function_name"
    echo ""
done

# Clean up builder
echo "🧹 Cleaning up Docker buildx builder..."
docker buildx rm amd64_builder 2>/dev/null || true
echo ""

# Summary
echo "=============================================="
echo "📊 Multi-Regional Build Summary:"
echo "   ✅ Successful operations: $total_success/$total_operations"
echo "   ❌ Failed operations: ${#failed_operations[@]}"
if [ ${#failed_operations[@]} -gt 0 ]; then
    echo "   Failed details:"
    for failed_op in "${failed_operations[@]}"; do
        echo "      - $failed_op"
    done
fi
echo "=============================================="
echo ""

if [ $total_success -gt 0 ]; then
    echo "🎉 Multi-regional build completed!"
    echo ""
    echo "📋 Built images by region:"
    for region in "${TARGET_REGIONS[@]}"; do
        echo "   Region: $region"
        BASE_ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${region}.amazonaws.com"
        for function_name in "${LAMBDA_FUNCTIONS[@]}"; do
            # Check if this combination was successful
            if [[ ! " ${failed_operations[@]} " =~ " ${function_name}@${region} " ]]; then
                ECR_REPOSITORY_NAME="fl-${function_name}"
                echo "      ${BASE_ECR_URI}/${ECR_REPOSITORY_NAME}:${IMAGE_TAG}"
            fi
        done
        echo ""
    done
    
    echo "💡 Next steps:"
    echo "   1. Update CloudFormation templates with region-specific image URIs"
    echo "   2. Deploy to each region:"
    for region in "${TARGET_REGIONS[@]}"; do
        echo "      ./scripts/deploy-cloudformation.sh $region"
    done
fi

# Exit with error if any operations failed
if [ ${#failed_operations[@]} -gt 0 ]; then
    exit 1
fi