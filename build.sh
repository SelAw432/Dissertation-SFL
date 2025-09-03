#!/bin/bash

set -e

AWS_ACCOUNT_ID="838814607172"
AWS_REGION="eu-west-1"
ECR_REPOSITORY_NAME="aggregator-repo"
IMAGE_TAG="v1.0.0"

# Create a new builder instance for amd64
docker buildx create --name amd64_builder --platform linux/amd64 --use

# Login to ECR
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names ${ECR_REPOSITORY_NAME} --region ${AWS_REGION} || aws ecr create-repository --repository-name ${ECR_REPOSITORY_NAME} --region ${AWS_REGION}

# Remove any existing images
docker rmi ${ECR_REPOSITORY_NAME}:${IMAGE_TAG} 2>/dev/null || true
docker rmi ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY_NAME}:${IMAGE_TAG} 2>/dev/null || true

# Build image for x86_64 only
docker buildx build \
  --builder amd64_builder \
  --platform linux/amd64 \
  --load \
  -t ${ECR_REPOSITORY_NAME}:${IMAGE_TAG} .

# Tag image
docker tag ${ECR_REPOSITORY_NAME}:${IMAGE_TAG} ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY_NAME}:${IMAGE_TAG}

# Push image
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY_NAME}:${IMAGE_TAG}

# Clean up the builder
docker buildx rm amd64_builder

echo "Image URI: ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY_NAME}:${IMAGE_TAG}"