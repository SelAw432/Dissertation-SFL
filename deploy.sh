#!/bin/bash

# -------------------------------- SET GLOBAL VARIABLES ------------------------------------
AWS_ACCOUNT_ID="838814607172"


#-------------------------------- SET REGIONAL VARIABLES -----------------------------------
echo "=============================================="
echo "🚀 Multi-Region Federated Learning Deployment"
echo "=============================================="
echo "Templates: network-stack.yaml, central-stack.yaml, node-stack.yaml"
echo "Environment: fedlearn"
echo "Regions: EU-WEST-1, US-WEST-1, US-WEST-2"
echo "=============================================="

echo ""
echo "Starting multi-region deployment..."
echo ""

# 1. EU-WEST-1
AWS_REGION="eu-west-1"
TARGET_REGIONS=("us-west-1" "us-west-2" "eu-west-1")  # Your federated learning regions
AGG_REPOSITORY_NAME="fedlearn-aggregator"
INF_REPOSITORY_NAME="fedlearn-inference"
MCO_REPOSITORY_NAME="fedlearn-model-copier"
OCH_REPOSITORY_NAME="fedlearn-orchestrator"

IMAGE_TAG="latest"

STACK_NAME="global-resource-stack"
echo "Deploying: $STACK_NAME..."
aws cloudformation deploy \
  --template-file central-stack.yaml \
  --stack-name ${STACK_NAME} \
  --region ${AWS_REGION}  \
  --capabilities CAPABILITY_NAMED_IAM

for region in "${TARGET_REGIONS[@]}"; do

  echo ""
  echo "🌍 Deploying to region: $region..."
  echo ""

  STACK_NAME="network-stack"
  echo "Deploying: $STACK_NAME..."
  aws cloudformation deploy \
    --template-file network-stack.yaml \
    --stack-name ${STACK_NAME} \
    --region ${region}  \
    --capabilities CAPABILITY_NAMED_IAM

  STACK_NAME="node-stack"
  echo "Deploying: $STACK_NAME..."
  aws cloudformation deploy \
    --template-file node-stack.yaml \
    --stack-name ${STACK_NAME} \
    --parameter-overrides \
      AggregatorImageURI="${AWS_ACCOUNT_ID}.dkr.ecr.${region}.amazonaws.com/${AGG_REPOSITORY_NAME}:${IMAGE_TAG}" \
      InferenceImageURI="${AWS_ACCOUNT_ID}.dkr.ecr.${region}.amazonaws.com/${INF_REPOSITORY_NAME}:${IMAGE_TAG}" \
      ModelCopierImageURI="${AWS_ACCOUNT_ID}.dkr.ecr.${region}.amazonaws.com/${MCO_REPOSITORY_NAME}:${IMAGE_TAG}" \
      OrchestratorImageURI="${AWS_ACCOUNT_ID}.dkr.ecr.${region}.amazonaws.com/${OCH_REPOSITORY_NAME}:${IMAGE_TAG}" \
      UseExistingSageMakerRole="false" \
    --region ${region}  \
    --capabilities CAPABILITY_NAMED_IAM
done

for region in "${TARGET_REGIONS[@]}"; do
  echo ""
  echo "Uploading training data to local bucket in region: $region..."
  echo ""
  aws s3 cp sample_train.csv s3://fedlearn-local-bucket-${AWS_ACCOUNT_ID}-${region}/data/training/
  echo ""
  echo "Uploading python training script to local bucket in region: $region..."
  echo ""
  aws s3 cp sourcedir.tar.gz s3://fedlearn-local-bucket-${AWS_ACCOUNT_ID}-${region}/code/
done



