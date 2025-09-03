# Create project directory
mkdir p2p-federated-nids
cd p2p-federated-nids

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (should show python path in venv)
where python

# Create the project structure in cmd
mkdir src\node src\network src\models src\utils src\security
mkdir tests
mkdir configs
mkdir data

# Create __init__.py files
type nul > src\__init__.py
type nul > src\node\__init__.py
type nul > src\network\__init__.py
type nul > src\models\__init__.py
type nul > src\utils\__init__.py
type nul > src\security\__init__.py

# Upgrade pip first
 python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# For development dependencies only
pip install -r requirements-dev.txt  # We'll create this next

# install the relevant dependencies

# create an .env file


# Train on a single CSV file (e.g., Tuesday's data)
python -m src.train_local \
    --node-id node_001 \
    --data-path /path/to/Tuesday-WorkingHours.pcap_ISCX.csv \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --save-preprocessor

# Train with multiple files using the data loader
python -m src.utils.data_loader \
    --data-dir /path/to/MachineLearningCVE/ \
    --num-nodes 3 \
    --split-strategy non_iid \
    --balance

# Train with a sample for testing
python -m src.train_local \
    --node-id node_001 \
    --data-path /path/to/your/cicids2017.csv \
    --epochs 5 \
    --sample-size 50000 \
    --save-preprocessor

# download aws cli

# configure aws cli
aws configure


# Create Repo in AWS, Build and Deploy Images to Repo across all regions
./build-all.sh

# Deploy Stacks
./deploy.sh

# Go to AWS and upload training data to the s3 bucket in the path {bucket-name}/data/training/

# Go to AWS and upload model python code (the zipped file - tar.gz) to the s3 bucket in the path {bucket-name}/code/


# Deploy to a single region (Testing)
aws cloudformation create-stack --stack-name federated-learning-test --template-body file://main-stack.yaml --parameters ParameterKey=EnvironmentName,ParameterValue=FedLearnTest ParameterKey=ModelBucketName,ParameterValue=fedlearn-models-test-unique-name --capabilities CAPABILITY_NAMED_IAM



# Creates a CloudFormation StackSet Administration Role and the Execution Role in AWS. This is a standard setup used when you want to manage CloudFormation StackSets across multiple AWS accounts and/or Regions
aws cloudformation create-stack --stack-name AWSCloudFormationStackSetAdministrationRole --template-url https://s3.amazonaws.com/cloudformation-stackset-sample-templates-us-east-1/AWSCloudFormationStackSetAdministrationRole.yml --capabilities CAPABILITY_NAMED_IAM


aws cloudformation create-stack --region us-west-2 --stack-name AWSCloudFormationStackSetExecutionRole --template-url https://s3.amazonaws.com/cloudformation-stackset-sample-templates-us-east-1/AWSCloudFormationStackSetExecutionRole.yml --parameters ParameterKey=AdministratorAccountId,ParameterValue=838814607172 --capabilities CAPABILITY_NAMED_IAM

aws cloudformation create-stack --region eu-west-1 --stack-name AWSCloudFormationStackSetExecutionRole --template-url https://s3.amazonaws.com/cloudformation-stackset-sample-templates-us-east-1/AWSCloudFormationStackSetExecutionRole.yml --parameters ParameterKey=AdministratorAccountId,ParameterValue=838814607172 --capabilities CAPABILITY_NAMED_IAM

aws cloudformation create-stack --region us-west-1 --stack-name AWSCloudFormationStackSetExecutionRole --template-url https://s3.amazonaws.com/cloudformation-stackset-sample-templates-us-east-1/AWSCloudFormationStackSetExecutionRole.yml --parameters ParameterKey=AdministratorAccountId,ParameterValue=838814607172 --capabilities CAPABILITY_NAMED_IAM

aaws cloudformation create-stack-instances --stack-set-name federated-learning-stackset --accounts 838814607172 --regions us-east-1 us-west-2 eu-west-1 --operation-preferences RegionConcurrencyType=PARALLEL,MaxConcurrentPercentage=100
# Create the Stack-set
aws cloudformation create-stack-set  --stack-set-name federated-learning-stackset --template-body file://fl-stackset.yaml --parameters ParameterKey=EnvironmentName,ParameterValue=FederatedLearning ParameterKey=ModelBucketName,ParameterValue=fedlearn-central-models --capabilities CAPABILITY_NAMED_IAM

# Deploy to multiple regions

# deploy the stack

<!-- aws cloudformation create-stack-set  --stack-set-name federated-learning-stackset --template-body file://fl-stackset.yaml --parameters ParameterKey=EnvironmentName,ParameterValue=federatedlearning ParameterKey=ModelBucketName,ParameterValue=fl-central-models ParameterKey=UseExistingSageMakerRole,ParameterValue=true --capabilities CAPABILITY_NAMED_IAM -->

<!-- aws cloudformation create-stack --stack-name federated-learning-test --template-body file://main-stack.yaml --parameters ParameterKey=EnvironmentName,ParameterValue=fedlearntest ParameterKey=ModelBucketName,ParameterValue=fl-models-test --capabilities CAPABILITY_NAMED_IAM -->

aws cloudformation create-stack-set  --stack-set-name federated-learning-stackset --template-body file://fl-stackset.yaml --parameters ParameterKey=EnvironmentName,ParameterValue=federatedlearning ParameterKey=UseExistingSageMakerRole,ParameterValue=true --capabilities CAPABILITY_NAMED_IAM

aws cloudformation create-stack --stack-name federated-learning-test --template-body file://resource-stack.yaml --capabilities CAPABILITY_NAMED_IAM --region eu-west-1

aws cloudformation create-stack-instances --stack-set-name federated-learning-stackset --accounts $(aws sts get-caller-identity --query Account --output text) --regions us-west-1 us-west-2 eu-west-1 --operation-preferences RegionConcurrencyType=PARALLEL,MaxConcurrentPercentage=100

aws cloudformation create-stack --stack-name federated-central-resources --template-body file://central-stack.yaml --parameters ParameterKey=EnvironmentName,ParameterValue=fedlearn --capabilities CAPABILITY_NAMED_IAM 

aws cloudformation create-stack \
  --stack-name fedlearn-central-resources \
  --template-body file://central-resource-stack.yaml \
  --parameters \
    ParameterKey=EnvironmentName,ParameterValue=fedlearn \
    ParameterKey=ModelBucketName,ParameterValue=fedlearn-models-central-$(date +%s) \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-east-1

# Or update the stack set

aws cloudformation update-stack-set  --stack-set-name federated-learning-stackset --template-body file://fl-stackset.yaml --parameters ParameterKey=EnvironmentName,ParameterValue=federatedlearning ParameterKey=UseExistingSageMakerRole,ParameterValue=false --capabilities CAPABILITY_NAMED_IAM

aws cloudformation update-stack --stack-name federated-central-resources --template-body file://central-stack.yaml --parameters ParameterKey=EnvironmentName,ParameterValue=fedlearn --capabilities CAPABILITY_NAMED_IAM

# Get account ID
aws sts get-caller-identity --query Account --output text

# Get bucket name  
aws cloudformation describe-stacks  --stack-name federated-central-resources --query 'Stacks[0].Outputs[?OutputKey==`CentralModelBucketName`].OutputValue' --output text

aws iam get-role --role-name SageMakerExecutionRole-eu-west-1 --region eu-west-1    

# add this policy in the management console after obtaining the regions
<!-- ## 1.Go to S3 Console, find central bucket
## 2.Select the central bucket, go to permissions tab
## 3.Scroll down to "Bucket policy" section, click edit
## 4.Paste the policy JSON in the policy editor, click save changes -->

{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowFederatedLearningRoles",
            "Effect": "Allow",
            "Principal": {
                "AWS": [
                    "arn:aws:iam::838814607172:role/federatedlearning-eu-west-1-LambdaExecutionRole-eu-west-1",
                    "arn:aws:iam::838814607172:role/SageMakerExecutionRole-eu-west-1",
                    "arn:aws:iam::838814607172:role/SageMakerExecutionRole-us-east-1",
                    "arn:aws:iam::838814607172:role/federatedlearning-us-east-1-LambdaExecutionRole-us-east-1",
                    "arn:aws:iam::838814607172:role/SageMakerExecutionRole-us-west-2",
                    "arn:aws:iam::838814607172:role/federatedlearning-us-west-2-LambdaExecutionRole-us-west-2"
                ]
            },
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:GetObjectVersion",
                "s3:ListBucket",
                "s3:GetBucketLocation",
                "s3:GetBucketVersioning"
            ],
            "Resource": [
                "arn:aws:s3:::fl-modelbucket",
                "arn:aws:s3:::fl-modelbucket/*"
            ]
        }
    ]
}


# add s3 cross-region replication later to all the buckets
<!-- Go to S3 Console → Select your source bucket
Management tab → Replication rules → Create replication rule
Configure:

Rule name: ReplicateModels
Status: Enabled
Priority: 1
Scope: Choose a rule scope → Limit scope with prefix → models/
Destination: Choose bucket in another region → fl-modelbucket
IAM role: Create new role (AWS will create it automatically)
Storage class: Standard
Additional options: Enable Delete marker replication -->

# add layers for numpy and pytorch support for the aggregator


# delete the stack instances
aws cloudformation delete-stack-instances --stack-set-name federated-learning-stackset --accounts $(aws sts get-caller-identity --query Account --output text) --regions us-east-1 us-west-2 eu-west-1 --no-retain-stacks --operation-preferences RegionConcurrencyType=PARALLEL,MaxConcurrentPercentage=100

# Delete test stack
aws cloudformation delete-stack --stack-name federated-learning-test

# Delete stack set
aws cloudformation delete-stack-set --stack-set-name federated-learning-stackset



aws cloudformation describe-stacks --stack-name AWSCloudFormationStackSetAdministrationRole

# Empty and delete S3 buckets
aws s3 rm s3://fedlearn-models-test-unique-name --recursive
aws s3 rb s3://fedlearn-models-test-unique-name

filesystem id: fs-0d7844793c2fea23c

 172.31.0.0/16, vpc-0ba1820b887443afa

 subnet-0a605b707753f0e2b

mount id: fsmt-0c6fdb2cd3b0ee5ca

accesspoint id: fsap-07d12d9d3e4764cd9

accesspoint arn: arn:aws:elasticfilesystem:eu-west-1:838814607172:access-point/fsap-07d12d9d3e4764cd9

"security GroupId": sg-0cae7efcb4ae45845

# Create access point (replace fs-xxxxxxxxx with your EFS ID)
aws efs create-access-point --file-system-id fs-02840d8fe5c92e3de --posix-user Uid=1001,Gid=1001 --root-directory Path=/,CreationInfo='{OwnerUid=1001,OwnerGid=1001,Permissions=755}' --region eu-west-1

aws lambda update-function-configuration --function-name federatedlearning-eu-west-1-AggregatorLambda --file-system-configs Arn=arn:aws:elasticfilesystem:eu-west-1:838814607172:access-point/fsap-089556f880ae63c58,LocalMountPath=/mnt/efs --timeout 900 --memory-size 2048 --region eu-west-1


sg-05465c73c699c45a7

aws efs describe-mount-targets --file-system-id fs-0d7844793c2fea23c --region eu-west-1

fsmt-0c6fdb2cd3b0ee5ca


aws ecr create-repository --repository-name lambda-pytorch-example --image-scanning-configuration scanOnPush=true --region eu-west-1


aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 838814607172.dkr.ecr.eu-west-1.amazonaws.com
