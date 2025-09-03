# Use the official AWS Lambda base image for Python 3.9
FROM public.ecr.aws/lambda/python:3.9

# Make sure we're using the correct architecture
ENV LAMBDA_TASK_ROOT=/var/task

# Copy requirements.txt
COPY requirements.txt .

# Install the python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy function code
COPY app.py .

# Set the handler as the CMD
CMD ["app.handler"]
