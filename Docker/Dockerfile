# ✅ Use AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.10

# Set working directory
WORKDIR /var/task

# Install dependencies
COPY Docker/dockerrequirements.txt .
RUN pip install --upgrade pip && \
    pip install -r dockerrequirements.txt

# Copy your application code
COPY main.py .
COPY apis/ ./apis/
COPY models/ ./models/

# Set the Lambda handler (main.lambda_handler means function named `lambda_handler` in main.py)
CMD ["main.lambda_handler"]
