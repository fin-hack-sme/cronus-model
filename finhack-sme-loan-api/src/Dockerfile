# Step 1: Use official lightweight Python image as base OS.
FROM python:3.11-slim
# Allow statements and log
ENV PYTHONUNBUFFERED True

# Step 2. Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Step 3. Install production dependencies.
RUN pip install -r requirements.txt

# Run
CMD ["python", "main.py"]
