# Use an official Python runtime as a base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /api

# Copy the current directory contents into the container at /resnet50-docker
COPY resnet50_model.onnx /api/resnet50_model.onnx
COPY requirements.txt /api/requirements.txt
COPY api.py /api/api.py

#COPY /home/arsenalducvy/Desktop/resnet50/ /home/arsenalducvy/Desktop/image_recognition_ai_model/script/experiment/resnet50-docker/
# Install the required Python packages
RUN pip3 install --no-cache-dir uvicorn onnxruntime fastapi google-cloud-logging

# Expose port 8000
EXPOSE 8000

# Command to run the app with Uvicorn
CMD ["uvicorn", "api:api", "--host", "0.0.0.0", "--port", "8000"]
#CMD ["python", "api.py"]