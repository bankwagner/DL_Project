# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory to /model_1
WORKDIR /model_1

# Copy the current directory contents into the container at /model_1
COPY . /model_1

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME model_1

# Run model_1.py when the container launches
CMD ["python", "model_1.py"]
