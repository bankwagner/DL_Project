# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory to /model_2
WORKDIR /model_2

# Copy the current directory contents into the container at /model_2
COPY . /model_2

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME model_2

# Run model_2.py when the container launches
CMD ["python", "model_2.py"]
