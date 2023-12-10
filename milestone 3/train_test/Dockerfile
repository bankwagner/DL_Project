# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory to /test_train
WORKDIR /test_train

# Copy the current directory contents into the container at /test_train
COPY . /test_train

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN ./prep_setup.sh
RUN ./unet.sh
RUN ./linknet.sh
RUN ./fpn.sh
RUN ./ensemble.sh

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME test_train

# Run test.py when the container launches
CMD ["python", "test.py"]
