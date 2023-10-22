# Use an official Python runtime as a parent image
FROM python:3.9.12

# Set the working directory to /prep
WORKDIR /prep

# Copy the current directory contents into the container at /prep
COPY . /prep

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME prep

# Run prep.py when the container launches
CMD ["python", "prep.py"]
