# # Use an official Python runtime as a parent image
# FROM python:3.10

# # Set the working directory to /model_1
# WORKDIR /milestone_3

# # Copy the current directory contents into the container at /model_1
# COPY . /milestone_3

# # Install any needed packages specified in requirements.txt
# RUN pip install -r requirements.txt

# # Make port 80 available to the world outside this container
# EXPOSE 80

# # Define environment variable
# ENV NAME final

# # Run model_1.py when the container launches
# #CMD ["python", "model_1.py"]

FROM python:3.9

ARG GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

EXPOSE ${GRADIO_SERVER_PORT}

WORKDIR /workspace

COPY . /workspace

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#CMD ["/bin/bash", "-c", "bash prep_setup.sh", "bash unet_setup.sh" , "bash ensemble_setup.sh", "python", "final.py"]
CMD ["python", "final.py"]