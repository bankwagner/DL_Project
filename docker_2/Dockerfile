FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN apt-get update &&  \
	apt install -y  \
	openssh-server \
	git \
	libgl1 \
	vim \
	tmux \
	byobu

ENV HOME /home/custom_user
RUN mkdir -p $HOME/project1
WORKDIR $HOME/project1
ENV CSV=${CSV:-$HOME/project1/data/data.csv}

COPY . .
COPY ssh/ssh_config /etc/ssh/sshd_config

EXPOSE 22
EXPOSE 8888

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -rm -d $HOME -s /bin/bash -g root -G sudo -u 1000 custom_user
RUN echo 'custom_user:password' | chpasswd

SHELL ["/bin/bash", "-l", "-c"]

# Start Jupyter lab with custom password
ENTRYPOINT service ssh start && jupyter-lab \
	--ip 0.0.0.0 \
	--port 8888 \
	--no-browser \
	--NotebookApp.notebook_dir='$home' \
	--ServerApp.terminado_settings="shell_command=['/bin/bash']" \
	--allow-root