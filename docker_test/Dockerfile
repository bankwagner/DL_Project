FROM python:3.9.12

RUN apt-get update &&  \
	apt install -y 

ENV HOME /home/custom_user
RUN mkdir -p $HOME/project1
WORKDIR $HOME/project1
ENV CSV=${CSV:-$HOME/project1/data/data.csv}

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -rm -d $HOME -s /bin/bash -g root -G sudo -u 1000 custom_user
RUN echo 'custom_user:password' | chpasswd

SHELL ["/bin/bash", "-l", "-c"]

ENTRYPOINT python src/main.py -csv $CSV