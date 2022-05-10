FROM ubuntu:18.04
RUN apt -y update
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update

RUN apt-get install -y python3.8

RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.8 /usr/bin/python &&  ln -s /usr/bin/python3.8 /usr/bin/python3 \
    && apt-get install -y python3-pip python-dev python3.8-dev && python3 -m pip install pip --upgrade

RUN pip3 --version
WORKDIR /code
RUN apt-get install -y libpq-dev
RUN apt install -y gcc
RUN apt-get install -y python3-dev
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD ["/usr/bin/python3", "api.py"]

