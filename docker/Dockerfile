FROM python:3.6-slim

RUN apt-get update
RUN apt-get install -y vim

RUN mkdir -p genetic-snake-ai \
&& mkdir -p genetic-snake-ai/genetic_snake \
&& mkdir -p genetic-snake-ai/experiments \
&& mkdir -p genetic-snake-ai/external

COPY genetic_snake genetic-snake-ai/genetic_snake/
COPY experiments genetic-snake-ai/experiments/
COPY setup.py genetic-snake-ai/
COPY external genetic-snake-ai/external/



RUN pip3 install --upgrade pip \
&& pip3 install genetic-snake-ai/external/genetic-algorithm/src \
&& pip3 install genetic-snake-ai/ \
&& pip3 install gin-config numpy pygame h5py pyzmq \
&& pip3 install genetic-snake-ai/external/evostra/
