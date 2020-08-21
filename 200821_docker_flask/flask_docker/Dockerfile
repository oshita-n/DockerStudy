FROM python:3.7.5

RUN pip install -U pip
RUN pip install tensorflow
RUN pip install flask
RUN pip install Pillow

WORKDIR /flask_docker/src/
ADD . /flask_docker/src/
RUN mkdir /flask_docker/src/uploads
