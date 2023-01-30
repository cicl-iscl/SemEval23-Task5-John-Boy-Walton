FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive


# install python, pip
RUN apt-get update &&\
    apt-get install python3.10 -y &&\
    apt-get install python3-pip -y

# making directory of app
WORKDIR /WebSemble 
COPY . .

# install packages
RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "/WebSemble/run.py" ]