FROM nvidia/cuda:11.3.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive


# install python, pip
RUN apt-get update &&\
    apt-get install python3.10 -y &&\
    apt-get install python3-pip -y

# making directory of app
WORKDIR /WebSemble

# copy contents elementwise to reduce layer sizes
COPY ./downstream ./downstream
COPY ./instructions_docker ./instructions_docker
COPY ./models/bart-base-webis22 ./models/bart-base-webis22
COPY ./models/bert-base-uncased-MNLI-webis22 ./models/bert-base-uncased-MNLI-webis22
COPY ./models/bert-large-uncased-whole-word-masking-finetuned-squad ./models/bert-large-uncased-whole-word-masking-finetuned-squad
COPY ./models/deberta-v3-base-tasksource-nli ./models/deberta-v3-base-tasksource-nli
COPY ./models/distilbert-base-cased-distilled-squad ./models/distilbert-base-cased-distilled-squad
COPY ./models/distilbert-base-uncased ./models/distilbert-base-uncased
COPY ./models/distilbert-base-uncased-webis22 ./models/distilbert-base-uncased-webis22
COPY ./models/pegasus-xsum ./models/pegasus-xsum
COPY ./models/roberta-base-squad2 ./models/roberta-base-squad2
COPY ./utils ./utils
COPY ./run.py ./run.py
COPY ./web_trainer.py ./web_trainer.py
COPY ./requirements.txt ./requirements.txt
COPY ./README.md ./README.md

# install packages
RUN pip install -r requirements.txt

# make script executable
RUN chmod +x /WebSemble/run.py
ENTRYPOINT ["python3", "/WebSemble/run.py", "$inputDataset", "$outputDir"]