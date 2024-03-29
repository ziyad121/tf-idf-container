FROM python:3.7

COPY ./script /script
COPY ./requirements.txt requirements.txt

RUN apt-get update && apt-get install -y python3
RUN pip install -r requirements.txt

CMD [ "python", "./script/scoring.py" ]
