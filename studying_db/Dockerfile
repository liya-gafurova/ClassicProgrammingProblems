FROM python:3.8

RUN python -m pip install -U pip
RUN pip install peewee==3.14.0

WORKDIR /app

COPY . .


CMD [ "python", "./main.py" ]
