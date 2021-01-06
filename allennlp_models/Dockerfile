FROM python:3.8

RUN python -m pip install -U pip
RUN pip install allennlp==1.0.0 allennlp-models==1.0.0
RUN python -c "import nltk; nltk.download('punkt')"
RUN python -c "import nltk; nltk.download('wordnet')"
RUN python -m spacy download en_core_web_sm
WORKDIR /app

COPY main.py .

CMD [ "python", "./main.py" ]
