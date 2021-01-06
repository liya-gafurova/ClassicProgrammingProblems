import logging
import os
import time
from datetime import datetime

from allennlp.models import Archive, load_archive
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
import allennlp_models.classification

# logging.basicConfig( level=logging.INFO)

# We will get models from URL, because it will load from cache
# in case, we dont have any cache, models will be downloaded from web. And cache will be filled
# Cache is stored on host machine (see run.txt -> docker run -> volumes)

dependency_model_url = "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz"
sentiment_analysis_model_url = "https://storage.googleapis.com/allennlp-public-models/sst-roberta-large-2020.06.08.tar.gz"
model_urls = [dependency_model_url, sentiment_analysis_model_url]


# pretrained_models_directory = "/root/.cache/torch/transformers/pretrained/"
# pretrained_models_directory_local = "/home/lia/.cache/torch/transformers/pretrained/"
#
# dependency_model_archive = "biaffine-dependency-parser-ptb-2020.04.06.tar.gz"
# sentiment_analysis_model_archive = "sst-roberta-large-2020.06.08.tar.gz"
# model_names = [dependency_model_archive, sentiment_analysis_model_archive]

start = datetime.now()

# archive:Archive = load_archive('/root/.cache/torch/transformers/pretrained/biaffine-dependency-parser-ptb-2020.04.06.tar.gz')
# predictor: Predictor = Predictor.from_archive(archive)
predictor: Predictor = Predictor.from_path(dependency_model_url)
prediction = predictor.predict(
  sentence="If I bring 10 dollars tomorrow, can you buy me lunch?"
)
print(prediction)

print(f"DEPENDENCY MODEL from cache: -->  {datetime.now() - start}")


start = datetime.now()
# archive_sentiment_analysis: Archive = load_archive("/root/.cache/torch/transformers/pretrained/sst-roberta-large-2020.06.08.tar.gz")
# sentiment_analysis_predictor = Predictor.from_archive(archive_sentiment_analysis)
sentiment_analysis_predictor: Predictor = Predictor.from_path(sentiment_analysis_model_url)
prediction = sentiment_analysis_predictor.predict(
  sentence="a very well-made, funny and entertaining and wonderful picture."
)
print(prediction)
print(f"SENTIMENT MODEL from cache: -->  {datetime.now() - start}")


