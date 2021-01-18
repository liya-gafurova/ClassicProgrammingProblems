"""
ALlenNLP использует библиотеку transformers

новый AllenNLP (1.3.0)
 torch [1.6.0 - 1.8.0)
 transformers [4.0 - 4.1)

happytransformer
 torch >= 1.0
 transformers >=4.0


"""


from typing import Callable, List, Dict, Any

import torch
from allennlp.predictors.predictor import Predictor
from happytransformer.happy_word_prediction import HappyWordPrediction
from transformers import RobertaTokenizerFast, RobertaForMaskedLM
import allennlp_models.classification

from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
from try_transformers import mask_predictor, predict_mask_with_roberta
snt_masked = "<mask> very well-made, funny and entertaining picture."

# happy_wp_roberta = HappyWordPrediction("ROBERTA", "roberta-large")
# predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/basic_stanford_sentiment_treebank-2020.06.09.tar.gz")
snt = "a very well-made, funny and entertaining picture."



# SENTIMENT
sentiment = predictor.predict(
  sentence=snt
)
print(sentiment)


# FIX happytransformer
predicted = happy_wp_roberta.predict_mask(snt_masked, top_k=3)
print(predicted)

# # TODO как  реализовать то же самое, но без happytransformer

predict_token: Callable = predict_mask_with_roberta(mask_predictor)
predicted_2: List[Dict[str,Any]] = predict_token(snt_masked.split(),0,options=['A', 'a', 'the'],  num_results=3,)
print(predicted_2)