import logging
from typing import Optional, List, Dict, Any, Callable

import mlconjug3
import nltk
import spacy
import torch
from nltk import WordNetLemmatizer
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, FillMaskPipeline
from allennlp.predictors.predictor import Predictor
import allennlp_models.classification

from core.settings import settings


logger = logging.Logger(__name__)


allennlp_sentiment_model = "https://storage.googleapis.com/allennlp-public-models/sst-roberta-large-2020.06.08.tar.gz"

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

spacy_tokenizer = spacy.load('en_core_web_md')
lemmatizer = WordNetLemmatizer()
default_conjugator = mlconjug3.Conjugator(language='en')

# # Download sentiment analysis model
# try:
#     logger.info(f"Downloading  sentiment analysis model from: {allennlp_sentiment_model}")
#     sentiment_analyser = Predictor.from_path(allennlp_sentiment_model)
# except Exception as es:
#     sentiment_analyser = None
#     print(es)

def load_mask_predictor(model_name='roberta-large'):
    logger.info(f"Downloading  roBERTa model from huggingface for Masked Text Prediction ")
    model = RobertaForMaskedLM.from_pretrained(model_name)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    device_number = torch.cuda.current_device() if torch.cuda.is_available() else -1
    predictor = FillMaskPipeline(model=model, tokenizer=tokenizer, device=device_number)

    def _postprocess_mask_prediction_token(text):
        return text[1:] if text[0] == "Ġ" else text

    def predict_mask(masked_text: str, options: Optional[List[str]] = None, num_results: int = 1) -> List[Dict[str, Any]]:

        results = predictor(masked_text, targets=options, top_k=num_results)

        parsed_results = []
        for result in results:
            parsed_result = {"word": _postprocess_mask_prediction_token(result['token_str']),
                             "softmax": result["score"]}
            parsed_results.append(parsed_result)
        return parsed_results

    return predict_mask


predict_mask: Callable = load_mask_predictor('roberta-large')

