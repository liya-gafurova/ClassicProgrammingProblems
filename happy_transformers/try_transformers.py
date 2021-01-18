from typing import Optional, List, Callable, Any, Dict
import torch
from transformers import FillMaskPipeline
from transformers import RobertaTokenizerFast, RobertaForMaskedLM

model = RobertaForMaskedLM.from_pretrained("roberta-large")
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
device_number = torch.cuda.current_device() if torch.cuda.is_available() else -1

mask_predictor = FillMaskPipeline(model=model, tokenizer=tokenizer, device=device_number)

#
# def predict_mask(text, predictor=mask_predictor, options: Optional[List[str]] = None, num_results: Optional[int] = 1):
#
#
#     results = predictor(text, targets=options, top_k=num_results)
#     parsed_results = []
#     for result in results:
#         parsed_result = {"word": _postprocess_mask_prediction_token(result['token_str']),
#                          "softmax": result["score"]}
#         parsed_results.append(parsed_result)
#
#     return parsed_results

def _postprocess_mask_prediction_token(text):
    return text[1:] if text[0] == "Ġ" else text

def _replace_token_in_sentence(words: List[str], id_of_token_to_be_replaced: int, token_for_replacement: str):
    last = [] if id_of_token_to_be_replaced == len(words) - 1 else words[id_of_token_to_be_replaced + 1:]
    return ' '.join(words[:id_of_token_to_be_replaced] + [token_for_replacement] + last)

# Через замыкание - то есть не надо при каждом вызоже создвать новый параметр, в которы передвать ссылку на \
# на модель для предсказания замаскированного токена

def predict_mask_with_roberta(predictor: FillMaskPipeline) -> Callable:

    def predict_mask(words: List[str], token_id_to_be_predicted: int,
                     options: Optional[List[str]] = None, num_results: int = 1) -> List[Dict[str, Any]]:
        # TODO compile str
        masked_str = _replace_token_in_sentence(words, token_id_to_be_predicted,'<mask>')

        results = predictor(masked_str, targets=options, top_k=num_results)

        parsed_results = []
        for result in results:
            parsed_result = {"word": _postprocess_mask_prediction_token(result['token_str']),
                             "softmax": result["score"]}
            parsed_results.append(parsed_result)
        return parsed_results

    return predict_mask