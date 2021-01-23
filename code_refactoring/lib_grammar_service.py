import json
import logging
import os
import sys
from typing import List, Dict, Any, Tuple, Optional

import nltk
import requests
import re
from happytransformer.happy_transformer import HappyTransformer

from transformers import RobertaForMaskedLM, RobertaTokenizerFast
import mlconjug3
import torch
from happytransformer.mlm_utils import FinetuneMlm, word_prediction_args
from happytransformer.sequence_classifier import SequenceClassifier
from nltk import sent_tokenize, pos_tag, word_tokenize
from nltk.metrics.distance import edit_distance
from nltk.stem import WordNetLemmatizer

from collections import namedtuple
import string
import re
import os
import sys
import csv
import logging
import logging.config
import numpy as np
import torch
import pandas as pd

from happytransformer.classifier_args import classifier_args
from happytransformer.sequence_classifier import SequenceClassifier
from happytransformer.mlm_utils import FinetuneMlm, word_prediction_args
from core.settings import settings
from models.grammar_models import SingleResult, IncomeRequestModel, Native_LT_API_Model, SingleResultFromNativeApi, \
    Categories, AdditionalChecking
from core.lib import DISABLED_CATEGORIES, DISABLED_RULES
from . import dependency_parser, happy_roberta
from happytransformer import HappyROBERTA, classifier_args
from happytransformer.happy_transformer import HappyTransformer
nltk.download('averaged_perceptron_tagger')

wnl = WordNetLemmatizer()
default_conjugator = mlconjug3.Conjugator(language='en')

list_possible_auxiliary_uncased = "'s,`s,’s,'am,`am,’am,'m,`m,’m,'ve,`ve,’ve,'ll,`ll,’ll,'d,`d,’d,'re,`re,’re,get,got,gotten,getting,be,am,are,is,was,were,being,can,could,do,did,does,doing,have,had,has,having,may,might,must,shall,should,will,would".split(
    ",")
AUX_CONFIG = json.loads(
    """{"'s": ["is", "has", "was"], "`s": ["is", "has", "was"], "’s": ["is", "has", "was"], "'am": ["am"], "`am": ["am"], "’am": ["am"], "'m": ["am"], "`m": ["am"], "’m": ["am"], "'ve": ["have"], "`ve": ["have"], "’ve": ["have"], "'ll": ["will"], "`ll": ["will"], "’ll": ["will"], "'d": ["had", "would", "should"], "`d": ["had", "would", "should"], "’d": ["had", "would", "should"], "'re": ["were", "are"], "`re": ["were", "are"], "’re": ["were", "are"], "get": ["get"], "got": ["got"], "gotten": ["gotten"], "getting": ["getting"], "be": ["be"], "am": ["am"], "are": ["are"], "is": ["is"], "was": ["was"], "were": ["were"], "being": ["being"], "can": ["can"], "could": ["could"], "do": ["do"], "did": ["did"], "does": ["does"], "doing": ["doing"], "have": ["have"], "had": ["had"], "has": ["has"], "having": ["having"], "may": ["may"], "might": ["might"], "must": ["must"], "shall": ["shall"], "should": ["should"], "will": ["will"], "would": ["would"]}""")
AUX_FULL_FORMS = json.loads(
    """{"all_aux_forms": ["does", "might", "is", "getting", "doing", "are", "be", "has", "were", "having", "have", "did", "am", "must", "gotten", "do", "would", "may", "can", "got", "could", "get", "will", "was", "being", "should", "had", "shall"]}""")[
    'all_aux_forms']

contracted_groups = [
    ("is", "’s", "'s", "`s"),
    ("has", "’s", "'s", "`s"),
    ("was", "’s", "'s", "`s"),
    ("were", "’re", "'re", "`re"),
    ("are", "’re", "'re", "`re"),
    ("have", "’ve", "'ve", "`ve"),
    ("am", "’m", "'m", "`m", "’am", "'am", "`am"),
    ("will", "’ll", "'ll", "`ll"),
    ("had", "’d", "'d", "`d"),
    ("would", "’d", "'d", "`d"),
    ("should", "’d", "'d", "`d"),
]

MASK = '[MASK]'
POSSIBLES_AUXILIARY = list_possible_auxiliary_uncased
LOW_THRESHOLD = 0.3
HIGH_THRESHOLD = 0.6
VERB_HIGH_THRESHOLD = 0.5
HIGH_USAGE_THRESHOLD = 0.7
ONE_LETTER_WORDS = ['I', 'a', "A", 'i']
VERB_TAGS = ("VB", 'VBP',"VBG", "VBZ", "VBN", 'VBD')
POS_VERB_SIMILAR_GROUPS = {"infinitive": ("VB", 'VBP', "NN"),
                           "gerund": ("VBG", 'NN'),
                           "3sg": ("VBZ"),
                           "ed_form": ("VBN", 'VBD', "JJ")}
RULES_TO_DISABLE = ['WHITESPACE_RULE']
REDUNDANT_CATEGORIES = (
    "PLAIN_ENGLISH", "CREATIVE_WRITING", "TEXT_ANALYSIS", "STYLE", "NONSTANDARD_PHRASES", "REDUNDANCY")



def _is_in_same_contracted_group(word, predicted_word):
    if word == predicted_word:
        return True
    for group in contracted_groups:
        if word in group and predicted_word in group:
            return True
    return False


def _check_auxiliary_verbs(words: List[str], sentence: str, sent_decontracted: str,
                           tagged_words: Tuple[str, str, str]) -> List[SingleResult]:
    possible_errors = []
    fixed_sentence: str = None
    for i, (word, pos, dependency) in enumerate(tagged_words):
        if pos == 'AUX':
            word= word.lower()
            err, to_be_corrected = False, False
            # aux_full_form = AUX_CONFIG[word]
            compiled_str = _replace_token_in_sentence(words, i, [MASK])
            results = {i: word_info for i, word_info in enumerate(
                happy_roberta.predict_mask(compiled_str, num_results=len(AUX_FULL_FORMS), options=AUX_FULL_FORMS))}
            first, second, third = results[0], results[1], results[3]
            print(f"NEW: {word} -- {pos} -- {[first, second, third]}")
            if first['softmax'] > 0.79 and not  _is_in_same_contracted_group(word, first['word']):
                err, to_be_corrected = True, True
            elif first['softmax'] < 0.01 and  not _is_in_same_contracted_group(word, first['word']) and \
                not _is_in_same_contracted_group(word, second['word']) and \
                not _is_in_same_contracted_group(word, third['word']):
                err=True
            elif not _is_in_same_contracted_group(word, first['word']) and \
                not _is_in_same_contracted_group(word, second['word']):
                err = True

            replacements = [first['word'], second['word']]
            if err and to_be_corrected:
                fixed_sentence = _replace_token_in_sentence(words, i, [replacements[0]])

                err_sr = SingleResult(
                    category="GRAMMAR",
                    context=_get_window(number_of_neighbors=2, words=words, current_word_id=i),
                    errorLength=len(word),
                    matchedText=word,
                    message="Perhaps the wrong form of the auxiliary verb",
                    offset=sent_decontracted.find(word),
                    offsetInContext=None,
                    replacements=replacements,
                    ruleId='INCORRECT_AUXILIARY_VERB',
                    ruleIssueType='Mistake',
                    sentence=sentence
                )
                err_sr.offsetInContext = err_sr.context.find(word)
                possible_errors.append(err_sr)

            elif err:
                err_sr = SingleResult(
                    category="GRAMMAR",
                    context=_get_window(number_of_neighbors=2, words=words, current_word_id=i),
                    errorLength=len(word),
                    matchedText=word,
                    message="Perhaps the wrong form of the auxiliary verb",
                    offset=sent_decontracted.find(word),
                    offsetInContext=None,
                    replacements=replacements,
                    ruleId='INCORRECT_AUXILIARY_VERB',
                    ruleIssueType='Hint',
                    sentence=sentence
                )
                err_sr.offsetInContext = err_sr.context.find(word)
                possible_errors.append(err_sr)

    return possible_errors, fixed_sentence


def _first_level_verb_check(word, results):
    predicted_words = [r['word'] for r in results.values()]
    for pr in predicted_words:
        if word == pr:
            return True

        for gr in contracted_groups:
            if word in gr and pr in gr:
                return True
    return False


def _is_in_different_pos_group(real_verb_pos, first_result_pos):
    for group_name, group in POS_VERB_SIMILAR_GROUPS.items():
        if real_verb_pos in group and first_result_pos in group:
            return False
    return True


def _at_least_one_verb_in_results(results_tags):
    for tag in results_tags:
        if tag.find("VB") > -1:
            return True
    return False


def _has_verb_mistake(word, results) -> Tuple[bool, Optional[bool]]:
    real_verb_nltk_post_tagged, results_nltk_pos_tagged = pos_tag([word])[0], pos_tag(
        [r['word'] for r in results.values()])
    print(real_verb_nltk_post_tagged)
    print(results_nltk_pos_tagged)
    results_tags = [res_tagged[1] for res_tagged in results_nltk_pos_tagged]
    most_common_tag = _get_most_common_tag(results_nltk_pos_tagged)

    if not _at_least_one_verb_in_results(results_tags):
        print("INCORRECT WORD USAGE")
        return False, True
    elif results[0]['softmax'] > VERB_HIGH_THRESHOLD \
            and _is_in_different_pos_group(real_verb_nltk_post_tagged[1], results_nltk_pos_tagged[0][1]):
        return True, False
    elif results[0]['softmax'] > HIGH_THRESHOLD \
            and not _is_in_different_pos_group(real_verb_nltk_post_tagged[1], most_common_tag):
        return False, True
    elif _is_in_different_pos_group(real_verb_nltk_post_tagged[1], most_common_tag):
        return True, False

    return False, False


def _get_most_common_tag(results):
    d = dict.fromkeys([tag for _, tag in results], 0)
    for word, tag in results:
        d[tag] += 1
    max_key = max(d, key=d.get)
    return max_key


async def _check_verbs(words, sentence, sent_decontracted, tagged_words):
    # check correct verb form
    possible_errors, usage_hints = [], []
    for i, (word, pos, dependency) in enumerate(tagged_words):
        if pos == 'VERB':
            word = word.lower()
            compiled_str = _replace_token_in_sentence(words, i, [MASK])

            results = {i: word_info for i, word_info in
                       enumerate(happy_roberta.predict_mask(compiled_str, num_results=10))}

            predicted_words = [r['word'] for r in results.values()]
            real_verb_nltk_post_tagged, results_nltk_pos_tagged = pos_tag([word])[0], pos_tag(
                [r['word'] for r in results.values()])
            most_common_tag = _get_most_common_tag(results_nltk_pos_tagged)
            results_tags = [res_tagged[1] for res_tagged in results_nltk_pos_tagged]
            if not _first_level_verb_check(word, results):

                if not _at_least_one_verb_in_results(results_tags) or \
                        results[0]['softmax'] > HIGH_THRESHOLD \
                        and not _is_in_different_pos_group(real_verb_nltk_post_tagged[1], most_common_tag):
                    replacements = [result['word'] for result in results.values()]
                    err = SingleResult(
                        category="HINTS",
                        context=_get_window(number_of_neighbors=4, words=words, current_word_id=i),
                        errorLength=len(word),
                        matchedText=word,
                        message="Perhaps incorrect verb usage.",
                        offset=sent_decontracted.find(word),
                        offsetInContext=None,
                        replacements=replacements[:3],
                        ruleId='INCORRECT_WORD_USAGE',
                        ruleIssueType='Hint',
                        sentence=sentence
                    )
                    err.offsetInContext = err.context.find(word)
                    usage_hints.append(err.dict())
                    continue


                conjugation = default_conjugator.conjugate(wnl.lemmatize(word), subject="abbrev")

                verb_forms = []
                config = conjugation.conjug_info
                groups = ['indicative', 'infinitive', 'imperative']
                for g in groups:
                    for verb_group, value in config[g].items():
                        try:
                            verb_forms.extend(value.values())
                        except:
                            verb_forms.append(value)
                verb_forms = set(verb_forms)

                str_for_typos_checking = " ".join(verb_forms)

                errs = await send_to_language_tool(str_for_typos_checking,
                                                   f'http://{settings.LT_HOST}:{settings.LT_PORT}/v2{settings.LAGUAGETOOL_API_CHECK}')

                mathces = []
                for err_match in errs['matches']:
                    sr, _ = create_mistake_description(err_match)
                    mathces.append(sr)

                corrected_text = correct(str_for_typos_checking, mathces)
                verb_forms  = set(corrected_text.split())
                common = verb_forms.intersection(set(predicted_words))

                verb_form_predictions  = {i: word_info for i, word_info in
                           enumerate(happy_roberta.predict_mask(compiled_str, options=list(verb_forms), num_results=len(verb_forms)))}


                if common!= set() \
                        or (most_common_tag in VERB_TAGS and _is_in_different_pos_group(real_verb_nltk_post_tagged[1], most_common_tag))\
                        or word not in [verb_form_predictions[0]['word'], verb_form_predictions[1]['word']]:
                    sr = SingleResult(
                        category="GRAMMAR",
                        context=_get_window(number_of_neighbors=4, words=words, current_word_id=i),
                        errorLength=len(word),
                        matchedText=word,
                        message="Perhaps incorrect form of the word.",
                        offset=sent_decontracted.find(word),
                        offsetInContext=None,
                        replacements=list(common) if common!=set() else predicted_words[:3],
                        ruleId='INCORRECT_VERB',
                        ruleIssueType='Mistake',
                        sentence=sentence
                    )
                    possible_errors.append(sr)




                # r, word_usage_mistake_caught = _has_verb_mistake(words[i], results)
                # if word_usage_mistake_caught:
                #     replacements = [result['word'] for result in results.values()]
                #     err = SingleResult(
                #         category="HINTS",
                #         context=_get_window(number_of_neighbors=4, words=words, current_word_id=i),
                #         errorLength=len(word),
                #         matchedText=word,
                #         message="Perhaps incorrect verb usage.",
                #         offset=sent_decontracted.find(word),
                #         offsetInContext=None,
                #         replacements=replacements,
                #         ruleId='INCORRECT_WORD_USAGE',
                #         ruleIssueType='Hint',
                #         sentence=sentence
                #     )
                #     err.offsetInContext = err.context.find(word)
                #     usage_hints.append(err.dict())
                #
                # if r and not word_usage_mistake_caught:
                #     replacements = [result['word'] for result in results.values()]
                #     err = SingleResult(
                #         category="GRAMMAR",
                #         context=_get_window(number_of_neighbors=4, words=words, current_word_id=i),
                #         errorLength=len(word),
                #         matchedText=word,
                #         message="Perhaps incorrect usage of verb form.",
                #         offset=sent_decontracted.find(word),
                #         offsetInContext=None,
                #         replacements=replacements,
                #         ruleId='INCORRECT_VERB',
                #         ruleIssueType='Mistake',
                #         sentence=sentence
                #     )
                #     err.offsetInContext = err.context.find(word)
                #     possible_errors.append(err.dict())
                #     print('ERROR CAUGHT')
                #     print(f"VERB check: {word} -- {results}")

    return possible_errors, usage_hints


def _get_dependency_parsing_tags(sent: str):
    pos_tagged_raw = dependency_parser.predict(sent)
    tagged_words = [(word, pos_tagged_raw['pos'][i], pos_tagged_raw['predicted_dependencies'][i])
                    for i, word in enumerate(pos_tagged_raw['words'])]
    return tagged_words, pos_tagged_raw['words']


def _get_verb_phrase(dependency_tags, i):
    start_index = 0
    for j in range(i, -1, -1):
        if dependency_tags[j][1] in ['NOUN', 'PRON']:
            start_index = j
            break
    return ' '.join([word for (word, _, _) in dependency_tags[start_index:i + 1]])


def _check_to_too_is_correct(word, i, words):
    options = ['too', 'to']
    compiled = _replace_token_in_sentence(words, i, [MASK])
    res = happy_roberta.predict_mask(compiled, options=options, num_results=len(options))
    return word == res[0]['word']


async def _check_word_usage(words, sentence, sent_decontracted, tagged_words):
    possible_erros_usages = []
    for i, (word, tag, _) in enumerate(tagged_words):

        # Проверить на односложные слова
        if len(word) == 1 and word.isalpha() and word not in ONE_LETTER_WORDS and \
                not (word == 'e' and words[i + 1] == '-'):
            sr = SingleResult(
                category="HINTS",
                context=_get_window(number_of_neighbors=4, words=words, current_word_id=i),
                errorLength=len(word),
                matchedText=word,
                message="Perhaps one letter is the typo.",
                offset=sent_decontracted.find(word),
                offsetInContext=None,
                replacements=[],
                ruleId='ONE_LETTER_WORD',
                ruleIssueType='Hint',
                sentence=sentence
            )
            possible_erros_usages.append(sr)
        word = word.lower()
        to_too_correct = _check_to_too_is_correct(word, i, words) if word.lower() in ['to', 'too'] else True
        if not to_too_correct:
            sr = SingleResult(
                category="GRAMMAR",
                context=_get_window(number_of_neighbors=4, words=words, current_word_id=i),
                errorLength=len(word),
                matchedText=word,
                message="Incorrect to/too usage",
                offset=sent_decontracted.find(word),
                offsetInContext=None,
                replacements=['to' if word == 'too' else 'too'],
                ruleId='TO_TOO',
                ruleIssueType='Mistake',
                sentence=sentence
            )
            possible_erros_usages.append(sr)

        word_pos_tag = pos_tag([word])[0]
        # прилагательные, существительные, наречия
        if word_pos_tag[1] in ("JJ", "JJR", "JJS",
                               "RB", "RBS", "RBR"):
            compiled_str = _replace_token_in_sentence(words, i, [MASK])
            results = {i: word_info for i, word_info in
                       enumerate(happy_roberta.predict_mask(compiled_str, num_results=10))}
            predicted_words = [r['word'] for r in results.values()]

            if word not in predicted_words:
                conjugation = default_conjugator.conjugate(wnl.lemmatize(word), subject="abbrev")

                verb_forms = []
                config = conjugation.conjug_info
                groups = ['indicative', 'infinitive', 'imperative']
                for g in groups:
                    for verb_group, value in config[g].items():
                        try:
                            verb_forms.extend(value.values())
                        except Exception as e:
                            print(e)
                            verb_forms.append(value)
                verb_forms = set(verb_forms)

                str_for_typos_checking = " ".join(verb_forms)

                errs = await send_to_language_tool(str_for_typos_checking,
                                                   f'http://{settings.LT_HOST}:{settings.LT_PORT}/v2{settings.LAGUAGETOOL_API_CHECK}')

                mathces = []
                for err_match in errs['matches']:
                    sr, _ = create_mistake_description(err_match)
                    mathces.append(sr)

                corrected_text = correct(str_for_typos_checking, mathces)
                word_forms = set(corrected_text.split())
                common = word_forms.intersection(set(predicted_words))

                if common != set():
                    sr = SingleResult(
                        category="HINTS",
                        context=_get_window(number_of_neighbors=4, words=words, current_word_id=i),
                        errorLength=len(word),
                        matchedText=word,
                        message="Perhaps incorrect form of the word.",
                        offset=sent_decontracted.find(word),
                        offsetInContext=None,
                        replacements=list(common),
                        ruleId='INCORRECT_WORD_USAGE',
                        ruleIssueType='Hint',
                        sentence=sentence
                    )
                    possible_erros_usages.append(sr)
    return possible_erros_usages


async def additional_check_with_roBERTa(text, source_text) -> AdditionalChecking:
    real_text = sent_tokenize(text)
    real_sentences = sent_tokenize(source_text)

    possible_aux_errors_list = []
    possible_verb_errors_list = []
    usage_hints_list = []

    for i, sentence in enumerate(real_text):
        print(sentence)

        sent_decontracted = _decontracted(sentence)

        print(sent_decontracted)

        tagged_words, words = _get_dependency_parsing_tags(sent_decontracted)
        print(f"Words: {words}")

        static_errors = _static_rules(sentence)

        # AUX verb checking
        # if AUX mistake has been found, sentence is being fixed

        possible_aux_errors, fixed_sentence = _check_auxiliary_verbs(words, real_sentences[i], sent_decontracted, tagged_words)
        possible_aux_errors_list.extend(possible_aux_errors)

        # VERB checking
        # if AUX errors are found, verb checking uses fixed sentence
        # INCORRECT WORD (VERB) USAGE could be found

        if fixed_sentence is not None:
            tagged_words_for_verb_checking, words_for_verb_checking = _get_dependency_parsing_tags(fixed_sentence)
            possible_verb_errors, usage_hints = await _check_verbs(words_for_verb_checking, fixed_sentence, sent_decontracted,
                                                             tagged_words_for_verb_checking)
        else:
            possible_verb_errors, usage_hints = await _check_verbs(words, real_sentences[i], sent_decontracted,
                                                             tagged_words)

        possible_verb_errors_list.extend(possible_verb_errors)
        usage_hints_list.extend(usage_hints)

        # Check correct word usage
        usage_hints_all_verds = await _check_word_usage(words, real_sentences[i], sent_decontracted, tagged_words)
        usage_hints_list.extend(usage_hints_all_verds)

        # add static erros in conditions of
        for se in static_errors:
            if se.category == 'HINTS':
                usage_hints_list.append(se)

        print("==================================================================\n\n")
    adds = AdditionalChecking(
        auxiliary_verbs=possible_aux_errors_list,
        verbs=possible_verb_errors_list,
        incorrect_word_usage=usage_hints_list
    )
    return adds


def _find_closest(word, predicted_words):
    pr_with_dist = {}
    for pr in predicted_words:
        dist = edit_distance(word, pr)
        pr_with_dist[pr] = dist

    return {k: v for k, v in sorted(pr_with_dist.items(), key=lambda item: item[1])}


def _replace_token_in_sentence(words: List[str], id_of_token_to_be_replaced: int, token_for_replacement: List[str]):
    last = [] if id_of_token_to_be_replaced == len(words) - 1 else words[id_of_token_to_be_replaced + 1:]
    return ' '.join(words[:id_of_token_to_be_replaced] + token_for_replacement + last)


def _decontracted(phrase):
    # specific
    phrase = re.sub(r"won[\’\'\`]t", "will not", phrase)
    phrase = re.sub(r"can[\’\'\`]t", "can not", phrase)

    # general
    phrase = re.sub(r"n[\’\'\`]t", " not", phrase)
    phrase = re.sub(r"[\’\'\`]re", " are", phrase)

    # phrase = re.sub(r"[\’\'\`]s", " is", phrase)
    # phrase = re.sub(r"[\’\'\`]d", " would", phrase)

    phrase = re.sub(r"[\’\'\`]ll", " will", phrase)
    phrase = re.sub(r"[\’\'\`]t", " not", phrase)
    phrase = re.sub(r"[\’\'\`]ve", " have", phrase)
    phrase = re.sub(r"[\’\'\`]m", " am", phrase)
    return phrase


def _get_first_possible_auxiliary(results: List[Dict[str, Any]]):
    for i, word in results.items():
        if word['word'] in POSSIBLES_AUXILIARY:
            yield results[i]


def _check_first_level_passed(word, results):
    # if predicted word matches with source text word, we assume that source word is correct
    # we take into consideration only auxiliary verbs

    aux_generator = _get_first_possible_auxiliary(results)
    first = next(aux_generator, None)
    try:
        return first.get("word") == word
    except Exception:
        return False


def _get_window(number_of_neighbors, words, current_word_id):
    last_index = len(words)
    left_border = 0 if (current_word_id - number_of_neighbors) < 0 else (current_word_id - number_of_neighbors)
    right_border = last_index if (current_word_id + number_of_neighbors) > last_index else (
            current_word_id + number_of_neighbors)

    return " ".join(words[left_border:right_border + 1])


async def send_to_language_tool(text: str, url: str, with_disabled_rules: bool = False) -> Dict[str, Any]:
    params = Native_LT_API_Model(text=text, disabledRules=",".join(RULES_TO_DISABLE))
    if with_disabled_rules:
        params.enabledCategories = ','.join(DISABLED_CATEGORIES)
        params.enabledRules = ','.join(DISABLED_RULES)
        params.enabledOnly = True

    try:
        logging.info(f"send to LT..")
        response = requests.post(
            url=url,
            params=params.dict(exclude_none=True)
        )
    except:
        raise Exception(f'LanguageTool server is not responding.')

    if response.status_code != 200:
        raise Exception(f'Response status code: {response.status_code}, response: {response.text}')

    return json.loads(response.text)


def create_mistake_description(single_result: dict) -> SingleResultFromNativeApi:
    sr = SingleResultFromNativeApi(
        category=single_result['rule']['category']['name'],
        ruleId=single_result['rule']['id'],
        context=single_result['context']['text'],
        errorLength=single_result['length'],
        message=single_result['message'],
        offset=single_result['offset'],
        offsetInContext=single_result['context']['offset'],
        replacements=[rr['value'] for rr in single_result['replacements']],
        ruleIssueType=single_result['rule']['issueType'],
        sentence=single_result['sentence'],
        category_id=single_result['rule']['category']['id']
    )
    sr.matchedText = sr.context[sr.offsetInContext:sr.offsetInContext + sr.errorLength]
    return sr, sr.ruleId


def mistake_already_caught(results: List[SingleResultFromNativeApi],
                           current_mistake: SingleResultFromNativeApi) -> bool:
    for o in results:
        if current_mistake.offset == o.offset and current_mistake.ruleId == o.ruleId:
            return True
    return False


def correct(text: str, matches: List[SingleResult]) -> str:
    """Automatically apply suggestions to the text."""
    ltext = list(text)
    matches = [match for match in matches if match.replacements]
    errors = [ltext[match.offset:match.offset + match.errorLength]
              for match in matches]
    correct_offset = 0
    for n, match in enumerate(matches):
        frompos, topos = (correct_offset + match.offset,
                          correct_offset + match.offset + match.errorLength)
        if ltext[frompos:topos] != errors[n]:
            continue

        # repl = match.replacements[0]
        possible_replacements = [r for r in match.replacements if len(r.split()) == 1]
        if len(possible_replacements) > 1:

            context = ltext.copy()
            context[frompos:topos] = list('[MASK]')
            s = ''.join(context)
            res = happy_roberta.predict_mask(s, options=possible_replacements,
                                             num_results=len(possible_replacements))
            repl = res[0]['word']
        else:
            repl = match.replacements[0]

        ltext[frompos:topos] = list(repl)
        correct_offset += len(repl) - len(errors[n])
    return ''.join(ltext)


def get_errors_with_replacements(errs_matches):
    return [err for err in errs_matches['matches'] if err['replacements'] != []
            and err['rule']['category']['id'] not in REDUNDANT_CATEGORIES]


def _static_rules(sentence: str) -> Optional[List[SingleResult]]:
    errors_caught_with_static_rules = []
    words = word_tokenize(sentence)
    if "peoples" in words:
        context = _get_window(2, words, words.index("peoples"))
        errors_caught_with_static_rules.append(
            SingleResult(category="HINTS",
                         context=context,
                         errorLength=len("peoples"),
                         matchedText='peoples',
                         message="Word 'Peoples' may be used in the context of different nations. In other cases 'People' should singular.",
                         offset=sentence.find('peoples'),
                         offsetInContext=context.find("peoples"),
                         replacements=['people'],
                         ruleId="PEOPLE_PEOPLES",
                         ruleIssueType='Hint',
                         sentence=sentence
                         )
        )

    return errors_caught_with_static_rules

def _check_strange_replacements(sr):
    replacements =sr.replacements
    if sr.ruleId == "PEOPLE_VBZ":
        replacements =[r for r in sr.replacements if r != "am"]

    return replacements


