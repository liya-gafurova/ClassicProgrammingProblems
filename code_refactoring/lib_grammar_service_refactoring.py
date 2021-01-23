import copy
import hashlib
import itertools
import string
from datetime import datetime
from typing import Tuple
import requests
import re
import logging.config

from nltk import sent_tokenize, pos_tag, word_tokenize

from models.grammar_models import (SingleResult, Native_LT_API_Model, RuleIdNumber,
                                   SingleResultFromLanguageTool, AdditionalChecking)
from core.lib import DISABLED_CATEGORIES, DISABLED_RULES
from services.nlp_models import predict_mask, spacy_tokenizer, lemmatizer, default_conjugator
from services.constants import *

LT_URL = f'http://{settings.LT_HOST}:{settings.LT_PORT}/v2{settings.LAGUAGETOOL_API_CHECK}'


def _predict_words(words: List[str], position: int, num_results: int = 5, options=None):
    """
    :param words: list of words
    :param position: word index to be replaced with <mask>
    :param num_results: number of words to be predicted with masked language modeling (roBERTa here)
    :param options: predict words from given options.
    :return: predicted words.
    """
    compiled_str = _replace_token_in_sentence(words, position, MASK)
    results = predict_mask(compiled_str, options=options, num_results=num_results)
    return results


def is_random_set_of_characters(text):
    # delete punctuation
    text = re.sub(f"[{string.punctuation}]", ' ', text)

    # tokenize with spacy and find out of vocabulary tokens
    parsed_document = spacy_tokenizer(text)
    all_tokens = [(t, t.is_oov) for t in parsed_document]
    out_of_vocabulary_tokens = [t for t in all_tokens if t[1]]

    # find out of vocabulary tokens quantity
    oov_percentage = len(out_of_vocabulary_tokens) / len(parsed_document)

    return oov_percentage > 0.4


def _is_in_same_contracted_group(word, predicted_word):
    if word == predicted_word:
        return True
    for group in contracted_groups:
        if word in group and predicted_word in group:
            return True
    return False


def _check_auxiliary_verbs(words: List[str], sentence: str, sent_decontracted: str,
                           tagged_words: Tuple[str, str, str]) -> List[SingleResult]:
    """

    :param words: list of words in sentence
    :param sentence: real sentence, that has been written by user and not corrected at any way
    :param sent_decontracted: sentence that has been checked and corrected with LT, and part of contraction are removed
    :param tagged_words: [(word, universal_pos, detailed_pos), ...].
    :return: list of found mistakes and fixed sentence/
     Fixed sentence is returned if we have high probability that predicted verb is from another Tense group.
    """
    possible_errors = []
    fixed_sentence: Optional[str] = None
    for i, (word, pos, _) in enumerate(tagged_words):
        if pos == 'AUX':
            word = word.lower()
            err, to_be_corrected = False, False

            results = _predict_words(words, i, options=AUX_FULL_FORMS, num_results=len(AUX_FULL_FORMS))

            first, second, third, forth = results[0], results[1], results[2], results[3]
            logging.info(f"NEW: {word} -- {pos} -- {[first, second, third, forth]}")
            if first['softmax'] > 0.79 and not _is_in_same_contracted_group(word, first['word']):
                err, to_be_corrected = True, True
            elif first['softmax'] < 0.1 and not _is_in_same_contracted_group(word, first['word']) and \
                    not _is_in_same_contracted_group(word, second['word']) and \
                    not _is_in_same_contracted_group(word, third['word']) and \
                    not _is_in_same_contracted_group(word, forth['word']):
                err = True
            elif not _is_in_same_contracted_group(word, first['word']) and \
                    not _is_in_same_contracted_group(word, second['word']) and \
                    not _is_in_same_contracted_group(word, third['word']):
                err = True

            replacements = [first['word'], second['word']]

            if err:
                if to_be_corrected:
                    fixed_sentence = _replace_token_in_sentence(words, i, replacements[0])
                err_sr = SingleResult(
                    category="HINTS",
                    context=_get_mistake_context(number_of_neighbors=2, words=words, current_word_id=i),
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


def _first_level_verb_check(word, predicted_words):
    for predicted_word in predicted_words:
        if word == predicted_word:
            return True

        for group in contracted_groups:
            if word in group and predicted_word in group:
                return True
    return False


def _is_in_different_pos_group(real_verb_pos, first_result_pos):
    for group_name, group in POS_VERB_SIMILAR_GROUPS.items():
        if real_verb_pos in group and first_result_pos in group:
            return False
    return True


def _at_least_one_verb_in_results(results_tags):
    for tag in results_tags:
        if tag.find("VB") > -1 or tag.find('MD') > -1:
            return True
    return False


def _has_verb_mistake(word, results) -> Tuple[bool, Optional[bool]]:
    real_verb_nltk_post_tagged, results_nltk_pos_tagged = pos_tag([word])[0], pos_tag(
        [r['word'] for r in results])
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


def _get_possible_forms_of_verb(word, pos='v'):
    conjugation = default_conjugator.conjugate(lemmatizer.lemmatize(word, pos=pos), subject="abbrev")

    verb_forms = []
    config = conjugation.conjug_info

    for g in ['indicative', 'infinitive', 'imperative']:
        for verb_group, value in config[g].items():
            try:
                verb_forms.extend(value.values())
            except AttributeError:
                verb_forms.append(value)
    verb_forms = set(verb_forms)

    str_for_typos_checking = " ".join(verb_forms)

    errs = send_to_language_tool(str_for_typos_checking, LT_URL)

    matches = []
    for err_match in errs['matches']:
        sr, _ = create_mistake_description(err_match)
        matches.append(sr)

    corrected_text = correct_errors_found_with_Language_Tool(str_for_typos_checking, matches)
    corrected_text = corrected_text.lower().replace('to', '')
    verb_forms = set(corrected_text.split())
    return verb_forms


def _check_verbs(words, real_sentence, sent_decontracted, tagged_words):
    """
    :param words: list
    :param real_sentence: sentence written by user
    :param sent_decontracted:  sentence corrected with LT and without contractions
    :param tagged_words: [(word, universal_pos, detailed_pos), ...]
    :return: return grammar errors within verbs with issue type = 'Mistake'
    and incorrect verb usage with issue type 'Hint'.
    """
    possible_errors, usage_hints = [], []
    for i, (word, pos, _) in enumerate(tagged_words):
        if pos == 'VERB':
            # word = word.lower()

            results = _predict_words(words, i, num_results=10)

            predicted_words = [r['word'] for r in results if r['word'] != '']
            real_verb_nltk_post_tagged, results_nltk_pos_tagged = pos_tag([word])[0], pos_tag(predicted_words)
            most_common_tag = _get_most_common_tag(results_nltk_pos_tagged)
            results_tags = [res_tagged[1] for res_tagged in results_nltk_pos_tagged]
            if not _first_level_verb_check(word, predicted_words):

                if not _at_least_one_verb_in_results(results_tags) or \
                        results[0]['softmax'] > HIGH_USAGE_THRESHOLD \
                        and not _is_in_different_pos_group(real_verb_nltk_post_tagged[1], most_common_tag):
                    err = SingleResult(
                        category="HINTS",
                        context=_get_mistake_context(number_of_neighbors=4, words=words, current_word_id=i),
                        errorLength=len(word),
                        matchedText=word,
                        message="Perhaps incorrect verb usage.",
                        offset=sent_decontracted.find(word),
                        offsetInContext=None,
                        replacements=predicted_words[:3],
                        ruleId='INCORRECT_WORD_USAGE',
                        ruleIssueType='Hint',
                        sentence=real_sentence
                    )
                    err.offsetInContext = err.context.find(word)
                    usage_hints.append(err)
                    continue

                verb_forms = _get_possible_forms_of_verb(word)
                common = verb_forms.intersection(set(predicted_words))
                verb_form_predictions = _predict_words(words, i, num_results=len(verb_forms), options=list(verb_forms))
                predicted_verbs = [res['word'] for res in verb_form_predictions]

                if common != set() \
                        or (most_common_tag in VERB_TAGS and _is_in_different_pos_group(real_verb_nltk_post_tagged[1],
                                                                                        most_common_tag)) \
                        or word not in predicted_verbs[:3]:
                    sr = SingleResult(
                        category="HINTS",
                        context=_get_mistake_context(number_of_neighbors=4, words=words, current_word_id=i),
                        errorLength=len(word),
                        matchedText=word,
                        message="Perhaps incorrect form of the verb.",
                        offset=sent_decontracted.find(word),
                        offsetInContext=None,
                        replacements=list(common) if common != set() else predicted_words[:3],
                        ruleId='INCORRECT_VERB',
                        ruleIssueType='Hint',
                        sentence=real_sentence
                    )
                    sr.offsetInContext = sr.context.find(word)
                    possible_errors.append(sr)

    return possible_errors, usage_hints


def _get_pos_tags(sent: str) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    pos_tagged_raw = spacy_tokenizer(sent)
    tagged_words = [(token.text, token.pos_, token.tag_) for token in pos_tagged_raw]
    words = [token.text for token in pos_tagged_raw]
    # returns [('word', 'universal_pos_tag', 'detailed_pos_tag'), ...] and ['word', ...]
    return tagged_words, words


def _check_to_too_is_correct(word, i, words):
    options = ['too', 'to']
    results = _predict_words(words, i, options=options, num_results=len(options))
    return word == results[0]['word']


def _check_word_usage(words, sentence, sent_decontracted, tagged_words):
    possible_errors_usages = []
    for i, (word, tag, _) in enumerate(tagged_words):

        # Проверить на односложные слова
        if len(word) == 1 and word.isalpha() and word not in ONE_LETTER_WORDS and \
                not (word == 'e' and words[i + 1] == '-'):
            sr = SingleResult(
                category="HINTS",
                context=_get_mistake_context(number_of_neighbors=4, words=words, current_word_id=i),
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
            sr.offsetInContext = sr.context.find(word)
            possible_errors_usages.append(sr)
        word = word.lower()
        to_too_correct = _check_to_too_is_correct(word, i, words) if word.lower() in ['to', 'too'] else True
        if not to_too_correct:
            sr = SingleResult(
                category="HINTS",
                context=_get_mistake_context(number_of_neighbors=4, words=words, current_word_id=i),
                errorLength=len(word),
                matchedText=word,
                message="Incorrect to/too usage",
                offset=sent_decontracted.find(word),
                offsetInContext=None,
                replacements=['to' if word == 'too' else 'too'],
                ruleId='TO_TOO',
                ruleIssueType='Hint',
                sentence=sentence
            )
            sr.offsetInContext = sr.context.find(word)
            possible_errors_usages.append(sr)

        word_pos_tag = pos_tag([word])[0]

        # adverbs and adjectives
        if word_pos_tag[1] in ("JJ", "JJR", "JJS",):
            results = _predict_words(words, i, num_results=10)
            predicted_words = [r['word'] for r in results]

            if word not in predicted_words:
                word_forms = _get_possible_forms_of_verb(word, pos='a')
                common = word_forms.intersection(set(predicted_words))

                if common != set():
                    sr = SingleResult(
                        category="HINTS",
                        context=_get_mistake_context(number_of_neighbors=4, words=words, current_word_id=i),
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
                    sr.offsetInContext = sr.context.find(word)
                    possible_errors_usages.append(sr)

    return possible_errors_usages


def _check_prepositions(words, real_sentence, sent_decontracted, tagged_words) -> List[SingleResult]:
    prepositions_errors = []
    for i, (word, tag, _) in enumerate(tagged_words):
        if tag in ['ADP', 'CCONJ']:
            # we are going to check prepositions and postpositions. And also  coordinating conjunction
            results = _predict_words(words, i, num_results=5)
            results_dict = {result['word']: result['softmax'] for result in results}
            softmax_sum = sum(results_dict.values())
            if word not in results_dict.keys() and softmax_sum > 0.8:
                print(f"CANDIDATES for {word}: {results_dict}")
                logging.info(f"CANDIDATES for {word}: {results_dict}")
                sr = SingleResult(
                    category="HINTS",
                    context=_get_mistake_context(number_of_neighbors=4, words=words, current_word_id=i),
                    errorLength=len(word),
                    matchedText=word,
                    message="Perhaps incorrect preposition usage.",
                    offset=sent_decontracted.find(word),
                    offsetInContext=None,
                    replacements=list(results_dict.keys())[:3],
                    ruleId='INCORRECT_WORD_USAGE',
                    ruleIssueType='Hint',
                    sentence=real_sentence
                )
                sr.offsetInContext = sr.context.find(word)
                prepositions_errors.append(sr)

    return prepositions_errors


class ArticleMistake(Exception):
    def __init__(self, position_token="the", replacement='',
                 message: str = "Article 'the' is needed in the position: {}", position: int = 0):
        self.message = message.format(position)
        self.position_token = position_token
        self.replacement = replacement
        super().__init__(self.message)


def _check_artickes(words, sentence, sent_decontracted, tagged_words):
    doc = spacy_tokenizer(sentence)
    errs = []
    for ent in doc.ents:
        if ent.label_ in ['GPE']:
            # get NP containing GPE NER (country, state, city)
            NPs = [np.text.lower().split() for np in doc.noun_chunks if
                   (np.text.find(ent.text) > -1 or ent.text.find(np.text) > -1) and np.text != 'i']
            NPs_list = list(itertools.chain.from_iterable(NPs))
            chunk = ' '.join(NPs_list)
            GPE_subphrase = set(ent.text.lower().split()).union(set(NPs_list))

            # Check NP phrase with GPE entity with country names in provided list
            try:
                if chunk.find('the') > -1:
                    for country in NO_COUNTRIES:
                        diff = GPE_subphrase.difference(set(country.split()))
                        if diff == {'the'} or diff == {'a'}:
                            raise ArticleMistake(position_token=list(diff).pop(),
                                                 replacement='',
                                                 message="There should not be an article in the position: {}",
                                                 position=sentence.find(ent.text))
                elif chunk.find('the') == -1:
                    for country in THE_COUNTRIES:
                        if set(country.split()).difference(GPE_subphrase) == {'the'}:
                            raise ArticleMistake(position_token=ent.text,
                                                 replacement='the',
                                                 message="Article 'the' is needed in the position: {}",
                                                 position=sentence.find(ent.text))
            except ArticleMistake as e:
                print(e.message)
                sr = SingleResult(category="HINTS",
                                  context='',
                                  errorLength=3,
                                  matchedText=e.position_token,
                                  message=e.message,
                                  offset=sentence.find(e.position_token),
                                  replacements=[e.replacement],
                                  ruleId="ARTICLES_BEFORE_COUNTRIES",
                                  ruleIssueType='Hint',
                                  sentence=sentence
                                  )
                sr.offsetInContext = 0
                errs.append(sr)

    return errs


def additional_check_with_roBERTa(prettified_text: str, source_text: str, categories: dict) -> AdditionalChecking:
    prettified_sentences = sent_tokenize(prettified_text)
    real_sentences = sent_tokenize(source_text)

    possible_aux_errors_list = []
    possible_verb_errors_list = []
    usage_hints_list = []

    for i, sentence in enumerate(prettified_sentences):

        # get rid of contractions, like I`m -> I am
        sent_decontracted = _decontracted(sentence)

        tagged_words, words = _get_pos_tags(sent_decontracted)
        print(f"Words: {words}")

        static_errors = _static_rules(sentence)

        # AUX verb checking
        # if AUX mistake has been found, sentence is being fixed
        # TODO real_sentences[i] <- sentence
        possible_aux_errors, fixed_sentence = _check_auxiliary_verbs(words, sentence, sent_decontracted,
                                                                     tagged_words)
        possible_aux_errors_list.extend(possible_aux_errors)

        # VERB checking
        # if AUX errors are found, sentence is being fixed with correct aux verb. verb checking uses fixed sentence
        # INCORRECT WORD (VERB) USAGE could be found
        # TODO real_sentences[i] <- sentence
        params = (words, sentence, sent_decontracted, tagged_words)
        if fixed_sentence:
            tagged_words_for_verb_checking, words_for_verb_checking = _get_pos_tags(fixed_sentence)
            params = (words_for_verb_checking, sentence, sent_decontracted, tagged_words_for_verb_checking)

        possible_verb_errors, usage_hints = _check_verbs(*params)

        possible_verb_errors_list.extend(possible_verb_errors)
        usage_hints_list.extend(usage_hints)

        # Check correct word usage
        # TODO real_sentences[i] <- sentence
        usage_hints_all_verbs = _check_word_usage(words, sentence, sent_decontracted, tagged_words)
        usage_hints_list.extend(usage_hints_all_verbs)

        # Check prepositions
        possible_preposition_errors = _check_prepositions(words, sentence, sent_decontracted, tagged_words)
        usage_hints_list.extend(possible_preposition_errors)

        # Check articles before Countries
        possible_articles_mistakes = _check_artickes(words, sentence, sent_decontracted, tagged_words)
        usage_hints_list.extend(possible_articles_mistakes)

        # add static errors in conditions of correct word usage
        for se in static_errors:
            if se.category == 'HINTS':
                usage_hints_list.append(se)

        print("==================================================================\n\n")
    for err in possible_aux_errors_list + possible_verb_errors_list + usage_hints_list:
        categories[err.category]['ruleIds'].append(
            RuleIdNumber(ruleId=err.ruleId, number_of_mistakes=1).dict()
        )  # HERE
        categories[err.category]['mistakes_number'] += 1

    adds = AdditionalChecking(
        auxiliary_verbs=possible_aux_errors_list,
        verbs=possible_verb_errors_list,
        incorrect_word_usage=usage_hints_list
    )
    return adds, categories


def _replace_token_in_sentence(words: List[str], id_of_token_to_be_replaced: int, token_for_replacement: str):
    last = [] if id_of_token_to_be_replaced == len(words) - 1 else words[id_of_token_to_be_replaced + 1:]
    return ' '.join(words[:id_of_token_to_be_replaced] + [token_for_replacement] + last)


def _decontracted(phrase):
    # specific
    phrase = re.sub(r"won[’'`]t", "will not", phrase)
    phrase = re.sub(r"can[’'`]t", "can not", phrase)

    # general
    phrase = re.sub(r"n[’'`]t", " not", phrase)
    phrase = re.sub(r"[’'`]re", " are", phrase)

    # phrase = re.sub(r"[’'`]s", " is", phrase)
    # phrase = re.sub(r"[’'`]d", " would", phrase)

    phrase = re.sub(r"[’'`]ll", " will", phrase)
    phrase = re.sub(r"[’'`]t", " not", phrase)
    phrase = re.sub(r"[’'`]ve", " have", phrase)
    phrase = re.sub(r"[’'`]m", " am", phrase)
    return phrase


def _most_probable_aux_generator(results: List[Dict[str, Any]]):
    for i, word in results.items():
        if word['word'] in POSSIBLES_AUXILIARY:
            yield results[i]


def _check_if_most_probable_aux_is_our_word(word, results):
    # if predicted word matches with source text word, we assume that source word is correct
    # we take into consideration only auxiliary verbs

    aux_generator = _most_probable_aux_generator(results)
    first = next(aux_generator, None)
    try:
        return first.get("word") == word
    except AttributeError:
        return False


def _get_mistake_context(number_of_neighbors, words, current_word_id):
    last_index = len(words)
    left_border = 0 if (current_word_id - number_of_neighbors) < 0 else (current_word_id - number_of_neighbors)
    right_border = last_index if (current_word_id + number_of_neighbors) > last_index else (
            current_word_id + number_of_neighbors)

    return " ".join(words[left_border:right_border + 1])


def send_to_language_tool(text: str, url: str, with_disabled_rules: bool = False) -> Dict[str, Any]:

    params = Native_LT_API_Model(text=text, disabledRules=",".join(RULES_TO_DISABLE))
    if with_disabled_rules:
        params.enabledCategories = ','.join(DISABLED_CATEGORIES)
        params.enabledRules = ','.join(DISABLED_RULES)
        params.enabledOnly = True

    try:
        response = requests.post(
            url=url,
            params=params.dict(exclude_none=True)
        )
    except:
        raise Exception(f'LanguageTool server is not responding.')

    if response.status_code != 200:
        raise Exception(f'Response status code: {response.status_code}, response: {response.text}')

    return json.loads(response.text)


def mistake_already_caught(results: List[SingleResultFromLanguageTool],
                           current_mistake: SingleResultFromLanguageTool) -> bool:
    for o in results:
        if current_mistake.offset == o.offset and current_mistake.ruleId == o.ruleId:
            return True
    return False


def correct_errors_found_with_Language_Tool(text: str, matches: List[SingleResult]) -> str:
    """Automatically apply suggestions found with LanguageTool to the text."""
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

        possible_replacements = [r for r in match.replacements if len(r.split()) == 1]
        if len(possible_replacements) > 1:

            context = ltext.copy()
            context[frompos:topos] = list(MASK)
            compiled_text = nltk.sent_tokenize(''.join(context))
            compiled_str = next((sent for sent in sent_tokenize(compiled_text) if sent.find(MASK) > -1), None)
            res = predict_mask(compiled_str, options=possible_replacements,
                               num_results=len(possible_replacements))
            replacement = res[0]['word']
        else:
            replacement = match.replacements[0]

        ltext[frompos:topos] = list(replacement)
        correct_offset += len(replacement) - len(errors[n])
    return ''.join(ltext)


def get_errors_with_replacements(errs_matches):
    # get errors from LT which have replacements
    return [err for err in errs_matches['matches'] if err['replacements'] != []
            and err['rule']['category']['id'] not in REDUNDANT_CATEGORIES]


def _static_rules(sentence: str) -> Optional[List[SingleResult]]:
    """
    :param sentence: sentence from users text
    :return: list of errors found with heuristic methods
    """
    errors_caught_with_static_rules = []
    words = word_tokenize(sentence)
    if "peoples" in words:
        context = _get_mistake_context(2, words, words.index("peoples"))
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


def check_for_unsuitable_replacements(sr):
    replacements = sr.replacements
    has_to_be_correct = True
    if sr.ruleId == "PEOPLE_VBZ":
        replacements = [r for r in sr.replacements if r != "am"]
    if sr.ruleId == "EN_PLAIN_ENGLISH_REPLACE":
        replacements = [r for r in sr.replacements if r != "[OMIT]"]  # "OMIT" tells you to cut the  matched text
    if sr.ruleId == 'TOO_DETERMINER':  # this rule will be checked with roberta further
        replacements = [sr.matchedText]
        has_to_be_correct = False

    if not replacements:
        has_to_be_correct = False

    return replacements, has_to_be_correct


def create_mistake_description(single_result: dict) -> SingleResultFromLanguageTool:
    sr = SingleResultFromLanguageTool(
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


def create_hash_for_text(text: str):
    current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    hash_object = hashlib.md5(text.encode() + current_time.encode())
    return hash_object.hexdigest()


def reorganize_categories(categories: dict):
    out = copy.deepcopy(categories)
    for category, value in categories.items():
        rules = value['ruleIds']
        value['ruleIds'] = list()
        d = {}
        values = []
        for rule in rules:
            rule_name = list(rule.items())[0][1]
            if d.get(rule_name) is None:
                d[rule_name] = 1
            else:
                d[rule_name] += 1
        for r, v in d.items():
            rr = RuleIdNumber(ruleId=r, number_of_mistakes=v)
            values.append(rr.dict())
        out[category]['ruleIds'] = values
    return out


def get_avg_grade(readability_desc: str):
    """
    :param readability_desc: textstat library returns result in format:
    "{}{} and {}{} grade".format(lower_score, get_grade_suffix(lower_score),
                upper_score, get_grade_suffix(upper_score)
            ).
    :return: average grade gotten from "lower_score" and "upper_score" values.
    """
    numbers = re.findall(r'\d+', readability_desc)
    return sum(list(map(int, numbers))) / len(numbers)

