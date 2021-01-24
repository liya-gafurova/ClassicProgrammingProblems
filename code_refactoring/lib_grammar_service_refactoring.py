import hashlib
import itertools
import string
from collections import Counter
from datetime import datetime
from typing import Tuple, List

import nltk
import requests
import re
import logging.config

from nltk import sent_tokenize, pos_tag, word_tokenize, WhitespaceTokenizer

from models.grammar_models import (SingleResult, Native_LT_API_Model, RuleIdNumber,
                                   SingleResultFromLanguageTool, AdditionalChecking, Categories, CategoryDescription)
from core.lib import DISABLED_CATEGORIES, DISABLED_RULES

from .decorators import create_logger, logged
from .nlp_models import predict_mask, spacy_tokenizer, lemmatizer, default_conjugator
from .constants import *

LT_URL = f'http://{settings.LT_HOST}:{settings.LT_PORT}/v2{settings.LAGUAGETOOL_API_CHECK}'

logger = create_logger()

class TextGrammarCheckingResult:
    """
    class for creating checking pipeline and storing roBERTa grammar checking
    """

    def __init__(self, source_text: str, categories: dict, corrected_text: str = None):
        self.source_text = source_text
        self.corrected_text = corrected_text if corrected_text else source_text
        self.corrected_sentences = sent_tokenize(corrected_text)
        self.real_sentences = sent_tokenize(source_text)

        self._possible_aux_errors_list = []
        self._possible_verb_errors_list = []
        self._usage_hints_list = []

        self.additional_checking_result = AdditionalChecking()
        self.result_analytics = categories if categories else self._create_categories_structure()

    def make_complex_checking(self):

        for i, sentence in enumerate(self.corrected_sentences):

            sent_decontracted = self._decontracted(sentence)
            tagged_words, words = self._get_pos_tags(sent_decontracted)
            logging.info(f'Words: {words}')

            logging.info(f"Start checking with specific rules...")
            static_errors = _static_rules(sentence)
            self._usage_hints_list.extend(static_errors)

            logging.info(f'Starting to check Auxiliary Verbs...')
            possible_aux_errors, fixed_sentence = _check_auxiliary_verbs(words, sentence, sent_decontracted,
                                                                         tagged_words)
            self._possible_aux_errors_list.extend(possible_aux_errors)

            logging.info(f"Starting to check Verbs...")
            params = (words, sentence, sent_decontracted, tagged_words)
            if fixed_sentence:
                tagged_words_for_verb_checking, words_for_verb_checking = self._get_pos_tags(fixed_sentence)
                params = (words_for_verb_checking, sentence, sent_decontracted, tagged_words_for_verb_checking)

            possible_verb_errors, usage_hints = _check_verbs(*params)

            self._possible_verb_errors_list.extend(possible_verb_errors)
            self._usage_hints_list.extend(usage_hints)

            logging.info(f'Starting to check Word Usage...')
            usage_hints_all_verbs = _check_word_usage(words, sentence, sent_decontracted, tagged_words)
            self._usage_hints_list.extend(usage_hints_all_verbs)

            logging.info(f"Starting to check Articles Usage before Country Names")
            possible_articles_mistakes = _check_artickes(words, sentence, sent_decontracted, tagged_words)
            self._usage_hints_list.extend(possible_articles_mistakes)

    @logged(logger)
    def get_result_with_updated_categories(self):

        for err in self._possible_aux_errors_list + self._possible_verb_errors_list + self._usage_hints_list:
            self.result_analytics[err.category]['ruleIds'].append(
                RuleIdNumber(ruleId=err.ruleId, number_of_mistakes=1).dict()
            )
            self.result_analytics[err.category]['mistakes_number'] += 1

        self.additional_checking_result = AdditionalChecking(
            auxiliary_verbs=self._possible_aux_errors_list,
            verbs=self._possible_verb_errors_list,
            incorrect_word_usage=self._usage_hints_list
        )

        return self.additional_checking_result, self.result_analytics

    def _create_categories_structure(self):
        categories_analytics = Categories().dict()
        for category, value in categories_analytics.items():
            categories_analytics[category] = CategoryDescription().dict()

        return categories_analytics

    @staticmethod
    def _get_pos_tags(sent: str) -> Tuple[List[Tuple[str, str, str]], List[str]]:
        """ Get POS tags for Sentence"""
        pos_tagged_raw = spacy_tokenizer(sent)
        tagged_words = [(token.text, token.pos_, token.tag_) for token in pos_tagged_raw]
        words = [token.text for token in pos_tagged_raw]
        # returns [('word', 'universal_pos_tag', 'detailed_pos_tag'), ...] and ['word', ...]
        return tagged_words, words

    @staticmethod
    def _decontracted(phrase):
        """ Remove contractions """
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

@logged(logger)
def _create_single_error(word: str, words: List[str], current_word_position: str, sent_decontracted: str, sentence: str,
                         category: str, message: str,
                         replacements: List[str], rule_id: str, rule_issue_type: str,
                         matched_text: str = None, ) -> SingleResult:
    """create Single Error description object."""
    @logged(logger)
    def get_mistake_context(number_of_neighbors, words, current_word_id):
        last_index = len(words)
        left_border = 0 if (current_word_id - number_of_neighbors) < 0 else (current_word_id - number_of_neighbors)
        right_border = last_index if (current_word_id + number_of_neighbors) > last_index else (
                current_word_id + number_of_neighbors)

        return " ".join(words[left_border:right_border + 1])

    context = get_mistake_context(number_of_neighbors=2, words=words, current_word_id=current_word_position)
    error_length = len(word)
    matched_text = matched_text if matched_text else word
    err_sr = SingleResult(
        category=category,
        context=context,
        errorLength=error_length,
        matchedText=matched_text,
        message=message,
        offset=sent_decontracted.find(matched_text),
        offsetInContext=None,
        replacements=replacements,
        ruleId=rule_id,
        ruleIssueType=rule_issue_type,
        sentence=sentence
    )
    err_sr.offsetInContext = err_sr.context.find(matched_text)

    return err_sr


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

@logged(logger)
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
    @logged(logger)
    def is_in_same_contracted_group(word, predicted_word):
        if word == predicted_word:
            return True
        for group in contracted_groups:
            if word in group and predicted_word in group:
                return True
        return False

    possible_errors = []
    fixed_sentence: Optional[str] = None
    for i, (word, pos, _) in enumerate(tagged_words):
        if pos == 'AUX':
            word = word.lower()
            err, to_be_corrected = False, False

            results = _predict_words(words, i, options=AUX_FULL_FORMS, num_results=len(AUX_FULL_FORMS))

            first, second, third, forth = results[0], results[1], results[2], results[3]
            logging.info(f"NEW: {word} -- {pos} -- {[first, second, third, forth]}")
            if first['softmax'] > 0.79 and not is_in_same_contracted_group(word, first['word']):
                err, to_be_corrected = True, True
            elif first['softmax'] < 0.1 and not is_in_same_contracted_group(word, first['word']) and \
                    not is_in_same_contracted_group(word, second['word']) and \
                    not is_in_same_contracted_group(word, third['word']) and \
                    not is_in_same_contracted_group(word, forth['word']):
                err = True
            elif not is_in_same_contracted_group(word, first['word']) and \
                    not is_in_same_contracted_group(word, second['word']) and \
                    not is_in_same_contracted_group(word, third['word']):
                err = True

            replacements = [first['word'], second['word']]

            if err:
                if to_be_corrected:
                    fixed_sentence = _replace_token_in_sentence(words, i, replacements[0])

                possible_errors.append(
                    _create_single_error(word, words, i, sent_decontracted, sentence, "HINTS",
                                        "Perhaps the wrong form of the auxiliary verb", replacements,
                                        'INCORRECT_AUXILIARY_VERB', 'Hint', word)
                )

    return possible_errors, fixed_sentence

@logged(logger)
def _get_possible_forms_of_verb(word:str , pos='v'):
    """
    :param word: word (verb  mostly)
    :param pos: part of speech. ADJ, ADJ_SAT, ADV, NOUN, VERB = "a", "s", "r", "n", "v"
    :return: most probable form of the verb
    """
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

@logged(logger)
def _check_verbs(words, real_sentence, sent_decontracted, tagged_words):
    """
    :param words: list
    :param real_sentence: sentence written by user
    :param sent_decontracted:  sentence corrected with LT and without contractions
    :param tagged_words: [(word, universal_pos, detailed_pos), ...]
    :return: return grammar errors within verbs with issue type = 'Mistake'
    and incorrect verb usage with issue type 'Hint'.
    """

    def first_level_verb_check(word, predicted_words):
        for predicted_word in predicted_words:
            if word == predicted_word:
                return True

            for group in contracted_groups:
                if word in group and predicted_word in group:
                    return True
        return False

    def is_in_different_pos_group(real_verb_pos, first_result_pos):
        for group_name, group in POS_VERB_SIMILAR_GROUPS.items():
            if real_verb_pos in group and first_result_pos in group:
                return False
        return True

    def get_most_common_tag(results_pos_taged):
        try:
            most_common_tag = Counter([tag for word, tag in results_pos_taged]).most_common(1)[0][0]
        except IndexError as e:
            logging.debug(f"No most common tag founded in {results_pos_taged}. Got error: {e.message}")
            most_common_tag = 'None'

        return most_common_tag

    def at_least_one_verb_in_results(results_tags):
        verb = next((tag for tag in results_tags if tag.find("VB") > -1 or tag.find('MD') > -1), False)
        return isinstance(verb, str)

    possible_errors, usage_hints = [], []
    for i, (word, pos, _) in enumerate(tagged_words):
        if pos == 'VERB':

            results = _predict_words(words, i, num_results=10)

            predicted_words = [r['word'] for r in results if r['word'] != '']
            real_verb_nltk_post_tagged, results_nltk_pos_tagged = pos_tag([word])[0], pos_tag(predicted_words)
            most_common_tag = get_most_common_tag(results_nltk_pos_tagged)
            results_tags = [res_tagged[1] for res_tagged in results_nltk_pos_tagged]
            if not first_level_verb_check(word, predicted_words):

                if not at_least_one_verb_in_results(results_tags) or \
                        results[0]['softmax'] > HIGH_USAGE_THRESHOLD \
                        and not is_in_different_pos_group(real_verb_nltk_post_tagged[1], most_common_tag):
                    usage_hints.append(
                        _create_single_error(word, words, i, sent_decontracted, real_sentence, "HINTS",
                                            "Perhaps incorrect word usage", predicted_words[:3],
                                            'INCORRECT_WORD_USAGE', 'Hint', word)
                    )
                    continue

                verb_forms = _get_possible_forms_of_verb(word)
                common = verb_forms.intersection(set(predicted_words))
                verb_form_predictions = _predict_words(words, i, num_results=len(verb_forms), options=list(verb_forms))
                predicted_verbs = [res['word'] for res in verb_form_predictions]

                if common != set() \
                        or (most_common_tag in VERB_TAGS and is_in_different_pos_group(real_verb_nltk_post_tagged[1],
                                                                                       most_common_tag)) \
                        or word not in predicted_verbs[:3]:
                    replacements = list(common) if common != set() else predicted_words[:3]
                    possible_errors.append(
                        _create_single_error(word, words, i, sent_decontracted, real_sentence,
                                            'HINTS', "Perhaps incorrect form of the verb.",
                                             replacements, 'INCORRECT_VERB', 'Hint', word)
                    )

    return possible_errors, usage_hints

@logged(logger)
def _check_word_usage(words, sentence, sent_decontracted, tagged_words):
    """ Check One-Letter-Typos, Adjectives and to/too usage  """

    @logged(logger)
    def check_to_too_is_correct(word, i, words):
        options = ['too', 'to']
        results = _predict_words(words, i, options=options, num_results=len(options))
        return word == results[0]['word']

    possible_errors_usages = []
    for i, (word, tag, _) in enumerate(tagged_words):

        if len(word) == 1 and word.isalpha() and word not in ONE_LETTER_WORDS and \
                not (word == 'e' and words[i + 1] == '-'):
            possible_errors_usages.append(
                _create_single_error(word, words, i, sent_decontracted, sentence,
                                    "HINTS", "Perhaps one letter is a Typo.",
                                     [], 'ONE_LETTER_WORD', "Hint", word)
            )
        word = word.lower()
        to_too_correct = check_to_too_is_correct(word, i, words) if word.lower() in ['to', 'too'] else True
        if not to_too_correct:
            replacements = ['to' if word == 'too' else 'too']
            possible_errors_usages.append(
                _create_single_error(word, words, i, sent_decontracted, sentence,
                                    "HINTS", "Incorrect to/too usage.",
                                     replacements, 'TO_TOO', 'Hint', word)
            )

        word_pos_tag = pos_tag([word])[0]

        # adverbs and adjectives
        if word_pos_tag[1] in ("JJ", "JJR", "JJS",):
            results = _predict_words(words, i, num_results=10)
            predicted_words = [r['word'] for r in results]

            if word not in predicted_words:
                word_forms = _get_possible_forms_of_verb(word, pos='a')
                common = word_forms.intersection(set(predicted_words))

                if common != set():
                    possible_errors_usages.append(
                        _create_single_error(word, words, i, sent_decontracted, sentence,
                                            "HINTS", "Perhaps incorrect form of the word.",
                                             list(common), 'INCORRECT_WORD_USAGE', 'Hint', word)
                    )

    return possible_errors_usages

@logged(logger)
def _check_prepositions(words, real_sentence, sent_decontracted, tagged_words) -> List[SingleResult]:
    """ Check preposition and coordinating conjunction usage  """
    prepositions_errors = []
    for i, (word, tag, _) in enumerate(tagged_words):
        if tag in ['ADP', 'CCONJ']:
            # we are going to check prepositions and postpositions. And also  coordinating conjunction
            results = _predict_words(words, i, num_results=5)
            results_dict = {result['word']: result['softmax'] for result in results}
            softmax_sum = sum(results_dict.values())
            if word not in results_dict.keys() and softmax_sum > 0.8:
                logging.info(f"CANDIDATES for {word}: {results_dict}")
                replacements = list(results_dict.keys())[:3]
                prepositions_errors.append(
                    _create_single_error(word, words, i, sent_decontracted, real_sentence,
                                        "HINTS", "Perhaps incorrect preposition usage.",
                                         replacements, 'INCORRECT_WORD_USAGE', "Hint", word)
                )

    return prepositions_errors

@logged(logger)
def _check_artickes(words, sentence, sent_decontracted, tagged_words):
    """ Check articles usage before names of the Countries """

    @logged(logger)
    def get_article_token_id(sentence_substr: str):
        span_generator = WhitespaceTokenizer().span_tokenize(sentence_substr)
        spans = [span for span in span_generator]
        return spans.index(spans[-1])

    doc = spacy_tokenizer(sentence)
    errs = []
    error_params = []
    for ent in doc.ents:
        if ent.label_ in ['GPE']:
            # get NP containing GPE NER (country, state, city)
            NPs = [np.text.lower().split() for np in doc.noun_chunks if
                   (np.text.find(ent.text) > -1 or ent.text.find(np.text) > -1) and np.text != 'i']
            NPs_list = list(itertools.chain.from_iterable(NPs))
            chunk = ' '.join(NPs_list)
            GPE_subphrase = set(ent.text.lower().split()).union(set(NPs_list))

            # Check NP phrase with GPE entity with country names in provided list

            if chunk.find('the') > -1:
                for country in NO_COUNTRIES:
                    diff = GPE_subphrase.difference(set(country.split()))
                    if diff == {'the'} or diff == {'a'}:
                        article_token_id = get_article_token_id(sentence[:ent.start_char])
                        error_params = [list(diff)[0], words, article_token_id, sent_decontracted, sentence,
                                        "HINTS",
                                        f"There should not be an article in the position: {sentence.find(ent.text)}",
                                        [''], "ARTICLES_BEFORE_COUNTRIES", 'Hint', list(diff)[0]]

            elif chunk.find('the') == -1:
                for country in THE_COUNTRIES:
                    if set(country.split()).difference(GPE_subphrase) == {'the'}:
                        error_params = ['', words, ent.start_char, sent_decontracted, sentence,
                                        "HINTS", f"Article 'the' is needed in the position: {sentence.find(ent.text)}",
                                        ['the'], "ARTICLES_BEFORE_COUNTRIES", 'Hint', ""]

            if error_params:
                errs.append(
                    _create_single_error(*error_params)
                )

    return errs

@logged(logger)
def _replace_token_in_sentence(words: List[str], id_of_token_to_be_replaced: int, token_for_replacement: str):
    last = [] if id_of_token_to_be_replaced == len(words) - 1 else words[id_of_token_to_be_replaced + 1:]
    return ' '.join(words[:id_of_token_to_be_replaced] + [token_for_replacement] + last)

@logged(logger)
def _static_rules(sentence: str) -> Optional[List[SingleResult]]:
    """
    :param sentence: sentence from users text
    :return: list of errors found with heuristic methods
    """
    errors_caught_with_static_rules = []
    words = word_tokenize(sentence)
    if "peoples" in words:
        errors_caught_with_static_rules.append(
            _create_single_error(words, words, words.index("peoples"), sentence, sentence,
                                "HINTS",
                                "Word 'Peoples' may be used in the context of different nations. In other cases 'People' should singular.",
                                 ['people'], "PEOPLE_PEOPLES", 'Hint')
        )

    return errors_caught_with_static_rules

@logged(logger)
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

@logged(logger)
def mistake_already_caught(results: List[SingleResultFromLanguageTool],
                           current_mistake: SingleResultFromLanguageTool) -> bool:
    """check for duplicate errors found within LanguageTool (with and without disabled rules)."""
    for o in results:
        if current_mistake.offset == o.offset and current_mistake.ruleId == o.ruleId:
            return True
    return False

@logged(logger)
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

@logged(logger)
def get_errors_with_replacements(errs_matches):
    # get errors from LT which have replacements
    return [err for err in errs_matches['matches'] if err['replacements'] != []
            and err['rule']['category']['id'] not in REDUNDANT_CATEGORIES]

@logged(logger)
def check_for_unsuitable_replacements(sr):
    """remove the changes replacements by LanguageTool if they are completely unsuitable for the fixing."""
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

@logged(logger)
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

@logged(logger)
def reorganize_categories(categories: dict) -> dict:
    """
    :param categories: dictionary with Categories analytics
    :return: found mistakes ruleIds are being counted
    """
    for category, category_statistic in categories.items():
        rules = category_statistic['ruleIds']
        category_statistic['ruleIds'] = list()
        rule_names = [list(rule.keys())[0] for rule in rules]
        counter = Counter(rule_names)
        for r, v in counter.items():
            rule_statistics = RuleIdNumber(ruleId=r, number_of_mistakes=v)
            category_statistic.append(rule_statistics.dict())

    return categories

@logged(logger)
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

@logged(logger)
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
