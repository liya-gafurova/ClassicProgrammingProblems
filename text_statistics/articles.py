# NP chuncs
import copy
import itertools
from typing import Optional, List, Dict, Any

import spacy
import torch
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, FillMaskPipeline

nlp = spacy.load("en_core_web_md")
s = """I have been working in United Kingdom. I like visiting the Spain, but i have never been in the Russia. But was in the Czeck Republic. the I was living in the Bahamas. But most of the time I liked living in United States of America. And in the Central African Republic and in the Armenia I was working. """
doc = nlp(s)

with open("/home/lia/PycharmProjects/grammar-checker/reviews/countries.txt") as f:
    _COUNTRIES = f.read().splitlines()

with open("/home/lia/PycharmProjects/grammar-checker/reviews/THE_countries2.txt") as f2:
    THE_COUNTRIES = f2.read().splitlines()





class ArticleMistake(Exception):
    def __init__(self, message: str = "Article 'the' is needed in the position: {}", position: int = 0):
        self.message = message.format(position)
        super().__init__(self.message)


for ent in doc.ents:
    if ent.label_ in ['GPE']:
        sent_doc = nlp(ent.sent.text)

        NPs = [np.text.lower().split() for np in sent_doc.noun_chunks if
               (np.text.find(ent.text) > -1 or ent.text.find(np.text) > -1) and np.text != 'i']
        NPs_list = list(itertools.chain.from_iterable(NPs))
        chunk = ' '.join(NPs_list)
        print(chunk)

        GPE_subphrase = set(ent.text.lower().split()).union(set(NPs_list))
        print(f"GPE set of tokens = {GPE_subphrase}")

        try:
            if chunk.find('the') > -1:
                for country in _COUNTRIES:
                    diff = GPE_subphrase.difference(set(country.split()))
                    if diff == {'the'} or diff == {'a'}:
                        raise ArticleMistake(message="There should not be an article in the position: {}", position=s.find(ent.text))
            elif chunk.find('the') == -1:
                for country in THE_COUNTRIES:
                    if set(country.split()).difference(GPE_subphrase) == {'the'}:
                        raise ArticleMistake(message="Article 'the' is needed in the position: {}", position=s.find(ent.text))
        except ArticleMistake as e:
            print(e.message)

    print("==============================")
