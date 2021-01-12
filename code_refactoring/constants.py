import json

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