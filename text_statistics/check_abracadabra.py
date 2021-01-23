import re
from datetime import datetime
import string
import spacy

start = datetime.now()
spacy_tokenizer = spacy.load('en_core_web_md')

print(f"TIME : {datetime.now() - start}")
abra = """
;oahiewg vwoeighro[wiher oihwgrfhpi[ergfij[ pij[ergjp[io jpi[ergjio oijerg084w5 wrthg452y
"""

text_with_errors = """
how r u? Whats goin on
"""

text_ok = "Hello world"

good_big_text = """My name is Susan. I'm forteen and I life in Germany. My hobbys are go to discos, sometimes I hear music in the radio. In the July I go bathing in a lake. I haven't any brothers or sisters. We take busses to scool. I visit year 9 at my school. My birthday is on Friday. I hope I will become a new guitar."""

abra_100 = """Duty rye shortish orthjoi onjitfgji nojip jprtgjip pjitrgpji jerghoi lmkrtghjo mlkerjgoi mrtghjoi nlkrtghoit nlkrthojitgrmnlk oij mlkrtghoi mlk ojirt ij 98 kj 98 jk 98 k -0lksahfa kjergjp orrgfh nkergh knergoihv n oerg oih iug r3othi pu iugsrfjlk 09u kjbegooii hiu g ljergoi iguefiuglkj po iugh oiherfbjk poegrkj neriu lker oijerh oijergopk jh ijrpogjb iu poj iug po iug poi iuig pi uygg o y ;oihiyg iulg e45t89y ui 98t4kjb3 98 kjn3 \98 ni  olhirtgkn jitgn oirwgtnk oir oj oih h ohi ohi ohi oiheert90nj45  98 98 98 nk 78 nj 67 b 6 h 67 hj jk 98j jj 78 nj lo k 78 hjbkisf 78 jkn o87"""
good = """I had a great weekend, at Friday I went to Oxford Street and did some shopping with my friends. On Saturday I went to Liverpool street, there are a cool market there. I buyed some clothes. I had a rest on Sunday, I was so tired.
 
You have to come and visit me soon, you will loving London. I will show you all of the sights, Big Ben, the London Eye and best of all, the pubs. There is a lot of great pubs here in London.
 
Anyway, I have to go now, write back soon.
Dear Jane, 
 
I was delighted to read you're letter last week. Its always a pleasure to recieve the latest news and to here that you and your family had a great summer.
 
"""

half_abra = """I had a great gfnbsfgnsrf, at Friday I went to Oxford Street and did some shopping with my friends. On Saturday I went to Liverpool fgnsfgns, there are a dfbdgbdbg fgnfhn there. I buyed some clothes. I had a rest on Sunday, I was so tired.
 
You dbdbdb to fsgnszfgn and visit dfbdgb soon, you dbsFvs loving fgnsf n. I will show you all of the sights, Big Ben, fsgnsf sfvzdfb Eye and dfbdfbdfb of all, the pubs. Fg szfg sfgn is a lot of dfbgdbgdf pubs here in London.
 
edbfdgb, I have to go now, write sfgnsfgn soon.
Dear Jane, 
 
I was dfb to fdbdfbdf you're letter dgnbdsgbdf dgbdgb. Its always a pleasure to recieve the latest news zdfb dzfbzdfb here dzfbzdfb you and your sfgnsfgn had a great summer.
 """

print('==================================================================')



def _is_random_set_of_characters(text):
    # delete punctuation
    text = re.sub( f"[{string.punctuation}]", ' ', text)

    # tokenize with spacy and find out of vocabulary tokens
    parsed_document = spacy_tokenizer(text)
    all_tokens = [(t, t.is_oov, t.prob) for t in parsed_document]
    out_of_vocabulary_tokens = [t for t in all_tokens if t[1]]

    # find out of vocabulary tokens quantity
    oov_percentage = len(out_of_vocabulary_tokens) / len(parsed_document)

    return oov_percentage > 0.4


start = datetime.now()

_is_random_set_of_characters(text_ok)

_is_random_set_of_characters(text_with_errors)

_is_random_set_of_characters('I am sflmgmavknsaccasas dog cat bird bulbasaur')

_is_random_set_of_characters(abra)

_is_random_set_of_characters(good_big_text)

_is_random_set_of_characters(abra_100)

_is_random_set_of_characters(good)

_is_random_set_of_characters(half_abra)


print(f'Total time: {datetime.now() - start}')


print([(tok.text, tok.pos_, tok.tag_) for tok in spacy_tokenizer(good_big_text)])



