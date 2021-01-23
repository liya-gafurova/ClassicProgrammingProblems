import re

import textstat

test_data = (
    "Playing games has always been thought to be important to "
    "the development of well-balanced and creative children; "
    "however, what part, if any, they should play in the lives "
    "of adults has never been researched that deeply. I believe "
    "that playing games is every bit as important for adults "
    "as for children. Not only is taking time out to play games "
    "with our children and other adults valuable to building "
    "interpersonal relationships but is also a wonderful way "
    "to release built up tension."
)

sent = "My kids are back in school to. There is a small bath in the forest. But i think it was quite wonderful.  I find their are less things to worry about now that the kids are at school all day."

print(f"textstat.flesch_reading_ease= {textstat.flesch_reading_ease(test_data)}")
print(f"textstat.flesch_kincaid_grade= {textstat.flesch_kincaid_grade(test_data)}")
print(f"textstat.smog_index= {textstat.smog_index(test_data)}") # 30 sentences
print(f"textstat.coleman_liau_index= {textstat.coleman_liau_index(test_data)}")
print(f"textstat.automated_readability_index= {textstat.automated_readability_index(test_data)}")
print(f"textstat.linsear_write_formula= {textstat.linsear_write_formula(test_data)}")
print(f"textstat.gunning_fog= {textstat.gunning_fog(test_data)}")

print(f"textstat.dale_chall_readability_score= {textstat.dale_chall_readability_score(test_data)}")

print(f"textstat.difficult_words= {textstat.difficult_words(test_data)}")


print(f"textstat.text_standard= {textstat.text_standard(test_data)}") ## суммирует все вышеперечисленные метрики
print(f"textstat.text_standard= {textstat.text_standard(sent)}") ## суммирует все вышеперечисленные метрики
print(f"textstat.smog_index= {textstat.smog_index(sent)}") # 30 sentences
def _get_avg(readability_desc:str):
    ''' return "{}{} and {}{} grade".format(
                lower_score, get_grade_suffix(lower_score),
                upper_score, get_grade_suffix(upper_score)
            )'''
    numbers = re.findall(r'\d+', readability_desc)
    return sum(list(map(int,numbers))) / len(numbers)

readability_desc: str = textstat.text_standard(sent)
score = _get_avg(readability_desc)
abracadabra = """
ergaer ergeqrghoierr setrpgjoerg ergohiqerig 3089y4r aer98t h oitrh'poer 98 iubaer988 iuawerpui
"""

print(f" textstat.lexicon_count(abracadabra) = {textstat.lexicon_count(abracadabra)}")