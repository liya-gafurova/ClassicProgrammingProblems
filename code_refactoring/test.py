import re

def q(phrase):
    phrase = re.sub(r"[’'`]ll", " will", phrase)
    phrase = re.sub(r"[’'`]t", " not", phrase)
    phrase = re.sub(r"[’'`]ve", " have", phrase)
    return phrase

print(q("""I`ve in lo35tkbj with coco"""))
print(q("""I've in lo35tkbj with coco"""))
print(q("""I’ve in lo35tkbj with coco"""))

try:
    print('hello'.values())
except Exception as e:
    print(e.__class__.__name__ )