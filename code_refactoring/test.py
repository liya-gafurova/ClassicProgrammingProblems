import re

import requests


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

try:
    response = requests.post(
        url=f'http://0.0.0.0:8011/v2/check',
        params={"text": "how is you"}
    )
except Exception  as e:
    print(e.__class__.__name__)

ss = ["here is no msk", 'here is a mask']
v = None
for s in ss:
    if s.find('[MASK]') >-1:
        v = s
        break

r = next((s for s in ss if s.find('[MASK]') > -1), None)
print(v)
print(r)

class SimpleIterator:
    def __iter__(self):
        return self

    def __init__(self, limit):
        self.limit = limit
        self.counter = 0

    def __next__(self):
        if self.counter < self.limit:
            self.counter += 1
            return 1
        else:
            raise StopIteration

s_iter2 = SimpleIterator(5)
for i in s_iter2:
    print(i)
