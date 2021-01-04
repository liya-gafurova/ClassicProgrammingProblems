import  peewee
from peewee import Model, BooleanField, TextField, SqliteDatabase

import hashlib
import json
from datetime import datetime



CHECKING_RESULTS_DICT = {
    "language_tool_result" : [],
    "additional_checking": {
        "aux_verbs": [],
        "verbs":[],
        "incorrect_verb_usage": []
    }
}


database = SqliteDatabase("db/test4.db")


def _create_hash_for_text(text):
    current_time =  datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    hash_object = hashlib.md5(text.encode()+current_time.encode())
    return hash_object.hexdigest()


class Texts(Model):
    id = TextField(primary_key=True, null=False)
    text = TextField(null=False)
    checking_result = TextField(null=True)
    is_text_checked = BooleanField(default=False)
    is_text_sent_to_client = BooleanField(default=False)

    class Meta:
        database = database


def create_table():
    try:
        Texts.create_table(safe=True)
    except peewee.OperationalError:
        print("Texts table already exists!")

def drop_table():
    Texts.drop_table(safe=True)


def insert_text(text):
    Texts.insert(
        id = _create_hash_for_text(text),
        text = text
    ).execute()


def update_ckecked_status(hash):
    processed_text: Texts = Texts.get(id = hash)
    processed_text.checking_result = json.dumps(CHECKING_RESULTS_DICT)
    processed_text.is_text_checked = True
    processed_text.save()


def update_sent_to_client_status(hash: str):
    text: Texts = Texts.get(id = hash)
    text.is_text_sent_to_client = True
    text.save()


def delete_texts_which_are_sent_to_client():
    q = Texts.delete().where(Texts.is_text_sent_to_client == True)
    q.execute()


def show_all_texts_in_db():
    for text in Texts.select():
        print(f"{text.id} -- {text.text} -- {text.checking_result} -- {text.is_text_checked} -- {text.is_text_sent_to_client} ")


