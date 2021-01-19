import ast
import json
import requests
from json import dumps
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from celery import  Celery
load_dotenv(verbose=True)

app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379',
    CELERY_RESULT_BACKEND='redis://localhost:6379'
)

def make_celery(app):
    celery = Celery(app.import_name, backend=app.config['CELERY_RESULT_BACKEND'],
                    broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)
    TaskBase = celery.Task
    class ContextTask(TaskBase):
        abstract = True
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)
    celery.Task = ContextTask
    return celery

celery_app = make_celery(app)
celery_app.autodiscover_tasks()


@celery_app.task
def check_text(text:str = "hello? how is you?"):

    # Так делать неправильно, так как:
    # Запрос рест не должен висеть долго - самый максимум это 5 секунд
    #  в противном случае надо задуматься об очереди задач
    try:
        res = requests.post('http://localhost:8000/grammar/check_grammar?enable_disabled_rules_and_categories=true',
                            data=json.dumps({"source_text": text}),
                            )
        result = res.json()
    except Exception as e:
        result = "Could not process it"

    # TODO load to database
    print(result)



@app.route("/api/v1/texts", methods=["POST"])
def save_text():
    body = ast.literal_eval(dumps(request.get_json()))
    source_text = body["text"]

    check_text.delay(source_text)

    return jsonify({"Status": "The text is sent for processing"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)