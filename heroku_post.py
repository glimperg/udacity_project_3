import json
import requests

data = {
    "age": 30,
    "workclass": "Federal-gov",
    "fnlgt": 59951,
    "education": "Some-college",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Adm-clerical",
    "relationship": "Own-child",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

r = requests.post("https://census-bureau-classification.herokuapp.com/infer/", data=json.dumps(data))
print("STATUS CODE:", r.status_code)
print("RESPONSE:", r.json())
