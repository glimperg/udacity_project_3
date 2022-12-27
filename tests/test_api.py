"""
Tests for API in main.py.
"""
import json
from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


def test_prediction_get():
    """
    Test if the welcome GET request is completed successfully.
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"welcome": "Welcome to this page."}


def test_prediction_less_than_50k():
    """
    Test POST request by giving an example from the dataset which
    should return a prediction of <=50K.
    """
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
    r = client.post("/infer/", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == {"salary": "<=50K"}


def test_prediction_more_than_50k():
    """
    Test POST request by giving an example from the dataset which
    should return a prediction of >50K.
    """
    data = {
        "age": 47,
        "workclass": "Private",
        "fnlgt": 51835,
        "education": "Prof-school",
        "education-num": 15,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 1902,
        "hours-per-week": 60,
        "native-country": "Honduras"
    }
    r = client.post("/infer/", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == {"salary": ">50K"}