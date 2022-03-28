from fastapi.testclient import TestClient
from api_server import app

client = TestClient(app)


def test_get_path():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Greetings!"}


def test_prediction_less_50k():
    record = {
        "age": 49,
        "workclass": "Private",
        "fnlgt": 160187,
        "education": "9th",
        "education-num": 5,
        "marital-status": "Married-spouse-absent",
        "occupation": "Other-service",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 16,
        "native-country": "Jamaica"
    }

    response = client.post("/prediction/", json=record)
    assert response.status_code == 200
    assert response.json() == {"salary": "<=50K"}


def test_prediction_more_50k():
    record = {
        "age": 40,
        "workclass": "Private",
        "fnlgt": 193524,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    }

    response = client.post("/prediction/", json=record)
    assert response.status_code == 200
    assert response.json() == {"salary": ">50K"}