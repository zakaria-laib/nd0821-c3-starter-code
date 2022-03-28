import requests

record = {
    "age": 35,
    "workclass": "Private",
    "fnlgt": 181371,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Married-spouse-absent",
    "occupation": "Exec-managerial",
    "relationship": "Wife",
    "race": "White",
    "sex": "Female",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States"
}

response = requests.post(
    "https://mlops-app-zak.herokuapp.com/prediction/",
    json=record)
print(response.status_code)
print(response.json())
