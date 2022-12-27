import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.model import CAT_FEATURES, inference
from starter.ml.data import process_data


app = FastAPI()

class Person(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13, alias="education-num")
    marital_status: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


@app.get("/")
def welcome():
    return {"welcome": "Welcome to this page."}


@app.post("/infer/")
def model_inference(data: Person):
    data = pd.DataFrame.from_dict([data.dict(by_alias=True)])
    model = joblib.load("model/model.pkl")
    encoder = joblib.load("model/encoder.pkl")

    X, _, _, _ = process_data(
        data,
        categorical_features=CAT_FEATURES,
        training=False,
        encoder=encoder
    )
    preds = inference(model, X)
    if preds[0] == 0:
        salary = "<=50K"
    else:
        salary = ">50K"
    return {"salary": salary}
