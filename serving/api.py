from fastapi import FastAPI, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pickle import *
from utils import *
from uuid import uuid4

from pydantic import BaseModel

current_prediction = {}

app = FastAPI()

@app.get("/")
def read_root():
    return {"Routes: /predict, /feedback"}

@app.post("/predict")
def predict(data: UploadFile = File(...)):
    #vector= algo de greg(data)
    vector=None
    #prediction=model.predict(vector)
    prediction="Chien"
    id=uuid4()
    json_compatible_item_data = jsonable_encoder({"id": id, "prediction": prediction})

    current_prediction[id]=(prediction,vector)

    return JSONResponse(content=json_compatible_item_data)

class FeedBackModel(BaseModel):
    id_image: str
    data: bool

@app.post("/feedback")
def feedback(feedback: FeedBackModel):
    print(f"Feedback received for image {feedback.id_image}: {feedback.data}")
    SaveFeedBackData(current_prediction[feedback.id_image][1], feedback.data)
    check_for_new_pickle()

if __name__ == "__main__":
    doTraining()
    model = open_pickle()
