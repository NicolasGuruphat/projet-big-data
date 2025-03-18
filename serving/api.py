from fastapi import FastAPI, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pickle import *

from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def read_root():
    return {"Routes: /predict, /feedback"}

@app.get("/classes")
def get_classes():
    return {"classes": ["dog", "cat"]}

@app.post("/predict")
def predict(data: UploadFile = File(...)):
    #json_compatible_item_data = jsonable_encoder(model.predict(data))
    # TODO: stocker l'image vectorisée, assigner un id et faire la prédiction
    json_compatible_item_data = jsonable_encoder({"id": "1234", "prediction": "dog"})

    return JSONResponse(content=json_compatible_item_data)

class FeedBackModel(BaseModel):
    id_image: str
    data: str

@app.post("/feedback")
def feedback(feedback: FeedBackModel):
    print(f"Feedback received for image {feedback.id_image}: {feedback.data}")
    # TODO: save feedback to database/file
    return feedback

def open_pickle():
    with open('../artifacts/best_model.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    model = open_pickle()