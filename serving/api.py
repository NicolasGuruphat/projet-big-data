from fastapi import FastAPI, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pickle import *

app = FastAPI()

@app.get("/")
def read_root():
    return {"Routes: /predict, /feedback"}

@app.post("/predict")
def predict(data: UploadFile = File(...)):
    #json_compatible_item_data = jsonable_encoder(model.predict(data))
    json_compatible_item_data = jsonable_encoder("Chien")
    return JSONResponse(content=json_compatible_item_data)

@app.post("/feedback")
def feedback(data: bool):
    return data

def open_pickle():
    with open('../artifacts/best_model.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    model = open_pickle()