from fastapi import FastAPI, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pickle import *
from utils import *
from uuid import uuid4
import torch

from pydantic import BaseModel

from PIL import Image
from torchvision import transforms

import io

global current_prediction
global model

current_prediction = {}

app = FastAPI()

@app.get("/")
def read_root():
    return {"Routes: /predict, /feedback"}

@app.get("/classes")
def get_classes():
    return {"classes": ["dog", "cat"]}

@app.post("/predict")
async def predict(data: UploadFile = File(...)):
    model = open_pickle()
    if model is None:
        return JSONResponse(content={"error": "Model not found"}, status_code=404)

    image_bytes = await data.read()
    image_pillow = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    data_transform = transforms.Compose([
        # Resize the images to 64x64
        transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
        # Flip the images randomly on the horizontal
        # Turn the image into a torch.Tensor
        transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
    ])
    
    transformed_image = data_transform(image_pillow)

    flattened_image_tensor = torch.flatten(transformed_image)
    print(flattened_image_tensor)
    print(flattened_image_tensor.shape)
    prediction=model.predict([flattened_image_tensor])

    id=uuid4()
    json_compatible_item_data = jsonable_encoder({"id": id, "prediction": 'dog' if prediction[0] == 0 else 'cat'})

    current_prediction[str(id)] = (prediction, transformed_image)

    return JSONResponse(content=json_compatible_item_data)

class FeedBackModel(BaseModel):
    id_image: str
    data: str

@app.post("/feedback")
def feedback(feedback: FeedBackModel):
    print(f"Feedback received for image {feedback.id_image}: {feedback.data}")
    # On stocke dans le fichier de feedback, le vecteur d'image, la classe prédite initialement et la classe de feedback
    SaveFeedBackData(current_prediction[feedback.id_image][1], current_prediction[feedback.id_image][0], 0.0 if feedback.data == 'dog' else 1.0)
    check_for_new_pickle()

@app.post("/retrain")
def retrain():
    print("Début de l'entrainement")
    doTraining()
    model = open_pickle()
    return {"message": "Model retrained"}
