from fastapi import FastAPI, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pickle import *
from utils import *
from uuid import uuid4

from pydantic import BaseModel

from PIL import Image
from torchvision import transforms

current_prediction = {}
global model

app = FastAPI()

@app.get("/")
def read_root():
    return {"Routes: /predict, /feedback"}

@app.get("/classes")
def get_classes():
    return {"classes": ["dog", "cat"]}

@app.post("/predict")
def predict(data: UploadFile = File(...)):
    image_pillow = Image.frombytes(data.file)
    data_transform = transforms.Compose([
        # Resize the images to 64x64
        transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
        # Flip the images randomly on the horizontal
        # Turn the image into a torch.Tensor
        transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
    ])
    
    transformed_image = data_transform(image_pillow)

    prediction=model.predict(transformed_image)
    id=uuid4()
    json_compatible_item_data = jsonable_encoder({"id": id, "prediction": prediction})

    current_prediction[id] = (prediction, transformed_image)

    return JSONResponse(content=json_compatible_item_data)

class FeedBackModel(BaseModel):
    id_image: str
    data: str

@app.post("/feedback")
def feedback(feedback: FeedBackModel):
    print(f"Feedback received for image {feedback.id_image}: {feedback.data}")
    SaveFeedBackData(current_prediction[feedback.id_image][1], feedback.data)
    check_for_new_pickle()

@app.post("/retrain")
def retrain():
    print("Début de l'entrainement")
    doTraining()
    model = open_pickle()
    return {"message": "Model retrained"}
