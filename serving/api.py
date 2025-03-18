import fastapi
import pickle

app = fastapi.FastAPI()

@app.get("/")
def read_root():
    return {"Routes: /predict, /feedback"}

@app.post("/predict")
def predict(data: fastapi.UploadFile = fastapi.File(...)):
    return {"filename": data.filename}

@app.post("/feedback")
def feedback(data: bool):
    return data

if __name__ == "__main__":
    open_pickle('./artifacts/model.pkl')

def open_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
