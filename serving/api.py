import fastapi
import pickle

app = fastapi.FastAPI()

@app.get("/")
def read_root():
    return {"Routes: /predict, /feedback"}

@app.post("/predict")
def predict(data: fastapi.UploadFile = fastapi.File(...)):
    return model.predict(data)

@app.post("/feedback")
def feedback(data: bool):
    return data

def open_pickle():
    with open('../artifacts/best_model.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    model = open_pickle()