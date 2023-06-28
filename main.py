from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os, platform



app = FastAPI()
model = None
# Define the input data model
class InputData(BaseModel):
    var1: float
    var2: float
    var3: float
    var4: float
    var5: float

# Define the POST endpoint to accept the model pickel location 
@app.post("/upload")
async def upload(input_path: str):
    global model
    print(f'Model Path is - {input_path}')
    # Load the pickle file containing the trained ML model test/drive/model.pkl
    # with open(f'{input_path}', 'rb') as f:
        # model = pickle.load(f)

# Define the GET endpoint to provide the swagger URL for the uploaded model
# http://localhost:8000/docs
# http://localhost:8000/openapi.json
@app.get("/tryout")
def tryout():
    swagger_url = "" 
    if platform.system() == "Windows":
        print(platform.uname().node)
        # swagger_url = platform.uname().node
    else:
        print(os.environ)   # doesnt work on windows
        swagger_url = os.uname()[1]
        # swagger_url = "Madan Madi"
    return swagger_url

# Define the POST endpoint to accept the input data and make predictions
@app.post("/predict")
async def predict(input_data: InputData):
    # Convert the input data into a numpy array
    input_array = [[input_data.var1, input_data.var2, input_data.var3, input_data.var4, input_data.var5]]
    # Make predictions using the loaded model
    prediction = model.predict(input_array)
    # Return the prediction as a dictionary
    return {"prediction": prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
