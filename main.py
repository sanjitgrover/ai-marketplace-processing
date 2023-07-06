from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, create_model
import pickle
import os, platform
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from array import array



app = FastAPI()
model = None

def convert_list_to_dict(lst):
   res_dict = {}
   for i in range(0, len(lst), 2):
       res_dict[lst[i]] = lst[i + 1]
   return res_dict

modelparams = {
    'va1': (float, 0.0),
    'va2': (float, 0.0),
    'va3': (float, 0.0),
    'va4': (float, 0.0),
    'va5': (float, 0.0),
    'va6': (float, 0.0),
    'va7': (float, 0.0),
    'va8': (float, 0.0)
} 

xyz = create_model('xyz', **modelparams)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define the input data model
class InputData(BaseModel):
    var1: float
    var2: float
    var3: float
    var4: float
    var5: float
    var6: float
    var7: float
    var8: float
    
class modelData(BaseModel):
    modellist: list

# Define the POST endpoint to accept data model parameters
@app.post("/modelparams")
async def modelparams(model_keys_list: modelData):
    print(model_keys_list)
    print(dict(model_keys_list))
    # print(convert_list_to_dict(model_keys_list))
    global xyz
    # xyz = create_model('xyz', model_keys)
    # print(xyz)
    return None

# Define the POST endpoint to accept the model pickel location 
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global model
    try:
        # contents = file.file.read()
        # with open(file.filename, 'wb') as f:
            # f.write(contents)
    # print(f'Model Path is - {input_path}')
    # Load the pickle file containing the trained ML model test/drive/model.pkl
        with open(file.filename, 'rb') as f:
            model = pickle.load(f)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}




# Define the POST endpoint to accept the input data and make predictions
@app.post("/predict")
async def predict(input_data: xyz):
    
    # Convert the input data into a numpy array
    # input_array = [[input_data.var1, input_data.var2, input_data.var3, input_data.var4, input_data.var5, input_data.var6, input_data.var7, input_data.var8]]
    # model_array=["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age"]
    model_array = ["va1", "va2", "va3", "va4", "va5", "va6","va7", "va8"]
    input_array = [[]]
    print(type(input_array))
    for val in model_array:
        print(getattr(input_data, val, None))
        input_array[0].append(getattr(input_data, val, None))
    print(f'Input array is {input_array}')
    # Make predictions using the loaded model
    prediction = model.predict(input_array)
    # Return the prediction as a dictionary
    return {"prediction": prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
