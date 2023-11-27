import sys 
sys.path.append("./model")
sys.path.append("./src")
import pickle
import os
import uvicorn
from fastapi.logger import logger
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import torch
from Inference import Inference
from convnext import ConvNeXt
from contextlib import asynccontextmanager

from pyngrok import ngrok




def init_webhooks(base_url):
    # Update inbound traffic via APIs to use the public-facing ngrok URL
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):

    model = ConvNeXt(depths=[3,3,27,3], dims=[32,64,128,256],in_chans=1,
                         num_classes=1,drop_path_rate=0,
                         dim_mul=2,dwconv_kernel_size=3,dwconv_padding=1,
                         downsample_stem=2)
    checkpoint = torch.load('./checkpoints/checkpoints.pth')
    model.load_state_dict(
        checkpoint['model_state_dict']
    )

    model.to('cuda')
    model.eval() 
    print("=> Model loaded successfully")
    app.package = {
        'model' : model
    }
    
    app.inference = Inference(app.package)
    print("=> Server listening on PORT")

    yield
    app.inference = None

app = FastAPI(title="Sample ML App using FastAPI", version="0.0.1",lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.get('/')
def root_dir():
    return {
        'Hello world': 'Welcome to FastAPI Tutorial to make ML powered REST APIs'
    }


@app.post('/predict')
def predict(request: Request, ppg):

    print(Request)
    print(ppg)
    response = app.inference.predict(ppg)

    return {
        'error': False, 
        'prediction': response['prediction']
    }

# GET Method to Information About the API 

@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available()
    }


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)