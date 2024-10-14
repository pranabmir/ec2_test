import warnings
warnings.filterwarnings('ignore')
from scripts.data_model import NLPDataInput,NLPDataOutput,ImageDataInput,ImageDataOutput
from fastapi import FastAPI
from fastapi import Request
import uvicorn
from scripts import s3
import os
import torch
from transformers import pipeline
import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#ml download code
mod_name = 'tiny_bert_sentiment_analysis/'
local_path = 'ml_model/'+mod_name
if not os.path.isdir(local_path):
    s3.download_dir(local_path,mod_name)
sentiment_model = pipeline('text-classification',model = local_path,device = device)

mod_name = 'tiny_bert_disaster_tweet/'
local_path = 'ml_model/'+mod_name
if not os.path.isdir(local_path):
    s3.download_dir(local_path,mod_name)
disaster_model = pipeline('text-classification',model= local_path,device = device)

#image model, commenting it as downloading take time due to large size
# from transformers import AutoImageProcessor
# model_ckpt = 'google/vit-base-patch16-224-in21k'
# image_processor = AutoImageProcessor.from_pretrained(model_ckpt,use_fast = True)
# mod_name = 'vit_human_pose_classification'
# local_path = 'ml_model/'+mod_name
# if not os.path.isdir(local_path):
#     s3.downoad_dir(local_path,mod_name)
# pose_model = pipeline('image-classification',model = local_path,device=device,image_processor= image_processor)

#ml download code ends

app = FastAPI()

@app.get('/')
def read_root():
    return 'server working'


@app.post("/api/v1/sentiment_analysis")
def sentiment_analysis(data:NLPDataInput):
    start = time.time()
    output = sentiment_model(data.text)
    end = time.time()
    prediction_time = end-start
    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]
    output = NLPDataOutput(mod_name='tiny_bert_sentiment_analysis',
                           text = data.text,
                           targets = labels,
                           scores =scores,
                           prediction_time = prediction_time)
    return output

@app.post("/api/v1/disaster_classifier")
def disaster_classifier(data: NLPDataInput):
    start= time.time()
    output = disaster_model(data.text)
    end = time.time()
    prediction_time  = end-start
    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]
    output = NLPDataOutput(mod_name = 'tiny_bert_disaster_tweet',
                           text = data.text,
                           targets = labels,
                           scores = scores,
                           prediction_time = prediction_time)
    return output

@app.post("/api/v1/pose_classifier")
def pose_classifier(data: ImageDataInput):
    return data






if __name__=='__main__':
    uvicorn.run(app = app)