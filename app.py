"""The API that serves my model"""

# imports 
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, conlist
from typing import List
import joblib 
from brock import CATEGORIES
import numpy as np

# load the model
model = joblib.load("model/model-final.joblib")

APP_DESC = """
##  Goals

- POC to highlight how a relatively flexible pipeline to train and serve ML models 
- Flag a model that can associate one or more tags to a question with a score

### Longer term:

- all sorts of ML models can be leveraged with the tensorflow deep learning engine included
- think about real time vs batch scoring of data

### Use cases
 - Real time could include the removal of toxic questions submitted to PQCast, or suggest a set of answers to common questions.
 - In batch, tag questions and use for thought leadership and internal strategic planning (with clients) based on trends in "needs or intents"

"""

# create the app
app = FastAPI(title="POC ML API with Scikit-learn for Text Classification", 
              description=APP_DESC, 
              version="0.1")


# create the class for the post param
class Message(BaseModel):
    text: str
    class Config:
        schema_extra = {
        "example": {
            "text": "I like computers and internet things"
        }
    }


@app.post('/predict', tags=["predictions"])
async def get_prediction(data: Message):
    """
    For a given question text, predict if the text belongs to a target category.  
    
     - Each question can be tagged with 1 or more categories in the taxonomy.  For example, some may have two tags.
     - Each <b>category</b> is the score of being tagged as that category in our taxonomy.  Scores go from 0 to 1, with 1 representing absolute confidence.
     - Practical usage: only associate a tag based on probability of having that tag (e.g. > .9, or some other score level)

    __Simply, ideal tag scores for a question tagged with `facts` and `admission` would be 1.0 for both of those labels.__
    """
    data = data.dict()
    question = data['text']
    pred = model.predict([question])
    int_p = int(np.round(pred, 4))
    confidence = list(model.predict_proba([question]))
    maxc = int(np.argmax(confidence[0]))
    float_c = float(np.round(confidence[0][maxc], 3))
    # pred_scores = model.predict_proba([question])
    # pred_scores = pred_scores[0]
    # pred_scores = np.array(pred_scores)
    # scores = [{CATEGORIES[k]:float(v) for k, v in enumerate(pred_scores)}]
    resp = {'pred': int_p,
            'label': str(CATEGORIES[int_p]),
            'score': float_c}
    # resp = {0:1, 1:2}
    return resp

