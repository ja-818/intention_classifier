import torch
import torch.nn.functional as F
from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from pydantic import BaseModel

###### Models #######

# Load the tokenizer and model
game_edit_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
game_edit_model = AutoModelForSequenceClassification.from_pretrained("AppOnboard/game_edit_classifier")
game_edit_id2label = {0: "asset_behavior_manipulation", 1: "scene_management", 2: "entity_creation"}

# Load the tokenizer and model
intention_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
intention_model = AutoModelForSequenceClassification.from_pretrained("AppOnboard/intention_classifier")
intention_id2label =  {0: "game_edit", 1: "ask_documentation", 2: "other"}

def make_prediction(user_input:str, model:AutoModelForSequenceClassification, tokenizer:AutoTokenizer, id2label:dict):
    inputs = tokenizer(user_input, padding=True, truncation=True, return_tensors="pt")

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Apply the Softmax function to the logits to get probabilities
    probabilities = F.softmax(outputs.logits[0], dim=-1)
    # Get the predicted class ID
    predicted_class_idx = torch.argmax(probabilities).item()
    # You can then map this index to the actual class label using the id2label dictionary, if available
    label = model.config.id2label[predicted_class_idx]

    return dict(prediction=label, probability=probabilities.tolist()[predicted_class_idx])

####### API ########
class UserInput(BaseModel):
    user_input: str = "make an enemy that shoots flowers"

app = FastAPI()
@app.post("/intention_classifier")
def predict(user_input:UserInput):
    prediction_dict = make_prediction(user_input.user_input, intention_model, intention_tokenizer, intention_id2label)
    return prediction_dict

@app.post("/game_edit_classifier")
def predict(user_input:UserInput):
    prediction_dict = make_prediction(user_input.user_input, game_edit_model, game_edit_tokenizer, game_edit_id2label)
    return prediction_dict