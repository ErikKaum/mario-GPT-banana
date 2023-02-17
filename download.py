# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from transformers import pipeline
from mario_gpt.lm import MarioLM

def download_model():
    
    # this will download all the weights
    MarioLM()

if __name__ == "__main__":
    download_model()