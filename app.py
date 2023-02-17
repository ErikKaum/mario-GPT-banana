from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from mario_gpt.lm import MarioLM
from mario_gpt.utils import convert_level_to_png, view_level
from io import BytesIO
import base64

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global tokenizer
    global mario_lm
    
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained("shyamsn97/Mario-GPT2-700-context-length")
    model = AutoModelForCausalLM.from_pretrained("shyamsn97/Mario-GPT2-700-context-length")
    mario_lm = MarioLM(model, tokenizer).to("cuda")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model

    generated_level = mario_lm.sample(
        prompts=prompt,
        num_steps=100, # change later
        temperature=2.0,
        use_tqdm=True
    )

    TILE_DIR = "tiles"

    print(generated_level.shape)

    img = convert_level_to_png(generated_level.squeeze(), TILE_DIR, mario_lm.tokenizer)[0]

    buffered = BytesIO()
    img.save(buffered,format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')


    # Return the results as a dictionary
    return { "image": image_base64}