from mario_gpt.lm import MarioLM
from mario_gpt.utils import convert_level_to_png 
from io import BytesIO
import base64

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    model = MarioLM().to("cuda")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    generated_level = model.sample(
        prompts=prompt,
        num_steps=699,
        temperature=2.0,
        use_tqdm=True
    )

    TILE_DIR = "tiles"

    print(generated_level.shape)

    img = convert_level_to_png(generated_level.squeeze(), TILE_DIR, model.tokenizer)[0]

    buffered = BytesIO()
    img.save(buffered,format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')


    # Return the results as a dictionary
    return { "image": image_base64 }