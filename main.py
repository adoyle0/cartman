from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

api = FastAPI()

api.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*']
        )

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("../southpark/output-medium")

def cartman_respond(input_text):
    input_ids = tokenizer(input_text + tokenizer.eos_token, return_tensors="pt").input_ids
    outputs = model.generate(
            input_ids,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens = 200,
            num_beams = 8,
            num_beam_groups = 4,
            no_repeat_ngram_size=3,
            length_penalty = 1.4,
            diversity_penalty = 0,
            repetition_penalty = 2.1,
            early_stopping = True,

          # do_sample = True,
          # top_k = 100,
          # top_p = 0.7,
          # temperature = 0.8,
            )
    return tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

@api.post('/chat/')
async def getInformation(data : Request):
    packet = await data.json()
    message = packet.get('Message')
    print(message)
    response = cartman_respond(message)
    print(response)

    return {
        "Cartman" : response
    }

