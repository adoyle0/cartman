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

def cartman_respond(packet):
    input_ids = tokenizer(str(packet.get('message')) + tokenizer.eos_token, return_tensors="pt").input_ids
    outputs = model.generate(
            input_ids,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens = int(packet.get('max_new_tokens')),
            num_beams = int(packet.get('num_beams')),
            num_beam_groups = int(packet.get('num_beam_groups')),
            no_repeat_ngram_size = int(packet.get('no_repeat_ngram_size')),
            length_penalty = float(packet.get('length_penalty')),
            diversity_penalty = float(packet.get('diversity_penalty')),
            repetition_penalty = float(packet.get('repetition_penalty')),
            early_stopping = bool(packet.get('early_stopping')),

          # do_sample = True,
          # top_k = 100,
          # top_p = 0.7,
          # temperature = 0.8,
            )
    return tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

@api.post('/chat/')
async def getInformation(data : Request):
    packet = await data.json()
    print(packet)
    message = str(packet.get('message'))
    print(message)
    response = cartman_respond(packet)
    print(response)

    return {
        "Cartman" : response
    }

