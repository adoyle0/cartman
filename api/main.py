from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/DialoGPT-large", padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    "../train/cartman/models/output-medium")


class Packet(BaseModel):
    message: str
    max_new_tokens: int
    num_beams: int
    num_beam_groups: int
    no_repeat_ngram_size: int
    length_penalty: float
    diversity_penalty: float
    repetition_penalty: float
    early_stopping: bool


def cartman_respond(packet: Packet) -> str:
    input_ids = tokenizer(packet.message +
                          tokenizer.eos_token, return_tensors="pt").input_ids

    outputs = model.generate(
        input_ids,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=packet.max_new_tokens,
        num_beams=packet.num_beams,
        num_beam_groups=packet.num_beam_groups,
        no_repeat_ngram_size=packet.no_repeat_ngram_size,
        length_penalty=packet.length_penalty,
        diversity_penalty=packet.diversity_penalty,
        repetition_penalty=packet.repetition_penalty,
        early_stopping=packet.early_stopping,

        # do_sample = True,
        # top_k = 100,
        # top_p = 0.7,
        # temperature = 0.8,
    )
    return tokenizer.decode(outputs[:, input_ids.shape[-1]:][0],
                            skip_special_tokens=True)


api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.post('/chat/')
async def getInformation(request: Request) -> dict[str, str]:
    data = await request.json()

    packet = Packet(
        message=data.get('message'),
        max_new_tokens=data.get('max_new_tokens'),
        num_beams=data.get('num_beams'),
        num_beam_groups=data.get('num_beam_groups'),
        no_repeat_ngram_size=data.get('no_repeat_ngram_size'),
        length_penalty=data.get('length_penalty'),
        diversity_penalty=data.get('diversity_penalty'),
        repetition_penalty=data.get('repetition_penalty'),
        early_stopping=data.get('early_stopping'),
    )

    print(packet.message)
    response = cartman_respond(packet)
    print(response)

    return {"Cartman": response}
