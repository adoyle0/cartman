from pydantic import BaseModel


class Packet(BaseModel):
    bot_name: str
    message: str
    max_new_tokens: int
    num_beams: int
    num_beam_groups: int
    no_repeat_ngram_size: int
    length_penalty: float
    diversity_penalty: float
    repetition_penalty: float
    early_stopping: bool


class BotResponse(BaseModel):
    name: str
    message: str
