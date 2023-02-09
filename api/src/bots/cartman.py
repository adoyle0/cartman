from ..models import Packet, BotResponse

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/DialoGPT-large", padding_side='left'
)
model = AutoModelForCausalLM.from_pretrained(
    "../train/cartman/models/output-medium"
)


def cartman(packet: Packet) -> BotResponse:
    input_ids = tokenizer(
        packet.message + tokenizer.eos_token,
        return_tensors="pt"
    ).input_ids

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

    return BotResponse(
        name='Cartman',
        message=tokenizer.decode(
            outputs[:, input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
    )
