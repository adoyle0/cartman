from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.models import Packet, BotResponse
from src.bots.cartman import cartman


api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.post('/chat/')
async def receive_packet(packet: Packet) -> BotResponse:
    match packet.bot_name:
        case 'cartman':
            return cartman(packet)
        case _:
            return BotResponse(
                name='Error',
                message='bot_name is invalid'
            )
