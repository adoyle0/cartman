# Chatbots API

[FastAPI](https://fastapi.tiangolo.com/) and [PyTorch](https://pytorch.org/)

To build one yourself you'll need to first [train a model](../train), 
place the entire directory (checkpoints aren't needed) containing pytorch_model.bin in [bots](./src/bots), 
then edit or duplicate [cartman.py](./src/bots/cartman.py).

Cartman Docker images are availible for 
[x86_64](https://doordesk.net/files/chatbots_api_x86_64.tar.gz) (1.6GB) and 
[aarch64](https://doordesk.net/files/chatbots_api_x86_64.tar.gz) (1.4GB)

See [run](./run) and [test](./test) to interact with it

Live demo [here](https://doordesk.net/cartman)
