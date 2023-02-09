import requests

while True:
    user_input: str = input('>> ')
    if user_input in 'qx':
        break
    else:
        packet = {
            'message': user_input,
            'max_new_tokens': 20,
            'num_beams': 2,
            'num_beam_groups': 2,
            'no_repeat_ngram_size': 3,
            'length_penalty': 1.4,
            'diversity_penalty': 0.1,
            'repetition_penalty': 2.1,
            'early_stopping': True,
        }

    response = requests.post(
        'http://127.0.0.1:8000/chat/',
        json=packet,
    ).json()

    print(f"{response.get('name')}: {response.get('message')}")
