import requests

API_ENDPOINT = 'http://localhost:5000/chat'

def send_message(message):
    data = {'message': message}
    response = requests.post(API_ENDPOINT, data=data)
    return response.json()['response']

while True:
    user_input = input('👨‍🦰 You: ')
    response = send_message(user_input)
    print('🤖 ChatbotX:', response)
    if response == 'Goodbye':
        break