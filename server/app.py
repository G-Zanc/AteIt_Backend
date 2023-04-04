from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import datetime
import openai
from dotenv import load_dotenv
from pymongo import MongoClient
from pathlib import Path
import os

path = Path('./config.env')
load_dotenv(dotenv_path=path)
ATLAS_URL = os.getenv('ATLAS_URI')
API_KEY = os.getenv('API_KEY')
app = Flask(__name__)
cors = CORS(app, resources={r"/message": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
client = MongoClient(ATLAS_URL)
db = client.AteIt
users = db.users
openai.api_key = API_KEY

#MESSAGING ROUTE PLAN TO ADD CHATGPT CONNECTION, AS WELL AS FILTERING BOTS TO DETECT AND STORE PIECES OF INFORMATION
@app.route('/message', methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def gpt3_response():
    data = request.get_json()
    prompt = data['prompt']
    email = data['email']
    user = users.find_one({'email': email})
    chats = [x['response'] for x in user['messages']]
    
    if prompt == "" or email == "": 
        response = {
            "response": "no information provided, please try again"
        }
        return jsonify(response)
    
    api_message = {"role": "user", "content": prompt}
    chats.append(api_message)
    chats.insert(0, {"role": "system", "content": "You are a nutrition, fitness, and health assistant. Any questions unrelated to these topics should not be acknowledged"})
    message = {
        "response": api_message,
        "date": datetime.datetime.now(),
    }


    result = users.update_one({'email': email}, {'$push': {'messages': message}})
    if result.modified_count == 0:
        response = {
            "response": "error occurred try again"
        }

        return(jsonify(response))


    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chats
    )

    response = {
        "response": {"role": "assistant", "content": completions.choices[0].message.content},
        "date": datetime.datetime.now(),
    }

    users.update_one({'email': email}, {'$push': {'messages': response}})

    return jsonify(response)


if __name__ == '__main__':
    app.run()
