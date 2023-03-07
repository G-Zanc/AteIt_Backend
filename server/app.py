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
app = Flask(__name__)
cors = CORS(app, resources={r"/message": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
client = MongoClient(ATLAS_URL)
db = client.AteIt
users = db.users

#openai.api_key = "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"

@app.route('/message', methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def gpt3_response():
    data = request.get_json()
    print(data)
    prompt = data['prompt']
    user = data['email']
    if prompt == "" or user == "": 
        response = {
            "response": "no information provided, please try again"
        }
        return jsonify(response)
    
    message = {
        "text": prompt,
        "date": datetime.datetime.now(),
        "byUser": True
    }


    result = users.update_one({'email': user}, {'$push': {'messages': message}})
    if result.modified_count == 0:
        response = {
            "response": "error occurred try again"
        }
        return(jsonify(response))


    '''completions = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    response = {
        "response": completions.choices[0].text.strip()
    }'''

    response = {
        "response": "prompt: {} Response: Testing Hello World!".format(prompt),
        "date": datetime.datetime.now(),
        "byUser": False
    }

    users.update_one({'email': user}, {'$push': {'messages': response}})

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
