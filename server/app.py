from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import datetime
import openai
from dotenv import load_dotenv
from pymongo import MongoClient
from pathlib import Path
from waitress import serve
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
    try:
        data = request.get_json()
        prompt = data['prompt']
        email = data['email']
        user = users.find_one({'email': email})
        chats = [x['response'] for x in user['messages']]
        print("Got user chat history")
        if prompt == "" or email == "": 
            print("Empty prompt or incorrect email provided")
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
            print("Error occurred while updating user chat history")
            response = {
                "response": "error occurred try again"
            }

            return(jsonify(response))

        #ADD FILTERING FOR RESPONSES
        #UNDERSTAND WHAT KIND OF REQUEST IS BEING MADE
        #IF USER IS ASKING FOR MEAL PLAN, CREAT A JSON FORMAT FOR MEAL PLAN THAT CAN BE FORMATED BY FRONTEND
        #IF USER IS ASKING FOR A RECIPE, CREATE A JSON FORMAT FOR RECIPE THAT CAN BE FORMATED BY FRONTEND
        #WHEN PROMPTED WITH QUESTION THAT ASKS FOR PROGRESS UPDATE, QUERY DB FOR METRICS SUCH AS WEIGHT, SEND TALORED RESPONSE TO USER
        #IF USER IS ASKING FOR A WORKOUT, CREATE A JSON FORMAT FOR WORKOUT THAT CAN BE FORMATED BY FRONTEND
        
        completions = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=chats
        )

        response = {
            "response": {"role": "assistant", "content": completions.choices[0].message.content},
            "date": datetime.datetime.now(),
        }

        users.update_one({'email': email}, {'$push': {'messages': response}})
        print("Sending healthy response")
        return jsonify(response)
    except Exception as e:
        print(e)
        response = {
            "response": "error occurred try again"
        }
        return(jsonify(response))


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
