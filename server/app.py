from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from pathlib import Path
from waitress import serve
import os
import json
from langchain.agents import initialize_agent, Tool, load_tools, AgentType
from langchain import OpenAI
from pymongo import MongoClient
import traceback
import json
import sys
from LLM_Tools.CustomTools import *
#sys.path.append('filter/chatfunc')
#import filter.chatfunc 

path = Path('./config.env')
load_dotenv(dotenv_path=path)
ATLAS_URL = os.getenv('ATLAS_URI')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['WOLFRAM_ALPHA_APPID'] = os.getenv('WOLFRAM_ALPHA_APPID')
os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')
app = Flask(__name__)
cors = CORS(app, resources={r"/message": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
client = MongoClient(ATLAS_URL)
db = client.AteIt
users = db.users

wolf_llm = OpenAI(temperature=0, verbose=False)
tool_names = ['google-serper', 'wolfram-alpha']
wolf_agent = initialize_agent(load_tools(tool_names), wolf_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

#MESSAGING ROUTE PLAN TO ADD CHATGPT CONNECTION, AS WELL AS FILTERING BOTS TO DETECT AND STORE PIECES OF INFORMATION
@app.route('/', methods=['GET'])
def home():
    return "API is running"

@app.route('/getChat', methods=['POST'])
def getChat():
    try:
        data = request.get_json()
        pageNum = data['pageNum']
        pageSize = data['pageSize']
        email = data['email']
        skip = -(pageNum * pageSize + pageSize)
        messages = users.find({'email': email}, {'messages': {'$slice': [skip, pageSize]}})
        documents_list = [doc for doc in messages]
        print(documents_list[0]['messages'])
        json_result = json.dumps(documents_list[0]['messages'], default=str)
        return(json_result)
    except Exception as e:
        print(e)
        response = {
            'error': e
        }
        return(jsonify(response))

@app.route('/message', methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def gpt3_response():
    try:
        print("request made!")
        data = request.get_json()
        prompt = data['prompt']
        email = data['email']
        user = users.find_one({'email': email}, {"_id": 1})

        print(f'Request made by: {email}')
        if user == None :
            print("User not found")
            response = {
                "response": "User not found"
            }
            return jsonify(response)
        
        if prompt == "" or email == "": 
            print("Empty prompt or incorrect email provided")
            response = {
                "response": "no information provided, please try again"
            }
            return jsonify(response)
        
        #FILTER RESPONSE, UNDERSTAND TYPE OF QUESTION AND CREATE CUSTOMIZED RESPONSE BASED ON REQUEST TYPE (MEAL, WORKOUT, RESEARCH, DB SEARCH, etc)
        api_message = {"role": "user", "content": prompt}
        message = {
            "response": api_message,
            "date": datetime.datetime.now(),
        }

        result = users.update_one({'email': email}, {'$push': {'messages': message}})
        if result.modified_count == 0:
            print("Error occurred while updating user chat history")
            response = {
                "response": "Error occurred while updating user chat history"
            }
            return(jsonify(response))

        '''reply = filter.chatfunc.runFilter(prompt)
        if reply != "callGPT":
            response = {
            "response": {"role": "assistant", "content": reply},
            "date": datetime.datetime.now(),
            }
            users.update_one({'email': email}, {'$push': {'messages': response}})
            app.logger.info("Sending healthy response")       
            return jsonify(response)
        '''

        #CREATE TOOLS AND AGENT
        tools = [
            Tool(
                name="Make Recipe",
                func= lambda n: makeRecipe(n, email, users),
                description="anytime meal or recipe is mentioned use this tool. n is the main item in the meal or recipe. FOR FOODS ONLY",
                return_direct=True
            ),
            Tool(
                name="Workout Plan",
                func= lambda n: makeWorkout(n),
                description="anytime workout is mentioned use this tool. n is the number of days. FOR EXERCISES ONLY",
                return_direct=True
            ),
            Tool(
                name="Normal Response",
                func= lambda n: normalResponse(prompt, email),
                description="If there is no tool that can be used, use this tool to respond normally. n is the prompt.",
                return_direct=True
            ),
            Tool(
                name="Meal Plan",
                func= lambda n: mealPlan(n),
                description="anytime meal plan is mentioned use this tool. n is the number of days. FOR FOODS ONLY",
                return_direct=True
            ),
            Tool(
                name="Remember restriction",
                func= lambda n: rememberRestriction(n, email, users),
                description="Anytime some kind of dietary restriction is mentioned, use this tool. n is the restriction. This could be things such as a gluten allergy, or a vegan diet. Foods that a user doesn't want is included such as not liking broccoli.",
                return_direct=True
            )
        ]

        #CREATE AGENT AND MAKE CALL
        NutriAgent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

        #ADD FILTERING FOR RESPONSES
        #UNDERSTAND WHAT KIND OF REQUEST IS BEING MADE
        #IF USER IS ASKING FOR MEAL PLAN, CREAT A JSON FORMAT FOR MEAL PLAN THAT CAN BE FORMATED BY FRONTEND
        #IF USER IS ASKING FOR A RECIPE, CREATE A JSON FORMAT FOR RECIPE THAT CAN BE FORMATED BY FRONTEND
        #WHEN PROMPTED WITH QUESTION THAT ASKS FOR PROGRESS UPDATE, QUERY DB FOR METRICS SUCH AS WEIGHT, SEND TALORED RESPONSE TO USER
        #IF USER IS ASKING FOR A WORKOUT, CREATE A JSON FORMAT FOR WORKOUT THAT CAN BE FORMATED BY FRONTEND
        '''
        completions = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=chats
        )

        response = {
            "response": {"role": "assistant", "content": completions.choices[0].message.content},
            "date": datetime.datetime.now(),
        }
        '''

        answer = NutriAgent.run(prompt)
        response = {
            "response": {"role": "assistant", "content": answer},
            "date": datetime.datetime.now(),
        }

        users.update_one({'email': email}, {'$push': {'messages': response}})
        print("Sending healthy response")
        return jsonify(response)
    
    except Exception as e:
        print(e)
        response = {
            "response": e
        }
        return(jsonify(response))

if __name__ == '__main__':
    print("Starting server")
    serve(app, host='0.0.0.0', port=5000)
