from langchain.prompts import PromptTemplate
from LLM_Tools.NutritionClasses import MealCard, Calories, Workout
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import initialize_agent, Tool, load_tools, AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

path = Path('.\config.env')
load_dotenv(dotenv_path=path)
ATLAS_URL = os.getenv('ATLAS_URI')
client = MongoClient(ATLAS_URL)
db = client.AteIt
users = db.users
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
llm = OpenAI(temperature=1, verbose=True)
parser = PydanticOutputParser(pydantic_object=MealCard)
cal_parser = PydanticOutputParser(pydantic_object=Calories)
workout_parser = PydanticOutputParser(pydantic_object=Workout)

chatllm = OpenAI(temperature=1, verbose=True)
chat_tools=['google-serper']

def mealPlan(n):
    # Create Meal Plan. Return a list of json objects for each meal
    # Each meal should have a name, description, ingredients, and instructions
    return f'Eat {n} Salads!'

#MAKES NORMAL LLM RESPONSE
def normalResponse(prompt, email):
    #LOAD MESSAGE HISTORY AND CREATE RESPONSE
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    skip = -20
    messages = users.find({'email': email}, {'messages': {'$slice': [skip, 20]}})
    documents_list = [doc for doc in messages]

    if(documents_list != []):  
        for doc in documents_list[0]['messages']:
            if(doc['response']['role'] == 'user'):
                memory.chat_memory.add_user_message(doc['response']['content'])
            else:
                memory.chat_memory.add_ai_message(doc['response']['content'])

    chat_agent = initialize_agent(load_tools(chat_tools), chatllm, 
                                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                                memory=memory, verbose=True, max_iterations=3)
    
    response = chat_agent.run(prompt)
    return response

#CREATES JSON TEMPLATE FOR A WORKOUT
def makeWorkout(n):
    # Create a workout. Return a json object with name, description, and exercises
    prompt = PromptTemplate(
        template="Anser the user query. \n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": workout_parser.get_format_instructions()},
    )

    _input = prompt.format_prompt(query=f'Make a workout for me for {n} days')
    output = llm(_input.to_string())
    return output

#ADDS RESTRICTION TO USER DATABASE
def rememberRestriction(n, email, users):
    #ADD RESTRICTION TO DATABASE FOR USER, DATABASE WILL HAVE A LIST OF RESTRICTIONS IN STRING REPRESENTATION
    result = users.update_one({'email': email}, {'$push': {'restrictions': n}})
    if result.modified_count > 0:
        print("Recipe added to database")
    else:
        print("Recipe not added to database")
        
    return f"I'll remember the restriction: {n}"

#CREATES RECIPE RETURNS JSON TEMPLATE
def makeRecipe(n, email, users):
    # Create a recipe. Return a json object with name, description, ingredients, and instructions
    prompt = PromptTemplate(
        template="Anser the user query. \n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    calPrompt = PromptTemplate(
        template="Anser the user query. \n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": cal_parser.get_format_instructions()},
    )

    #CREATE RECIPE JSON TEMPLATE
    #IMPLEMENT ERROR CHECKING HERE TO INSURE THAT A RECIPE IS RETURNED
    #CONSIDER ALERGIES AND DIETARY RESTRICTIONS IN PROMPT CREATION

    #CREATE QUESTION FLOW IF A TYPE OF RECIPE IS NOT PROVIDED
    #WHAT KIND OF INGREDIENTS DO YOU WANT/HAVE?
    
    notDone = True
    if(n != ''): 
        meal_query = f"Make a healthy recipe for {n}"
    else:
        meal_query = f"Make a different healthy recipe"

    #ADD RECIPE TO DATABASE
    #CALCULATE CALORIES AND NUTRITION?

    #CREATE WOLFRAM ALPHA AGENT AND MAKE CALL TO AGENT
    #INFORMATION NEEDS TO BE CORRECTLY FORMATED FOR WOLFRAM ALPHA

    #CHECK IF SIMILAR RECIPE EXISTS?
    
    while(notDone):
        try:
            _input = prompt.format_prompt(query=meal_query)
            output = llm(_input.to_string())
            mongoOut = json.loads(output)
            ingredients = mongoOut['ingredients'].split(',')
            totalCal = 0

            '''
            for ingredient in ingredients:
                lock = True
                while lock:
                    try:
                        ing_input = calPrompt.format_prompt(query=ingredient)
                        result = wolf_agent.run(ing_input.to_string())
                        calOut = llm(f'Extract just the number of calories as an integer from the following text: {result}')
                        print(calOut)
                        totalCal += int(calOut)
                        lock = False
                    except Exception as e:
                        print(e)
                        print("NOT JUST A NUMBER")
                        
            print(f'total calories: {totalCal}')
            '''

            result = users.update_one({'email': email}, {'$push': {'recipes': mongoOut}})
            if result.modified_count > 0:
                print("Recipe added to database")
            else:
                print("Recipe not added to database")
            notDone = False
        except Exception as e:
            print(e)
            print("BAD JSON OUT TRY AGAIN")
    
    return f"I've added the recipe to your book!"