from pydantic import BaseModel, Field, validator
from typing import List

class MealCard(BaseModel):
    name: str = Field(description="What is the name of the meal? Come up with an interesting name for the meal.")
    ingredients: str = Field(description="What are the ingredients for the meal? Give a list of ingredients and their quantities.")
    instructions: str = Field(description="What are the instructions for the meal? Give a numbered list of instructions.")

class Calories(BaseModel):
    calories: int = Field(description="Report the number of calories in the item. Just the number only.")

class Workout(BaseModel):
    name: str = Field(description="What is the name of the workout? Come up with an interesting name for the workout.")
    muscleGroup: str = Field(description="What muscle group does the workout target?")
    equipment: str = Field(description="What equipment is needed for the workout? Give a list of equipment needed.")
    instructions: List[str] = Field(description="For each day, Give a numbered list of instructions. Include the number of sets and reps. Each element in this list is a list of instructions for a day.")