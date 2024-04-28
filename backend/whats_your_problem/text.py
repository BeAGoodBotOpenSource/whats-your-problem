import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import (
    CommaSeparatedListOutputParser,
    PydanticOutputParser,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

with open("prompts/advisors.md", "r") as f:
    advisors_prompt = f.read()


class Personas(BaseModel):
    personas: list[str] = Field(
        description="A list of personas that have insight on the problem."
    )


advisors_parser = PydanticOutputParser(pydantic_object=Personas)

advisors_prompt_template = PromptTemplate(
    template=advisors_prompt,
    input_variables=["background", "problem"],
    partial_variables={
        "format_instructions": advisors_parser.get_format_instructions()
    },
)


def get_personas(background: str, problem: str) -> list[str]:
    """Return a list of personas"""

    prompt = advisors_prompt_template.format(background=background, problem=problem)
    personas = chat_model.predict(prompt)
    personas = advisors_parser.parse(personas)

    return personas.model_dump()["personas"]


with open("prompts/questions.md", "r") as f:
    questions_prompt = f.read()


class Category(BaseModel):
    category: str = Field(
        description="A category to be considered in relation to the problem"
    )
    questions_in_favour: list[str] = Field(
        "Quesions to consider in favor of a course of action."
    )
    questions_against: list[str] = Field(
        "Quesions to consider against of a course of action."
    )


class Question(BaseModel):
    categories: list[Category] = Field(
        description="Category to consider and questions in favor and against course of action."
    )


questions_parser = PydanticOutputParser(pydantic_object=Question)

questions_prompt_template = PromptTemplate(
    template=questions_prompt,
    input_variables=["background", "problem", "advisor"],
    partial_variables={
        "format_instructions": questions_parser.get_format_instructions()
    },
)


async def async_prompt(chat_model, persona, background, problem):
    prompt = questions_prompt_template.format(
        background=background, problem=problem, advisor=persona
    )
    c = await chat_model.agenerate([[HumanMessage(content=prompt)]])
    print(persona)
    return c


async def generate_concurrently(personas, background, problem):
    tasks = [async_prompt(chat_model, p, background, problem) for p in personas]
    return await asyncio.gather(*tasks)


def get_questions(personas, background, problem) -> list[dict]:
    """Get questions for each persona.

    Returns
    -------
    json
        Formatted like:
        [
            {
                "advisor": "Advisor Name",
                "categories":
                [
                    {
                        "category": "Some Category",
                        "questions_in_favor": [
                            "Question 1",
                            "Question 2",
                            ...
                        ],
                        "questions_against": [
                            "Question 1",
                            "Question 2",
                            ...
                        ],
                    },
                    ...
                ]
            },
            ...
        ]
    """
    questions = asyncio.run(generate_concurrently(personas, background, problem))

    questions = [
        questions_parser.parse(q.generations[0][0].text).model_dump() for q in questions
    ]

    for i in range(len(personas)):
        questions[i]["advisor"] = personas[i]

    return questions


with open("prompts/further_info.md", "r") as f:
    further_info_prompt = f.read()

further_info_prompt_template = PromptTemplate(
    template=further_info_prompt,
    input_variables=["background", "problem", "advisor", "question"],
)


def get_further_info(question, persona, background, problem):
    print(question)
    print(persona)
    print(background)
    print(problem)
    prompt = further_info_prompt_template.format(
        background=background, problem=problem, advisor=persona, question=question
    )
    further_info = chat_model.predict(prompt)
    return further_info


if __name__ == "__main__":
    background = "I'm a socially conscious AI developer, focusing on making AI that is smarter and more intelligent that humans, and can do the work that humans do more effectively."
    problem = "Should I move to South boston?"

    personas = get_personas(background, problem)
    print(personas)

    questions = get_questions(personas, background, problem)
    print(questions)

    question = "Is the community in South Boston welcoming and inclusive?"
    persona = "Small business owner"

    # further_info = get_further_info(question, persona, background, problem)
    # print(further_info)
