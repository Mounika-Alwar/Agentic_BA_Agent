# Planner Agent
import json
import re
from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import streamlit as st


# Uses gemini to make the detailed plan as per dataset and goal of the user
class PlannerAgent:
    """
    Planner Agent:
    Converts user goal into structured JSON plan using Gemini via LangChain.
    """

    def __init__(self, df, goal):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Add it to you .env for local or streamlit secrets for cloud"
            )

        self.df = df
        self.goal = goal
        self.columns = df.columns.tolist()

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=api_key,
            temperature=0
        )


    def build_prompt(self):
        template = """
You are an expert Business Analyst Planner Agent.

Convert the user goal into a structured execution plan.

Dataset Columns:
{columns}

User Goal:
{goal}

Available Functions:
- preview_data
- null_analysis
- describe_data
- histogram (requires: column)
- scatter_plot (requires: col1, col2)
- correlation_matrix
- regression_model
- classification_model
- clustering_model

STRICT RULES:
- Output ONLY valid JSON
- Output must be a LIST of steps
- Each step must contain "step"
- Add parameters where required
- Use ONLY dataset column names
- ALWAYS include:
    preview_data
    null_analysis
    describe_data

Example:
[
  {{"step": "preview_data"}},
  {{"step": "null_analysis"}},
  {{"step": "describe_data"}},
  {{"step": "histogram", "column": "age"}},
  {{"step": "scatter_plot", "col1": "age", "col2": "salary"}},
  {{"step": "regression_model"}}
]

Generate plan:
"""

        return PromptTemplate(
            input_variables=["columns", "goal"],
            template=template
        )

    def generate_plan(self):
        prompt = self.build_prompt()

        formatted_prompt = prompt.format(
            columns=self.columns,
            goal=self.goal
        )

        response = self.llm.invoke(formatted_prompt)
        text = response.content.strip()

        return self.safe_parse(text)


    def safe_parse(self, text):
        try:
            return json.loads(text)
        except:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass

        # fallback
        return [
            {"step": "preview_data"},
            {"step": "null_analysis"},
            {"step": "describe_data"}
        ]


    def format_plan(self, plan):
        return json.dumps(plan, indent=4)
