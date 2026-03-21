# Insight Agent
import json
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import streamlit as st

# Generates clean insights based on the analysis results, goal and dataset
class InsightAgent:
    """
    Insight Agent:
    Generates business insights from analysis results using Gemini.
    """

    def __init__(self, analysis_results, goal):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        self.results = analysis_results
        self.goal = goal

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=api_key,
            temperature=0.3
        )

    def prepare_context(self):
        """
        Convert results into LLM-friendly text
        """

        try:
            return json.dumps(self.results, indent=2, default=str)
        except:
            return str(self.results)

    def build_prompt(self):
        template = """
You are a senior Business Analyst.

Your job is to extract meaningful business insights from analysis results.

User Goal:
{goal}

Analysis Results:
{results}

INSTRUCTIONS:
- Generate 5 to 10 clear, high-quality insights
- Focus on business meaning, not raw numbers
- Identify patterns, trends, anomalies, risks, opportunities
- Keep insights short and impactful
- Each insight should be 1–2 lines max

OUTPUT FORMAT:
- Return ONLY bullet points
- No explanations
- No headings
- No numbering

Example:
- Sales increase significantly with customer age
- High correlation between income and spending score
- Cluster 2 represents high-value customers
- Model shows strong predictive performance

Generate insights:
"""

        return PromptTemplate(
            input_variables=["goal", "results"],
            template=template
        )

    def generate_insights(self):
        prompt = self.build_prompt()

        formatted_prompt = prompt.format(
            goal=self.goal,
            results=self.prepare_context()
        )

        response = self.llm.invoke(formatted_prompt)

        return self.parse_output(response.content)

    def parse_output(self, text):
        """
        Convert LLM output into clean list
        """
        lines = text.strip().split("\n")

        insights = []
        for line in lines:
            line = line.strip()

            if line.startswith("-"):
                insights.append(line[1:].strip())
            elif line:
                insights.append(line)

        return insights
