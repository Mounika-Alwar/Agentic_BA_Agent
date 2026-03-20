import os
import json
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

# PDF libraries
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


class ReportAgent:
    """
    Generates a business report and supports PDF export
    """

    def __init__(self, goal, plan, analysis_results, insights, user_preferences):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")

        self.goal = goal
        self.plan = plan
        self.analysis_results = analysis_results
        self.insights = insights
        self.user_preferences = user_preferences

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=api_key,
            temperature=0.4
        )

    # -----------------------------
    # BUILD PROMPT
    # -----------------------------
    def build_prompt(self):
        return f"""
You are a professional Business Analyst.

Generate a detailed business report based on the following:

USER GOAL:
{self.goal}

PLAN:
{json.dumps(self.plan, indent=2)}

ANALYSIS RESULTS:
{str(self.analysis_results)}

INSIGHTS:
{self.insights}

USER PREFERENCES:
{self.user_preferences}

INSTRUCTIONS:
- Write a well-structured report
- Use clear headings
- Include interpretation of results
- Keep it professional

OUTPUT:
Return clean markdown text
"""

    # -----------------------------
    # GENERATE REPORT
    # -----------------------------
    def generate_report(self):
        prompt = self.build_prompt()
        response = self.llm.invoke(prompt)
        return response.content.strip()

    # -----------------------------
    # SAVE AS PDF
    # -----------------------------
    def save_as_pdf(self, report_text, file_path="report.pdf"):
        """
        Saves the report text as a PDF file
        """

        doc = SimpleDocTemplate(file_path)
        styles = getSampleStyleSheet()

        content = []

        # Split report into lines
        for line in report_text.split("\n"):

            if line.strip() == "":
                content.append(Spacer(1, 10))
            else:
                content.append(Paragraph(line, styles["Normal"]))
                content.append(Spacer(1, 8))

        doc.build(content)

        return file_path
