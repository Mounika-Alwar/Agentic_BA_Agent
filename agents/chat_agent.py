# Chat Agent
import streamlit as st
import json
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

# Uses Gemini to facilitate chat interface which has all the info that was generated till now as context 
class ChatAgent:
    """
    Conversational Agent that answers user queries
    using full context (goal, plan, analysis, insights)
    """

    def __init__(self, goal, plan, analysis_results, insights):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")

        self.goal = goal
        self.plan = plan
        self.analysis_results = analysis_results
        self.insights = insights

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=api_key,
            temperature=0.3
        )


    def build_context(self):
        context = f"""
You are an intelligent Business Analyst Chat Assistant.

Use the following context to answer user questions.

USER GOAL:
{self.goal}

PLAN:
{json.dumps(self.plan, indent=2)}

ANALYSIS RESULTS:
{str(self.analysis_results)}

INSIGHTS:
{self.insights}

Rules:
- Answer clearly and professionally
- Be concise but insightful
- Use data from analysis when possible
- If user asks about trends, explain with reasoning
- If unknown, say "Not enough information"
"""

        return context

    def get_response(self, user_query, chat_history):
        context = self.build_context()

        history_text = ""
        for msg in chat_history:
            role = msg["role"]
            content = msg["content"]
            history_text += f"{role.upper()}: {content}\n"

        final_prompt = f"""
{context}

CHAT HISTORY:
{history_text}

USER QUESTION:
{user_query}

ANSWER:
"""

        response = self.llm.invoke(final_prompt)

        return response.content.strip()


def render_chat_interface(goal, plan, analysis_results, insights):

    st.title("Chat with AI Analyst")
    st.write("You can now chat with the agent about your data and analysis.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something about your data...")

    if user_input:

        # show user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # generate response
        agent = ChatAgent(
            goal=goal,
            plan=plan,
            analysis_results=analysis_results,
            insights=insights
        )

        response = agent.get_response(
            user_input,
            st.session_state.chat_history
        )

        # show assistant response
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })
