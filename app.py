# Main APP
import streamlit as st
import pandas as pd
from agents.planner_agent import PlannerAgent
from agents.analyst_agent import AnalystAgent
from agents.insight_agent import InsightAgent
from agents.chat_agent import render_chat_interface
from agents.report_agent import ReportAgent
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
st.set_page_config(page_title = "Agentic AI Business Analyst",layout="wide")



st.sidebar.title("📊 Navigation")
st.sidebar.markdown("---")

if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.sidebar.button("🏠 Home", use_container_width=True):
    st.session_state.page = "Home"

if st.sidebar.button("💬 Chat with Agent", use_container_width=True):
    st.session_state.page = "Chat"

if st.sidebar.button("📄 Report Generation", use_container_width=True):
    st.session_state.page = "Report"

page = st.session_state.page

if "df" not in st.session_state:
    st.session_state.df = None

if "goal" not in st.session_state:
    st.session_state.goal = ""\

if "plan" not in st.session_state:
    st.session_state.plan = None

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

if "insights" not in st.session_state:
    st.session_state.insights = None

if page == "Home":

    st.title("Agentic AI Powered Autonomous Business Analyst Agent - A Next Generation Framework for Intelligent Business Automation")

    st.write(
        """**This application demonstrates an agentic AI-powered Business Analyst that can autonomously analyze datasets, generate insights, and create reports based on user-defined goals. It leverages Google's Gemini model via LangChain to perform complex data analysis tasks and provide actionable business insights.** """
    )



    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success("Dataset uploaded successfully!")

        except Exception as e:
            st.error(f"Error loading file: {e}")

    goal_input = st.text_area("Enter your analysis goal")

    if goal_input:
        st.session_state.goal = goal_input

    if st.session_state.df is not None:
        df = st.session_state.df

        st.subheader("Dataset Preview")
        
        max_rows = df.shape[0]
        max_cols = df.shape[1]

        row_count = st.slider(
            "Select number of rows to view",
            min_value=1,
            max_value=min(100,max_rows),
            value=min(5,max_rows)
        )
        selected_columns = st.multiselect(
            "Select columns to display",
            df.columns.tolist(),
            default = df.columns[:5]
        )

        preview_df = df[selected_columns].head(row_count)

        st.dataframe(preview_df)

    st.subheader("Actions")

    #col1,col2,col3,col4,col5 = st.columns(5)

    if st.session_state.df is not None and st.session_state.goal:
        if st.button("Generate Plan"):
            if st.session_state.df is None or not st.session_state.goal:
                st.warning("Please upload dataset and enter goal first")
            else:
                planner = PlannerAgent(
                    df = st.session_state.df,
                    goal = st.session_state.goal
                )
                plan = planner.generate_plan()
                st.session_state.plan = plan

    if st.session_state.plan is not None:
        st.subheader("Generated Plan (Editable)")

        plan_text = st.text_area(
            "Edit Plan JSON",
            value = json.dumps(st.session_state.plan, indent=4),
            height = 300
        )

        try:
            edited_plan = json.loads(plan_text)
            st.session_state.plan = edited_plan
        except:
            st.warning("Invalid JSON format")

        if st.button("Run Analysis"):
            if st.session_state.plan is None:
                st.warning("Generate plan first")
            else:
                analyst = AnalystAgent(st.session_state.df)
                results = analyst.run_analysis(st.session_state.plan)
                st.session_state.analysis_results = results

    if st.session_state.analysis_results is not None:
        st.subheader("Analysis Results")

        results = st.session_state.analysis_results

        if "preview" in results:
            st.write("### Preview")
            st.dataframe(results["preview"])

        if "nulls" in results:
            st.write("### Missing Values")
            st.write(results["nulls"])

        if "description" in results:
            st.write("### Statistical Summary")
            st.write(results["description"])

        if "plots" in results:
            st.write("### Visualizations")

            plots = results["plots"]
            
            if "correlation" in plots:
                st.write("#### Correlation Matrix")
                st.plotly_chart(plots["correlation"],use_container_width=True)
            
            if "histograms" in plots:
                st.write("#### Histograms")
                for item in plots["histograms"]:
                    st.write(f"{item['column']}")
                    if isinstance(item["figure"],str):
                        st.error("Plot not generated properly")
                        st.write(item["figure"])
                    else:
                        st.plotly_chart(item["figure"],use_container_width=True)

            if "scatter_plots" in plots:
                st.write("#### Scatter Plots")
                for item in plots["scatter_plots"]:
                    st.write(f"{item['col1']} vs {item['col2']}")
                    st.plotly_chart(item["figure"],use_container_width=True)
        
        if "models" in results:
            st.write("### Model Results")
            for key, val in results["models"].items():
                st.write(f"#### {key}")

                # ------------------ REGRESSION ------------------
                if key == "regression":
                    st.write("R2 Score:", val.get("r2_score", "N/A"))
                    st.write("MSE:", val.get("mse", "N/A"))

                # ------------------ CLASSIFICATION ------------------
                elif key == "classification":
                    st.write("Accuracy:", val.get("accuracy", "N/A"))
                    st.write("F1 Score:", val.get("f1_score", "N/A"))

                    if "confusion_matrix" in val:
                        st.write("Confusion Matrix:")

                        cm = val["confusion_matrix"]

                        cm_df = pd.DataFrame(
                            cm,
                            columns=["Predicted 0", "Predicted 1"],
                            index=["Actual 0", "Actual 1"]
                        )

                        st.dataframe(cm_df)

                # ------------------ CLUSTERING ------------------
                elif key == "clustering":
                    st.write("Cluster Labels:")
                    st.write(val.get("clusters", []))

                # ------------------ FALLBACK ------------------
                else:
                    st.json(val)


        if st.button("Generate Insights"):
            if st.session_state.analysis_results is None:
                st.warning("Run analysis first")
            else:
                insight_agent = InsightAgent(
                    analysis_results = st.session_state.analysis_results,
                    goal = st.session_state.goal
                )
                insights = insight_agent.generate_insights()
                st.session_state.insights = insights

    if "insights" in st.session_state and st.session_state.insights:
        st.subheader("Key Insights")

        for insight in st.session_state.insights:
            st.markdown(f"- {insight}")

# chat tab

elif page == "Chat":

    if st.session_state.df is None:
        st.warning("Please upload dataset on Home tab first")
    elif not st.session_state.goal:
        st.warning("Please enter goal on Home tab first")
    else:
        
        render_chat_interface(
            goal = st.session_state.goal,
            plan = st.session_state.plan,
            analysis_results = st.session_state.analysis_results,
            insights = st.session_state.get("insights",[])
        )

# report tab



elif page == "Report":
    st.header("Automated Report Generation")

    if st.session_state.analysis_results is None:
        st.warning("Please complete analysis first")
    else:
        st.subheader("Customize Your Report")


        tone = st.selectbox(
            "Report Tone",
            ["Professional", "Technical", "Executive", "Simple"]
        )

        audience = st.selectbox(
            "Target Audience",
            ["Business Stakeholders", "Data Scientists", "Management", "General"]
        )

        sections = st.multiselect(
            "Select Sections",
            [
                "Executive Summary",
                "Data Overview",
                "Key Findings",
                "Visual Insights",
                "Model Performance",
                "Business Recommendations",
                "Conclusion"
            ],
            default=[
                "Executive Summary",
                "Key Findings",
                "Business Recommendations"
            ]
        )

        length = st.selectbox(
            "Report Length",
            ["Short", "Medium", "Detailed"]
        )

        extra_notes = st.text_area(
            "Any additional instructions?",
            placeholder="e.g., Focus more on sales trends..."
        )

        if st.button("Generate Report"):

            preferences = {
                "tone": tone,
                "audience": audience,
                "sections": sections,
                "length": length,
                "extra_notes": extra_notes
            }

            report_agent = ReportAgent(
                goal=st.session_state.goal,
                plan=st.session_state.plan,
                analysis_results=st.session_state.analysis_results,
                insights=st.session_state.get("insights", []),
                user_preferences=preferences
            )

            report = report_agent.generate_report()

            st.session_state.report = report


        if "report" in st.session_state and st.session_state.report:

            st.subheader("Generated Report")

            st.markdown(st.session_state.report)

            if st.button("Download as PDF"):

                agent = ReportAgent(
                    goal = st.session_state.goal,
                    plan = st.session_state.plan,
                    analysis_results = st.session_state.analysis_results,
                    insights = st.session_state.get("insights", []),
                    user_preferences = {}
                )

                file_path = agent.save_as_pdf(
                    st.session_state.report,
                    file_path = "business_report.pdf"
                )

                with open(file_path, "rb") as f:
                    st.download_button(
                        label="Download Report PDF",
                        data=f,
                        file_name="business_report.pdf",
                        mime="application/pdf"
                    )

                    
