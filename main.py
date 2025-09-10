from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from pandasai import Agent
from pandasai.llm import OpenAI
import matplotlib.pyplot as plt
import pathlib

# Load environment variables from the .env file
load_dotenv()

# Set the page configuration for a cleaner look
st.set_page_config(page_title="Prompt-Driven Analysis Tool", page_icon=":bar_chart:", layout="wide")

# Add some custom styling using Streamlit's markdown with a dark background
st.markdown("""
    <style>
    /* General Page Styling */
    .main {
        background: #000;  /* Set background color to black */
    }
    
    /* Text Area and Button Styling */
    .stTextInput > div > div > textarea {
        border-radius: 10px;
        border: 1px solid #1DB954;
        padding: 10px;
        animation: fadeInAnimation ease 3s;
        color: #fff;  /* Set text color to white */
        background: #333;  /* Set text area background to dark gray */
    }
    .stButton > button {
        background: linear-gradient(to right, #1DB954, #1AAE4A);
        color: white;
        border-radius: 12px;
        padding: 10px 15px;
        border: none;
        box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.3);
    }

    /* Info Box Styling */
    .stAlert {
        border-left: 5px solid #1DB954;
        border-radius: 5px;
    }

    /* Header Animation */
    @keyframes fadeInAnimation {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    
    h1 {
        text-align: center;
        font-weight: bold;
        color: #1DB954;  /* Bright green color for the main heading */
        animation: fadeInAnimation ease 2s;
    }
    
    h2 {
        text-align: center;
        color: #ffffff;  /* White for subheadings */
    }

    /* Chart Styling */
    img {
        border: 5px solid #1DB954;
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Retrieve the OpenAI API key from the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Display an error message if the API key is missing
if OPENAI_API_KEY is None:
    st.error("OpenAI API key not found. Please check your .env file.")
else:
    st.title("Prompt-Driven Analysis Tool")
    st.subheader("Analyze your CSV data with ease :bar_chart:")

    # Add a decorative info box
    st.info("Upload your dataset in CSV format to get started. Then, enter a prompt to ask questions or generate analysis.")

    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data", df.head(3))

        # Initialize the OpenAI LLM with the API key
        llm = OpenAI(api_key=OPENAI_API_KEY)

        # Initialize the agent with the uploaded DataFrame
        agent = Agent(dfs=[df])

        prompt = st.text_area("Enter your prompt:", placeholder="e.g., 'What are the main trends in the data?'")

        if st.button("Generate Analysis"):
            if prompt:
                with st.spinner("Generating an answer, please wait..."):
                    # Check and create the directory for saving plots
                    output_dir = pathlib.Path("exports/charts")
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Use the agent to generate the analysis with the prompt
                    response = agent.chat(prompt)

                    # If the response includes a command to plot, handle it
                    if 'plot' in prompt.lower():
                        # Extract necessary data from the DataFrame for plotting
                        # Example: Assuming 'price' and 'recommendations' are in the dataframe
                        if 'price' in df.columns and 'recommendations' in df.columns:
                            plt.figure(figsize=(10, 5))
                            plt.plot(df['price'], df['recommendations'], marker='o', color='#1DB954')
                            plt.title('Price vs Recommendations', fontsize=16, color='#1DB954')
                            plt.xlabel('Price', fontsize=14, color='#1DB954')
                            plt.ylabel('Recommendations', fontsize=14, color='#1DB954')
                            plt.grid()

                            # Save the plot to the directory
                            chart_path = output_dir / "temp_chart.png"
                            plt.savefig(chart_path)
                            plt.close()
                            
                            # Convert Path object to string before displaying
                            st.image(str(chart_path))
                        else:
                            st.warning("Columns 'price' and 'recommendations' not found in the dataset.")
                    else:
                        st.write("### Analysis Result", response)
            else:
                st.warning("Please enter a prompt.")

    # Footer decoration
    st.markdown("---")
    st.markdown("**:information_source: Tip:** Use prompts like 'Plot a graph of column X vs column Y' to visualize trends or 'What is the average value of column Z?' to get insights.")
    st.markdown("Created by Kaushal | Powered by OpenAI & Streamlit")
