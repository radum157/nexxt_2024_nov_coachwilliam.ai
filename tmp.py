import streamlit as st
from openai import AzureOpenAI
import os
from test_Df_Agent import *
from fpdf import FPDF
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def safe_text(text):
    # Encode in latin-1 with 'ignore' to remove unsupported characters
    return text.encode('latin-1', 'ignore').decode('latin-1')

def text_to_pdf(text):
    global cnt
    cnt += 1
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt = safe_text(text))
    try:
        return pdf.output(dest="S").encode("utf-8")
    except:
        return pdf.output(dest="S")
    
st.markdown(
    """
    <style>
        /* Set background color for the app */
        section.main {
            background-color: #d3d3d3;
        }
    </style>
    """, 
    unsafe_allow_html=True
)

# Display Raiffeisen Logo
current_dir = os.getcwd()
img_path = os.path.join(current_dir, "Raiffeisen/PNGs/RBI-Logo-Bank-Hz-Yell-Neg-RGB.png")
st.image(img_path, width=175)

client = AzureOpenAI(
    azure_endpoint="https://rbro-openai-hackatlon.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview",  # insert the provided endpoint here 
    api_key=OPENAI_API_KEY,  # insert the provided api key here  
    api_version="2024-08-01-preview"
)

# Page Title
st.title("Chat with William: Get Smart Responses Instantly 🧠")

# Initialize session state for chat history and topics
if 'history' not in st.session_state:
    st.session_state.history = []
if 'topics' not in st.session_state:
    st.session_state.topics = set()

global cnt
cnt = 0

# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = triage_agent  # Initialize the agent

for message in st.session_state.messages:
    if isinstance(message, dict):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Create PDF from assistant_message
            try:
                pdf = text_to_pdf(message["content"])
            except:
                pass
                
            # Create download button
            try:
                st.download_button(
                    label="Download as PDF",
                    data=pdf,
                    file_name=f"{cnt}.pdf",
                    mime="application/pdf"
                )
            except:
                pass


# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": str(prompt)})

    # Run the assistant's response logic
    response = run_full_turn(st.session_state.agent, st.session_state.messages)
    st.session_state.agent = response.agent  # Update agent if needed

    # Extend messages with new ones from the assistant
    st.session_state.messages.extend(response.messages)

    # Display the assistant's last response
    assistant_message = response.messages[-1].content  # Access content with dot notation
    with st.chat_message("assistant"):
        st.markdown(assistant_message)
        # Create PDF from assistant_message
        pdf = text_to_pdf(assistant_message)
            
        # Create download button
        try:
            
            st.download_button(
                label="Download as PDF",
                data=pdf,
                file_name=f"{cnt}.pdf",
                mime="application/pdf"
            )
        except:
            pass

    # Append assistant's response to the session state chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
