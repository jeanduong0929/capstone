import streamlit as st
from agent import convo_agent_executor
from PIL import Image

# Renders the page title "RevServe" along with Revature logo
st.set_page_config(page_title="RevServe")
revatureLogo = Image.open("assets/revature-logo.png")
st.image(revatureLogo, width=100)
st.title("RevServe")


# This function takes the user input and anonymizes it, then sends the anonymized user input to the conversational agent
# This is the main processor function that stitches everything together
def processUserInput(userInput):
    st.session_state["chat-history"].append({"content": userInput, "role": "user"})
    agentResponse = convo_agent_executor.run(input=userInput)
    st.session_state["chat-history"].append({"content": agentResponse, "role": "ai"})


# Checks Streamlit's session state and initializes the 'chat-history' if there is none
if "chat-history" not in st.session_state:
    st.session_state["chat-history"] = [
        {
            "content": "Hi, I'm RevServe. I help provide talent enablement solutions! How can I help today?",
            "role": "ai",
        }
    ]

userInput = st.chat_input("Message:", key="userInput")
if userInput:
    processUserInput(userInput)

if st.session_state["chat-history"]:
    for i in range(0, len(st.session_state["chat-history"])):
        msgObj = st.session_state["chat-history"][i]
        st.chat_message(msgObj["role"]).write(msgObj["content"])
