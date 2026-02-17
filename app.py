# import streamlit as st

# st.title("Hello GenAI 🚀")
# st.write("Streamlit is working!")
import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="GenAI Assistant", page_icon="🤖")
st.title(":streamlit: 🤖 GenAI Assistant ")
st.caption("Memory Enabled • Production Ready")

# -----------------------------
# Session State Setup
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "store" not in st.session_state:
    st.session_state.store = {}

# -----------------------------
# Memory Handler
# -----------------------------
def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]

# -----------------------------
# LLM Setup
# -----------------------------
llm = ChatOllama(
    model="gemma2:2b",
    temperature=0.7,
)

chain = RunnableWithMessageHistory(
    llm,
    get_session_history,
)

# -----------------------------
# Display Previous Messages
# -----------------------------
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# -----------------------------
# Chat Input
# -----------------------------
if prompt := st.chat_input("Ask me anything..."):

    # Show user message
    st.session_state.chat_history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        response = chain.invoke(
            [HumanMessage(content=prompt)],
            config={"configurable": {"session_id": "default"}}
        )
        st.markdown(response.content)

    st.session_state.chat_history.append(("assistant", response.content))

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙ Settings")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.store = {}
        st.rerun()

    st.markdown("---")
    st.write("Model: llama3 (Ollama)")
    st.write("Memory: Enabled")
