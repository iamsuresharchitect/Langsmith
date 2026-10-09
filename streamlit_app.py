"""
Streamlit chat UI for Claude, traced with LangSmith.

Setup:
    pip install langchain langchain-anthropic langsmith python-dotenv streamlit
    Keep your .env file next to this script:
        ANTHROPIC_API_KEY=sk-ant-...
        LANGSMITH_TRACING=true
        LANGSMITH_API_KEY=lsv2_...
        LANGSMITH_PROJECT=nutanix-trace
Run:
    streamlit run streamlit_app.py
Traces: smith.langchain.com -> Projects -> nutanix-trace (grouped per chat under Threads)
"""

import os
import uuid

from dotenv import load_dotenv

load_dotenv()  # loads ANTHROPIC_API_KEY and LANGSMITH_* from .env

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

PROJECT = os.getenv("LANGSMITH_PROJECT", "nutanix-trace")

# ---- Page setup -------------------------------------------------------------
st.set_page_config(page_title="Infra Trainer Chat", page_icon="💬")
st.title("💬 Infra Trainer Chat")
st.caption(f"Powered by Claude · traced to LangSmith project **{PROJECT}**")

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("ANTHROPIC_API_KEY not found. Add it to your .env file and restart.")
    st.stop()

# ---- Session state ----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str}
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ---- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["claude-haiku-5-5", "claude-sonnet-5-5"])
    max_tokens = st.slider("Max tokens", 256, 4096, 1024, step=256)

    st.divider()
    quiz_clicked = st.button(
        "📝 Quiz me on the last answer",
        use_container_width=True,
        disabled=not any(m["role"] == "assistant" for m in st.session_state.messages),
    )
    if st.button("🗑️ New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    st.markdown(f"**Tracing:** {'🟢 On' if tracing else '🔴 Off'}")
    st.caption(f"Thread ID: `{st.session_state.thread_id[:8]}…`")

# ---- Model and chains -------------------------------------------------------
llm = ChatAnthropic(model=model, max_tokens=max_tokens)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior IT infrastructure trainer specialising in Nutanix, "
     "VMware, Windows Server, backup and cloud. Be concise and practical. "
     "Use short bullet points where they help."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])
chat_chain = chat_prompt | llm | StrOutputParser()

quiz_prompt = ChatPromptTemplate.from_template(
    "Based on this explanation, write one multiple-choice question "
    "with 4 options (A-D) and mark the correct answer at the end:\n\n{explanation}"
)
quiz_chain = quiz_prompt | llm | StrOutputParser()


def to_lc_history(messages):
    """Convert stored chat messages to LangChain message objects."""
    return [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in messages
    ]


def trace_config(run_name: str) -> dict:
    """Run name, tags and thread metadata so LangSmith groups each chat as a thread."""
    return {
        "run_name": run_name,
        "tags": ["nutanix-trace", "streamlit", model],
        "metadata": {"thread_id": st.session_state.thread_id},
    }


# ---- Show chat history ------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Quiz button ------------------------------------------------------------
if quiz_clicked:
    last_answer = next(
        m["content"] for m in reversed(st.session_state.messages) if m["role"] == "assistant"
    )
    with st.chat_message("assistant"):
        try:
            quiz = st.write_stream(
                quiz_chain.stream({"explanation": last_answer}, config=trace_config("quiz"))
            )
            st.session_state.messages.append({"role": "assistant", "content": quiz})
        except Exception as e:
            st.error(f"Error from Claude: {e}")

# ---- Chat input -------------------------------------------------------------
if question := st.chat_input("Ask about Nutanix, VMware, backups..."):
    with st.chat_message("user"):
        st.markdown(question)

    history = to_lc_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(
                chat_chain.stream(
                    {"history": history, "question": question},
                    config=trace_config("chat_turn"),
                )
            )
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.session_state.messages.pop()  # drop the unanswered question
            st.error(f"Error from Claude: {e}")
