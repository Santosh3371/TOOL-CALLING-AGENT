import hashlib

import httpx
import streamlit as st
from google.genai.errors import APIError

import agent_core
from agent_core import GeminiInteractionError
from tools import AssignmentDataError

st.set_page_config(page_title="Student Study Planner", page_icon="📚")
st.title("📚 Student Study Planner")
st.caption(
    "Enter your own Gemini API key to use the planner. The key is used for Gemini "
    "requests in this session and is not written to project files or displayed."
)

api_key = st.sidebar.text_input(
    "Your Gemini API key",
    type="password",
    help="Get a free key at aistudio.google.com.",
)
if st.sidebar.button("Clear chat"):
    st.session_state.pop("chat", None)
    st.session_state.pop("history", None)

if not api_key:
    st.info("Paste your Gemini API key in the sidebar to start.")
    st.stop()

graph_cache_key = (
    agent_core.GEMINI_MODEL
    + ":"
    + hashlib.sha256(api_key.encode("utf-8")).hexdigest()
)
if "chat" not in st.session_state:
    st.session_state.chat = []
    st.session_state.history = []
if st.session_state.get("graph_cache_key") != graph_cache_key:
    st.session_state.graph = agent_core.build_graph(agent_core.make_llm(api_key))
    st.session_state.graph_cache_key = graph_cache_key

for role, text, steps, stopped in st.session_state.chat:
    with st.chat_message(role):
        st.write(text)
        if steps:
            with st.expander(
                f"Agent steps ({len(steps)} tool call{'s' if len(steps) != 1 else ''})"
            ):
                for number, step in enumerate(steps, 1):
                    st.markdown(f"**{number}. {step['tool']}** with `{step['args']}`")
                    st.caption(f"Result: {step['result'][:300]}")
        if stopped:
            st.warning("The step limit stopped this run.")

question = st.chat_input("Ask your agent")
if question:
    st.session_state.chat.append(("user", question, None, False))
    try:
        answer, steps, stopped, st.session_state.history = agent_core.run_agent(
            st.session_state.graph,
            st.session_state.history,
            question,
        )
    except (APIError, httpx.RequestError, GeminiInteractionError) as error:
        if isinstance(error, httpx.TimeoutException):
            answer = "Gemini took too long to respond. Please try again."
        elif isinstance(error, APIError):
            answer = (
                f"Gemini rejected the request ({type(error).__name__}). Check "
                "your model access and API key, then try again."
            )
        else:
            answer = (
                f"Gemini could not complete the request ({type(error).__name__}). "
                "Check your connection and try again."
            )
        steps, stopped = [], False
    except AssignmentDataError as error:
        answer = f"Assignment data error: {error}"
        steps, stopped = [], False
    st.session_state.chat.append(("assistant", answer, steps, stopped))
    st.rerun()
