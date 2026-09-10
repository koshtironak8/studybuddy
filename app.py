import os
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
import streamlit as st

st.set_page_config(page_title="StudyBuddy", page_icon="📚")

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY is missing. Please set it in your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

SYSTEM_PROMPT = """Role: Friendly tutor for Indian college students.
Task: Explain academic concepts simply and help students learn effectively.
Context: Assisting Indian college students with their studies.
Rules:
- Use simple English.
- Keep explanations under 150 words.
- Provide exactly one relatable example.
- If you are unsure, say "I'm not sure".
- Redirect off-topic questions back to studies.
- End with one check question."""

st.title("StudyBuddy 📚")
st.caption("Your friendly AI study partner for simple explanations and practice quizzes.")

quiz_mode = st.sidebar.toggle("Quiz mode")
if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

system_instruction = SYSTEM_PROMPT + ("\n- After explaining, ask 3 MCQs one at a time." if quiz_mode else "")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message("assistant" if msg.role == "model" else "user"):
        st.write(msg.parts[0].text)

prompt = st.chat_input("Ask StudyBuddy a question...")
if prompt:
    st.session_state.messages.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))
    with st.chat_message("user"):
        st.write(prompt)
    try:
        with st.spinner("Thinking..."):
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=st.session_state.messages,
                config=types.GenerateContentConfig(system_instruction=system_instruction),
            )
        st.session_state.messages.append(types.Content(role="model", parts=[types.Part.from_text(text=response.text)]))
        with st.chat_message("assistant"):
            st.write(response.text)
    except errors.APIError as e:
        st.session_state.messages.pop()
        if e.code in (400, 401, 403):
            st.error("Invalid API key or unauthorized request. Please check your GEMINI_API_KEY.")
        elif e.code == 429:
            st.error("Rate limit reached. Please wait a moment and try again.")
        else:
            st.error(f"Gemini API error ({e.code}): {e.message}")
