import os
import streamlit as st
from groq import Groq

# -----------------------------
# Page config (Professional UI)
# -----------------------------
st.set_page_config(
    page_title="Groq Chatbot",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Groq Chatbot")
st.caption("Fast, clean, and professional chatbot using Groq + Streamlit")

# -----------------------------
# Load API Key (Streamlit Secrets or Env)
# -----------------------------
# Streamlit Cloud: st.secrets["GROQ_API_KEY"]
# Local: export GROQ_API_KEY="..."
api_key = None
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("API key not found. Add GROQ_API_KEY in Streamlit Secrets or environment variables.")
    st.stop()

client = Groq(api_key=api_key)

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    model = st.selectbox(
        "Model",
        options=[
            "openai/gpt-oss-120b",
            # agar tumhare paas aur models available hon to yahan add kar do
        ],
        index=0
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    max_tokens = st.slider("Max tokens", 64, 2048, 512, 64)

    style = st.selectbox(
        "Assistant style",
        ["Professional", "Friendly", "Teacher", "Concise"],
        index=0
    )

    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

# -----------------------------
# System prompt presets
# -----------------------------
SYSTEM_PROMPTS = {
    "Professional": (
        "You are a professional assistant. "
        "Be clear, structured, and practical. "
        "Ask a short follow-up question if needed."
    ),
    "Friendly": (
        "You are a friendly, helpful assistant. "
        "Use a warm tone, simple words, and short paragraphs."
    ),
    "Teacher": (
        "You are a patient teacher. "
        "Explain step-by-step with examples, and check understanding."
    ),
    "Concise": (
        "You are a concise assistant. "
        "Give short, direct answers with bullet points when useful."
    )
}

system_prompt = SYSTEM_PROMPTS[style]

# -----------------------------
# Session state: chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show existing chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# Helper: trim history (avoid huge context)
# -----------------------------
def build_messages(history, user_text):
    """
    Converts Streamlit history into Groq/OpenAI format.
    Keeps last N messages to control context size.
    """
    MAX_TURNS = 12  # last 12 messages (user+assistant) => adjustable
    trimmed = history[-MAX_TURNS:]

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(trimmed)
    messages.append({"role": "user", "content": user_text})
    return messages

# -----------------------------
# Chat input
# -----------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user msg
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ Thinking...")

        try:
            messages = build_messages(st.session_state.messages[:-1], user_input)

            # Non-streaming response (simple + stable)
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            assistant_text = completion.choices[0].message.content
            placeholder.markdown(assistant_text)

            st.session_state.messages.append({"role": "assistant", "content": assistant_text})

        except Exception as e:
            placeholder.markdown("❌ Something went wrong. Please try again.")
            st.error(str(e))
