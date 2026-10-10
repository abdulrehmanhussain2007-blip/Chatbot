import os
import uuid
import inspect
import tempfile
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing from your .env file.")

MAX_HISTORY_MESSAGES = 20  # only the last N messages are sent to the model

# ---------------------------------------------------------------------------
# Model, prompt, memory
# ---------------------------------------------------------------------------
model = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key,
    temperature=0.4,
)

store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant specializing in Machine Learning, "
        "Deep Learning, Python, and Data Science. Give clear, "
        "beginner-friendly explanations with short examples when useful.",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])


def trim_history(inputs: dict) -> dict:
    """Keep requests small: send only the most recent messages to the model."""
    inputs["history"] = inputs["history"][-MAX_HISTORY_MESSAGES:]
    return inputs


chain = RunnableLambda(trim_history) | prompt | model

chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_text(content) -> str:
    """Normalize message content (str or list of blocks) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def respond(message, ui_history, session_id):
    """Stream the answer token by token into the chat window."""
    ui_history = list(ui_history or [])

    if not message or not message.strip():
        yield "", ui_history
        return

    ui_history.append({"role": "user", "content": message})
    ui_history.append({"role": "assistant", "content": ""})
    yield "", ui_history  # show the user's message immediately

    config = {"configurable": {"session_id": session_id}}
    partial = ""

    try:
        for chunk in chatbot.stream({"input": message}, config=config):
            partial += to_text(chunk.content)
            ui_history[-1]["content"] = partial
            yield "", ui_history

        if not partial.strip():
            ui_history[-1]["content"] = "⚠️ The model returned an empty response. Please try again."
    except Exception as e:
        ui_history[-1]["content"] = f"⚠️ Error: {e}"

    yield "", ui_history


def save_chat(session_id):
    """Write the conversation to a .txt file using the LangChain store."""
    history = store.get(session_id)
    if history is None or not history.messages:
        gr.Warning("No conversation to save yet.")
        return None

    now = datetime.now()
    folder = tempfile.mkdtemp()
    file_path = os.path.join(folder, f"chat_history_{now:%Y%m%d_%H%M%S}.txt")
    roles = {"human": "USER", "ai": "ASSISTANT"}

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"AI Learning Assistant - chat saved {now:%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 60 + "\n\n")
        for msg in history.messages:
            f.write(f"{roles.get(msg.type, msg.type.upper())}:\n{to_text(msg.content)}\n\n")

    return file_path


def clear_chat(session_id):
    store.pop(session_id, None)  # free old session memory
    return [], "", str(uuid.uuid4())


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
# Gradio <6 needs type="messages"; Gradio 6+ removed that argument.
CHATBOT_KWARGS = {"label": "Conversation", "height": 450}
if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
    CHATBOT_KWARGS["type"] = "messages"

with gr.Blocks(title="AI Learning Assistant") as demo:
    gr.Markdown("# 🤖 AI Learning Assistant")

    session_id = gr.State(value=lambda: str(uuid.uuid4()))

    chatbot_ui = gr.Chatbot(**CHATBOT_KWARGS)  # created INSIDE Blocks

    message = gr.Textbox(
        placeholder="Ask me anything about AI... (press Enter to send)",
        label="Your message",
    )

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        save_btn = gr.Button("Download Chat")
        clear_btn = gr.Button("Clear Chat")

    download_file = gr.File(label="Your saved conversation")

    send_btn.click(respond, [message, chatbot_ui, session_id], [message, chatbot_ui])
    message.submit(respond, [message, chatbot_ui, session_id], [message, chatbot_ui])

    save_btn.click(save_chat, inputs=[session_id], outputs=[download_file])

    clear_btn.click(
        save_chat,
        inputs=[session_id],
        outputs=[download_file],
    ).then(
        clear_chat,
        inputs=[session_id],
        outputs=[chatbot_ui, message, session_id],
    )

if __name__ == "__main__":
    demo.launch()