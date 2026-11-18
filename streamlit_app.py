import pandas as pd
import streamlit as st

from conversation import Conversation
from exceptions import InvalidPromptError
from manager import ModelManager
from models import GPTModel, LanguageModel, LlamaModel
from pricing import PricingPlan

st.set_page_config(page_title="LLM Manager", page_icon="🤖", layout="wide")

if "manager" not in st.session_state:
    st.session_state.manager = ModelManager()
    st.session_state.conversations = {}

manager = st.session_state.manager
conversations = st.session_state.conversations

with st.sidebar:
    st.header("Add model")
    model_type = st.radio("Type", ["GPT", "Llama"], horizontal=True)

    with st.form("add_model_form"):
        name = st.text_input("Model name", value="opus" if model_type == "GPT" else "Local Llama")
        provider = st.text_input("Provider", value="openai" if model_type == "GPT" else "Local")
        max_tokens = st.number_input("Max tokens", min_value=1, value=800, step=100)
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

        if model_type == "GPT":
            api_key = st.text_input("API key", type="password")
        else:
            model_path = st.text_input("Model path", value="model_path")
            quantization = st.text_input("Quantization", value="4bit")

        plan_name = st.text_input("Plan name", value="basic")
        price = st.number_input("Price per 1000 tokens", min_value=0.0, value=0.5, step=0.1)
        submitted = st.form_submit_button("Add")

    if submitted:
        if not name.strip():
            st.error("Model name is required")
        elif manager.get_model(name):
            st.error("A model with this name already exists")
        else:
            try:
                plan = PricingPlan(plan_name, price)
                if model_type == "GPT":
                    model = GPTModel(name, provider, int(max_tokens), temperature, api_key, plan)
                else:
                    model = LlamaModel(name, provider, int(max_tokens), temperature,
                                       model_path, quantization, plan)
                manager.add_model(model)
                st.success(f"Added {name}")
            except ValueError as e:
                st.error(str(e))

    st.divider()
    st.caption(LanguageModel.get_total_models())

st.title("LLM Model Manager")

if not manager.models:
    st.info("Add a model from the sidebar to get started.")
    st.stop()

names = [m.model_name for m in manager.models.values()]
tab_chat, tab_models, tab_cost, tab_select = st.tabs(["Chat", "Models", "Cost", "Select"])

with tab_chat:
    chosen = st.selectbox("Model", names, key="chat_model")
    model = manager.get_model(chosen)
    if chosen not in conversations:
        conversations[chosen] = Conversation(f"conv-{chosen}", model)
    conversation = conversations[chosen]

    for message in conversation.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Type a message")
    if prompt is not None:
        try:
            conversation.send_message(prompt)
            st.rerun()
        except InvalidPromptError as e:
            st.error(f"Error sending message: {e}")

    if st.button("Clear conversation"):
        conversation.messages.clear()
        st.rerun()

    if hasattr(model, "logs"):
        with st.expander("Logs"):
            for line in model.logs:
                st.text(f"[LOG] {line}")

with tab_models:
    rows = [
        {
            "name": m.model_name,
            "type": type(m).__name__,
            "provider": m.provider,
            "max_tokens": len(m),
            "temperature": m.temperature,
            "active": m.is_active,
        }
        for m in manager.models.values()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    selected = st.selectbox("Select model", names, key="edit_model")
    target = manager.get_model(selected)

    col1, col2 = st.columns(2)
    with col1:
        new_temp = st.number_input("New temperature", value=float(target.temperature), step=0.1)
        if st.button("Apply"):
            try:
                target.temperature = new_temp
                st.rerun()
            except ValueError as e:
                st.error(str(e))
    with col2:
        st.write("")
        st.write("")
        if st.button("Delete model", type="primary"):
            manager.remove_model(target)
            conversations.pop(selected, None)
            st.rerun()

    with st.expander("repr / str"):
        st.code(repr(target))
        st.code(str(target))

with tab_cost:
    chosen = st.selectbox("Model", names, key="cost_model")
    text = st.text_area("Prompt")
    if st.button("Calculate cost"):
        try:
            result = manager.get_model(chosen).calculate_request_cost(text)
            c1, c2 = st.columns(2)
            c1.metric("Input tokens", result["tokens"])
            c2.metric("Cost", f"{result['cost']:.5f}")
        except ValueError as e:
            st.error(str(e))

with tab_select:
    providers = sorted({m.provider for m in manager.models.values()})
    provider_choice = st.selectbox("Provider", providers)
    required = st.number_input("Required tokens", min_value=1, value=500, step=100)
    if st.button("Find model"):
        found = manager.select_model(provider_choice, int(required))
        if found:
            st.success(f"Matching model: {found.model_name}")
            st.code(repr(found))
        else:
            st.warning("No model matches these requirements")
