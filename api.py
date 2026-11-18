from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from conversation import Conversation
from exceptions import InvalidPromptError
from manager import ModelManager
from models import GPTModel, LanguageModel, LlamaModel
from pricing import PricingPlan

app = FastAPI(title="LLM OOP Project")
manager = ModelManager()
conversations = {}


class ModelCreate(BaseModel):
    model_type: Literal["gpt", "llama"]
    model_name: str
    provider: str
    max_tokens: int = Field(gt=0)
    temperature: float
    api_key: Optional[str] = None
    model_path: Optional[str] = None
    quantization: Optional[str] = None
    plan_name: str = "basic"
    price_per_1k_tokens: float = 0.0


class TemperatureUpdate(BaseModel):
    temperature: float


class PromptBody(BaseModel):
    prompt: str


def describe(model):
    return {
        "model_name": model.model_name,
        "type": type(model).__name__,
        "provider": model.provider,
        "max_tokens": model.max_tokens,
        "temperature": model.temperature,
        "is_active": model.is_active,
    }


def get_or_404(model_name):
    model = manager.get_model(model_name)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@app.post("/models", status_code=201)
def create_model(body: ModelCreate):
    if manager.get_model(body.model_name):
        raise HTTPException(status_code=409, detail="Model already exists")
    plan = PricingPlan(body.plan_name, body.price_per_1k_tokens)
    try:
        if body.model_type == "gpt":
            model = GPTModel(body.model_name, body.provider, body.max_tokens,
                             body.temperature, body.api_key, plan)
        else:
            model = LlamaModel(body.model_name, body.provider, body.max_tokens,
                               body.temperature, body.model_path, body.quantization, plan)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    manager.add_model(model)
    return describe(model)


@app.get("/models")
def list_models():
    return {"total": LanguageModel.total_models, "models": [describe(m) for m in manager.models]}


@app.get("/models/select")
def select_model(provider: str, required_tokens: int):
    model = manager.select_model(provider, required_tokens)
    if model is None:
        raise HTTPException(status_code=404, detail="No matching model")
    return describe(model)


@app.get("/models/{model_name}")
def get_model(model_name: str):
    return describe(get_or_404(model_name))


@app.patch("/models/{model_name}/temperature")
def update_temperature(model_name: str, body: TemperatureUpdate):
    model = get_or_404(model_name)
    try:
        model.temperature = body.temperature
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return describe(model)


@app.delete("/models/{model_name}", status_code=204)
def delete_model(model_name: str):
    model = get_or_404(model_name)
    manager.remove_model(model)
    conversations.pop(model_name, None)


@app.post("/models/{model_name}/cost")
def request_cost(model_name: str, body: PromptBody):
    model = get_or_404(model_name)
    try:
        return model.calculate_request_cost(body.prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/chat/{model_name}")
def chat(model_name: str, body: PromptBody):
    model = get_or_404(model_name)
    conversation = conversations.setdefault(model_name, Conversation(f"conv-{model_name}", model))
    try:
        response = conversation.send_message(body.prompt)
    except InvalidPromptError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"response": response}


@app.get("/chat/{model_name}")
def history(model_name: str):
    get_or_404(model_name)
    conversation = conversations.get(model_name)
    return {"messages": conversation.messages if conversation else []}
