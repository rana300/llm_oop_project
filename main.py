from conversation import Conversation
from exceptions import InvalidPromptError
from manager import ModelManager
from models import GPTModel, LanguageModel, LlamaModel
from pricing import PricingPlan


def main():
    gpt = GPTModel("opus", "openai", 800, 0.7, "api-key", PricingPlan("basic", 0.5))
    llama = LlamaModel("Local Llama", "Local", 2048, 0.7, "model_path", "4bit", PricingPlan("local", 0.0))

    print(gpt == llama)
    print(len(llama))
    print(repr(gpt))
    print(LanguageModel.get_total_models())

    for model in [gpt, llama]:
        print(model.generate_response("What is Python"))

    manager = ModelManager()
    manager.add_model(gpt)
    manager.add_model(llama)
    manager.list_models()
    print(manager.select_model("openai", 800))

    print(gpt.calculate_request_cost("Hello how are you"))

    conversation = Conversation("conv-1", gpt)
    try:
        conversation.send_message("Hello")
        conversation.send_message("   ")
    except InvalidPromptError as e:
        print(f"Error sending message: {e}")
    conversation.display_history()


if __name__ == "__main__":
    main()
