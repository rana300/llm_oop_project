from models import LanguageModel


class ModelManager:
    def __init__(self, models=None):
        self.models = models if models is not None else []

    def add_model(self, model):
        self.models.append(model)

    def remove_model(self, model):
        self.models.remove(model)
        LanguageModel.total_models -= 1

    def get_model(self, model_name):
        for model in self.models:
            if model.model_name == model_name:
                return model
        return None

    def list_models(self):
        for model in self.models:
            print(model.model_name)

    def select_model(self, provider, required_tokens):
        for model in self.models:
            if model.provider == provider and model.max_tokens >= required_tokens:
                return model
        return None
