from models import LanguageModel


class ModelManager:
    def __init__(self, models=None):
        self.models = dict(models) if models is not None else {}

    def add_model(self, model):
        if self.get_model(model.model_name) is not None:
            raise ValueError("Model name already exists")
        self.models[model.model_name] = model

    def remove_model(self, model):
        if isinstance(model, str):
            model = self.get_model(model)
        if model is None:
            return
        del self.models[model.model_name]
        LanguageModel.total_models -= 1

    def get_model(self, model_name):
        return self.models.get(model_name)

    def list_models(self):
        for model in self.models.values():
            print(model.model_name)

    def select_model(self, provider, required_tokens):
        for model in self.models.values():
            if model.provider == provider and model.max_tokens >= required_tokens:
                return model
        return None