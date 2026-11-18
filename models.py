from abc import ABC, abstractmethod

from exceptions import InvalidPromptError
from tokenizer import Tokenizer


class LoggingMixin:
    def log(self, message):
        if not hasattr(self, "logs"):
            self.logs = []
        self.logs.append(message)
        print(f"[LOG] {message}")


class LanguageModel(ABC):
    total_models = 0

    def __init__(self, model_name, provider, max_tokens, temperature, pricing_plan=None):
        self.model_name = model_name
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.pricing_plan = pricing_plan

        self.tokenizer = Tokenizer()
        self.__is_active = True

        LanguageModel.total_models += 1

    @abstractmethod
    def generate_response(self, prompt):
        pass

    def display_info(self):
        print(f"Model Name: {self.model_name}")
        print(f"Provider: {self.provider}")

    @classmethod
    def get_total_models(cls):
        return f"Total models: {cls.total_models}"

    @staticmethod
    def validate_temperature(temperature):
        if temperature < 0.0 or temperature > 2.0:
            raise ValueError("Temperature must be between 0 and 2")

    @property
    def temperature(self):
        return self.__temperature

    @temperature.setter
    def temperature(self, value):
        self.validate_temperature(value)
        self.__temperature = value

    @property
    def is_active(self):
        return self.__is_active

    def calculate_request_cost(self, prompt):
        if self.pricing_plan is None:
            raise ValueError("Pricing plan not set")
        tokens = self.tokenizer.count_tokens(prompt)
        cost = self.pricing_plan.calculate_cost(tokens)
        return {"tokens": tokens, "cost": cost}

    def _validate_prompt(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise InvalidPromptError("Invalid prompt")

    def __repr__(self) -> str:
        return (
            f"model_name={self.model_name!r}, provider={self.provider}, "
            f"max_tokens={self.max_tokens}, temperature={self.temperature}"
        )

    def __str__(self) -> str:
        return (
            f"LanguageModel(model_name={self.model_name}, provider={self.provider}, "
            f"max_tokens={self.max_tokens}, temperature={self.temperature})"
        )

    def __len__(self):
        return self.max_tokens

    def __eq__(self, other):
        if not isinstance(other, LanguageModel):
            return NotImplemented
        return self.model_name == other.model_name and self.provider == other.provider

    def __hash__(self):
        return hash((self.model_name, self.provider))


class GPTModel(LoggingMixin, LanguageModel):
    def __init__(self, model_name, provider, max_tokens, temperature, api_key, pricing_plan=None):
        super().__init__(model_name, provider, max_tokens, temperature, pricing_plan)
        self.api_key = api_key

    def generate_response(self, prompt):
        self._validate_prompt(prompt)
        self.log(f"sending request to {self.model_name}")
        tokens = self.tokenizer.count_tokens(prompt)
        return f"{self.model_name} response to: {prompt}, input tokens: {tokens}"


class LlamaModel(LoggingMixin, LanguageModel):
    def __init__(self, model_name, provider, max_tokens, temperature,
                 model_path, quantization, pricing_plan=None):
        super().__init__(model_name, provider, max_tokens, temperature, pricing_plan)
        self.model_path = model_path
        self.quantization = quantization

    def generate_response(self, prompt):
        self._validate_prompt(prompt)
        self.log(f"running local model {self.model_name}")
        tokens = self.tokenizer.count_tokens(prompt)
        return f"{self.model_name} response to: {prompt}, input tokens: {tokens}"
