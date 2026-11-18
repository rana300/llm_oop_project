class PricingPlan:
    def __init__(self, plan_name, price_per_1k_tokens):
        self.plan_name = plan_name
        self.price_per_1k_tokens = price_per_1k_tokens

    def calculate_cost(self, tokens):
        return (tokens / 1000) * self.price_per_1k_tokens
