from typing import Dict, Any, Type
from .base import BaseRNN
from .simple_rnn import SimpleRNN
from .gru import GRU
from .lstm import LSTM

class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, Type[BaseRNN]] = {}
        self.register('simple_rnn', SimpleRNN)
        self.register('gru', GRU)
        self.register('lstm', LSTM)

    def register(self, name: str, model_class: Type[BaseRNN]):
        self._models[name.lower()] = model_class

    def create_model(self, model_name: str, config: Dict[str, Any]) -> BaseRNN:
        model_name = model_name.lower()
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model_class = self._models[model_name]
        return model_class(config)

_registry = ModelRegistry()

def create_model(model_name: str, config: Dict[str, Any]) -> BaseRNN:
    return _registry.create_model(model_name, config)