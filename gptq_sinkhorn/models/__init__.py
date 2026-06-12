from .base import ModelHandler
from .llama import LlamaHandler
from .opt import OPTHandler


def get_handler(model_type: str) -> ModelHandler:
    handlers = {
        "llama": LlamaHandler,
        "opt": OPTHandler,
    }
    if model_type not in handlers:
        raise ValueError(f"Unknown model type: {model_type}. Supported: {list(handlers)}")
    return handlers[model_type]()


def detect_and_get_handler(model) -> ModelHandler:
    class_name = model.__class__.__name__.lower()
    if "llama" in class_name or "mistral" in class_name or "qwen" in class_name:
        return LlamaHandler()
    elif "opt" in class_name:
        return OPTHandler()
    else:
        raise ValueError(f"Unsupported model class: {model.__class__.__name__}")
