from ModelComparison.ModelWrappers.ChatGPTWrapper import ChatGPTWrapper
from ModelComparison.ModelWrappers.DeepSeekWrapper import DeepSeekWrapper
from ModelComparison.ModelWrappers.ZientCoreWrapper import ZientCoreWrapper
from ModelComparison.ModelWrappers.CommandRPlusWrapper import CommandRPlusWrapper
from ModelComparison.ModelWrappers.LLaMaWrapper import LLaMAaWrapper
from ModelComparison.ModelWrappers.MistralWrapper import MistralWrapper
from ModelComparison.ModelWrappers.MixtralWrapper import MixtralWrapper
from ModelComparison.ModelWrappers.GemmaWrapper import GemmaWrapper
from ModelComparison.ModelWrappers.FinBERTWrapper import FinBERTWrapper


MODEL_REGISTRY = {
    #"chatgpt": ChatGPTWrapper,
    "deepseek": DeepSeekWrapper,
    "zient_core": ZientCoreWrapper,
    "command_r_plus": CommandRPlusWrapper,
    "llama": LLaMAaWrapper,
    "mistral": MistralWrapper,
    "mixtral": MixtralWrapper,
    "gemma": GemmaWrapper,
    "finbert": FinBERTWrapper
}


def get_available_models():
    return list(MODEL_REGISTRY.keys())


def get_model_instance(name: str, force_download: bool = False):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        available = ", ".join(get_available_models())
        raise ValueError(f"Unknown model name: '{name}'\nAvailable models: {available}")
    
    cls = MODEL_REGISTRY[name]

    if name in {"chatgpt", "zient_core"}:
        return cls()  # these don't support force_download
    else:
        return cls(force_download=force_download)
