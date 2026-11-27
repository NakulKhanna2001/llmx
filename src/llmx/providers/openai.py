from .chat import ChatProvider

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIProvider(ChatProvider):
    name = "openai"
    default_model = "gpt-6-luna"
    api_url = OPENAI_API_URL
    models_url = "https://api.openai.com/v1/models"
