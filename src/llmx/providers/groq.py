from .chat import ChatProvider

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(ChatProvider):
    name = "groq"
    default_model = "openai/gpt-oss-20b"
    api_url = GROQ_API_URL
    models_url = "https://api.groq.com/openai/v1/models"
