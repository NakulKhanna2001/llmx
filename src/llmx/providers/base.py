from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProviderResult:
    output: str
    success: bool
    provider: str
    model: str
    latency_ms: int = 0
    error_code: int | None = None
    error_message: str = ""


class BaseProvider(ABC):
    name: str
    default_model: str

    @abstractmethod
    def complete(self, prompt: str, model: str | None = None) -> ProviderResult:
        pass

    @abstractmethod
    def validate(self) -> bool:
        pass
