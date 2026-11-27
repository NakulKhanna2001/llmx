from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..attachments import Attachment


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

    def complete_with_attachments(
        self, prompt: str, attachments: list[Attachment], model: str | None = None,
    ) -> ProviderResult:
        return ProviderResult(
            output="", success=False, provider=self.name, model=model or self.default_model,
            error_message="This adapter does not accept attachments.",
        )

    @abstractmethod
    def validate(self) -> bool:
        pass
