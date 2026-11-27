import base64
import binascii
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Attachment(BaseModel):
    """An inline document or image supplied with a single task."""

    model_config = ConfigDict(hide_input_in_errors=True)
    mime_type: Literal["image/png", "image/jpeg", "image/webp", "application/pdf"]
    data: str = Field(min_length=4, max_length=4 * 1024 * 1024, repr=False)

    @field_validator("data")
    @classmethod
    def valid_base64(cls, data: str) -> str:
        try:
            base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error):
            raise ValueError("Attachment data must be standard base64.") from None
        return data
