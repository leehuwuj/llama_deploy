import re
from enum import Enum
from typing import Any, List, Optional

from llama_index.core.llms import MessageRole
from pydantic import BaseModel, field_validator


class StatusEnum(Enum):
    HEALTHY = "Healthy"
    UNHEALTHY = "Unhealthy"
    DOWN = "Down"


class Status(BaseModel):
    status: StatusEnum
    status_message: str
    max_deployments: int | None = None
    deployments: list[str] | None = None


class DeploymentDefinition(BaseModel):
    name: str


class ChatUIMessage(BaseModel):
    role: MessageRole
    content: str
    annotations: Optional[List[Any]] = None


class ChatRequest(BaseModel):
    """
    The request to the chat API.
    """

    id: str  # see https://ai-sdk.dev/docs/reference/ai-sdk-ui/use-chat#id - constant for the same chat session
    messages: List[ChatUIMessage]

    @field_validator("messages")
    def validate_messages(cls, v: List[ChatUIMessage]) -> List[ChatUIMessage]:
        if v[-1].role != MessageRole.USER:
            raise ValueError("Last message must be from user")
        return v

    @field_validator("id")
    def validate_id(cls, v: str) -> str:
        if re.search(r"[^a-zA-Z0-9_-]", v):
            raise ValueError("ID contains special characters")
        return v
