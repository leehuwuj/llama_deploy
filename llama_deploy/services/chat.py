import asyncio
import json
import logging
from typing import Any, AsyncGenerator

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from llama_deploy.types import ChatMessage, ChatRequest, TaskDefinition

logger = logging.getLogger("uvicorn")


class ChatUIService:
    @staticmethod
    def to_task_definition(chat_request: ChatRequest) -> TaskDefinition:
        """Convert a chat request to a task definition."""
        chat_history = [
            ChatMessage(
                role=message.role,
                content=message.content,
            )
            for message in chat_request.messages
        ]
        last_message = chat_request.messages[-1]
        if not last_message:
            raise HTTPException(status_code=400, detail="Last message is required")

        # task.input is equal to workflow.run() arguments
        # we need to serialize the chat history and last message
        chat_history_str = [
            ChatMessage(
                role=message.role,
                content=message.content,
            ).model_dump()
            for message in chat_history
        ]
        input = json.dumps(
            {
                "user_msg": last_message.content,
                "chat_history": chat_history_str,
            }
        )

        return TaskDefinition(input=input)


class VercelStreamResponse(StreamingResponse):
    """
    Converts preprocessed events into Vercel-compatible streaming response format.
    """

    TEXT_PREFIX = "0:"
    DATA_PREFIX = "8:"
    ERROR_PREFIX = "3:"

    def __init__(
        self,
        stream_generator: AsyncGenerator[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            content=self.stream_generator(stream_generator),
            *args,
            **kwargs,
        )

    @classmethod
    async def stream_generator(
        cls, stream_generator: AsyncGenerator[str, Any]
    ) -> AsyncGenerator[str, Any]:
        """Generate Vercel-formatted content from preprocessed events."""
        stream_started = False
        try:
            async for event in stream_generator:
                if not stream_started:
                    # Start the stream with an empty message
                    stream_started = True
                    yield cls.convert_text("")

                # Handle different types of events
                # This is a simplified from LlamaIndexServer's stream
                if "delta" in event.keys():
                    yield cls.convert_text(event["delta"])
                else:
                    yield cls.convert_data(event)

        except asyncio.CancelledError:
            logger.warning("Client cancelled the request!")
            # TODO: cancel the task
        except Exception as e:
            logger.error(f"Error in stream response: {e}")
            # TODO: cancel the task
            yield cls.convert_error(str(e))

    @classmethod
    def convert_text(cls, token: str) -> str:
        """Convert text event to Vercel format."""
        # Escape newlines and double quotes to avoid breaking the stream
        token = json.dumps(token)
        return f"{cls.TEXT_PREFIX}{token}\n"

    @classmethod
    def convert_data(cls, data: dict[str, Any]) -> str:
        """Convert data event to Vercel format."""
        data_str = json.dumps(data)
        return f"{cls.DATA_PREFIX}[{data_str}]\n"

    @classmethod
    def convert_error(cls, error: str) -> str:
        """Convert error event to Vercel format."""
        error_str = json.dumps(error)
        return f"{cls.ERROR_PREFIX}{error_str}\n"
