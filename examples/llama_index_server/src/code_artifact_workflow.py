import time
from typing import Literal, Optional, Union

from llama_index.core.base.llms.types import ChatMessage  # type: ignore
from llama_index.core.memory import Memory
from llama_index.core.prompts import PromptTemplate
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,  # type: ignore
)
from llama_index.llms.openai import OpenAI

# Just use the imports from llama_index_server for now
from llama_index.server.api.utils import get_last_artifact  # type: ignore
from llama_index.server.models import (  # type: ignore
    Artifact,
    ArtifactEvent,
    ArtifactType,
    ChatRequest,
    CodeArtifactData,
    UIEvent,
)
from pydantic import BaseModel, Field


class PlanOutput(BaseModel):
    """
    This is the output of the plan step.
    """

    next_step: Literal["coding", "answering"] = Field(
        description="The next action to take. Only answer if user asking for clarification."
    )
    requirement: Optional[str] = Field(
        description="The requirement for the next step (if next_step is 'coding')"
    )
    language: Optional[Literal["typescript", "python"]] = Field(
        description="The language to use for coding step (if next_step is 'coding')"
    )
    file_name: Optional[str] = Field(
        description="The file name to use for the next step (if next_step is 'coding')"
    )

    def to_llm_content(self) -> str:
        if self.next_step == "coding":
            return f"I need to implement {self.file_name} in {self.language} with the following requirement: \n{self.requirement}\n"
        else:
            return "I need to answer the user's question"


class CodeArtifact(BaseModel):
    language: Literal["typescript", "python"] = Field(
        description="The language of the code"
    )
    file_name: str = Field(description="The file name of the code")
    code: str = Field(description="The code content")


class UIEventData(BaseModel):
    state: Literal["plan", "generate", "completed"]
    requirement: Optional[str] = None


## ==== Workflow Events ====
class ChatStartEvent(StartEvent):
    chat_request: ChatRequest  # TODO: Can use chat messages


class PlanEvent(Event):
    pass


class GenerateArtifactEvent(Event):
    context: str


class SynthesizeAnswerEvent(Event):
    pass


class StreamEvent(Event):
    delta: str


class CodeArtifactWorkflow(Workflow):
    """
    A simple workflow that help generate/update the chat artifact (code, document)
    e.g: Help create a NextJS app.
         Update the generated code with the user's feedback.
         Generate a guideline for the app,...
    """

    llm: OpenAI = OpenAI(model="gpt-4.1")  # Use gpt-4.1 for better code generation

    @step
    async def prepare_chat_history(self, ctx: Context, ev: ChatStartEvent) -> PlanEvent:
        chat_messages = ev.chat_request.messages
        user_msg = chat_messages[-1]
        if not user_msg:
            raise ValueError("Please send a message to start the workflow")

        chat_history = [ev.to_llamaindex_message() for ev in ev.chat_request.messages]
        chat_history.append(ChatMessage(role="user", content=user_msg))
        memory = Memory.from_defaults(chat_history=chat_history)  # type: ignore

        last_artifact_content = get_last_artifact(ev.chat_request)

        await ctx.set("user_msg", user_msg.content)
        await ctx.set("memory", memory)
        await ctx.set("last_artifact_content", last_artifact_content)

        return PlanEvent()

    @step
    async def planning(
        self, ctx: Context, event: PlanEvent
    ) -> Union[GenerateArtifactEvent, SynthesizeAnswerEvent]:
        """
        Based on the conversation history and the user's request
        this step will help to provide a good next step for the code or document generation.
        """
        ctx.write_event_to_stream(
            UIEvent(
                type="ui_event",
                data=UIEventData(
                    state="plan",
                    requirement=None,
                ),
            )
        )

        last_artifact_content = await ctx.get("last_artifact_content")
        if last_artifact_content:
            artifact_context = f"\n## The previous code is: \n{last_artifact_content}\n"
        else:
            artifact_context = ""
        user_msg = await ctx.get("user_msg")

        prompt = PromptTemplate("""
        You are a product analyst responsible for analyzing the user's request and providing the next step for code generation.
    
        Follow these instructions:
        1. Carefully analyze the conversation history and the user's request to determine what has been done and what the next step should be.
        2. The next step must be one of the following two options:
           - "coding": To implement or update the code.
           - "answering": If user asking for clarification.
        3. If the next step is "coding", you may specify the language ("typescript" or "python") and file_name if known, otherwise set them to null. 
        4. The requirement must be provided clearly what is the user request and what need to be done for the next step in details.
           as precise and specific as possible, don't be stingy with in the requirement.
        5. If the next step is "answering", set language and file_name to null, and the requirement should describe what to answer or explain to the user.

        {artifact_context}

        Now, plan the user's next step for this request:
        {user_msg}
        """)
        plan: PlanOutput = await self.llm.astructured_predict(
            output_cls=PlanOutput,
            prompt=prompt,
            artifact_context=artifact_context,
            user_msg=user_msg,
        )
        memory: Memory = await ctx.get("memory")
        memory.put(
            ChatMessage(
                role="assistant",
                content=plan.to_llm_content(),
            )
        )
        await ctx.set("memory", memory)
        if plan.next_step == "coding":
            return GenerateArtifactEvent(
                context=plan.to_llm_content(),
            )
        else:
            return SynthesizeAnswerEvent()

    @step
    async def generate_artifact(
        self, ctx: Context, event: GenerateArtifactEvent
    ) -> SynthesizeAnswerEvent:
        """
        Generate the code based on the user's request.
        """
        ctx.write_event_to_stream(
            UIEvent(
                type="ui_event",
                data=UIEventData(
                    state="generate",
                    requirement=event.context,
                ),
            )
        )

        last_artifact_content = await ctx.get("last_artifact_content")
        if last_artifact_content:
            artifact_context = f"The previous code is: \n{last_artifact_content}\n"
        else:
            artifact_context = "There is no code to update, start from scratch"

        prompt = PromptTemplate("""
         You are a skilled developer who can help user with coding.
         You are given a task to generate or update a code for a given requirement.

         ## Follow these instructions:
         **1. Carefully read the user's requirements.** 
            If any details are ambiguous or missing, make reasonable assumptions and clearly reflect those in your output.
            If the previous code is provided:
            + Carefully analyze the code with the request to make the right changes.
            + Avoid making a lot of changes from the previous code if the request is not to write the code from scratch again.
         **2. For code requests:**
            - If the user does not specify a framework or language, default to a React component using the Next.js framework.
            - For Next.js, use Shadcn UI components, Typescript, @types/node, @types/react, @types/react-dom, PostCSS, and TailwindCSS.
            The import pattern should be:
            ```
            import { ComponentName } from "@/components/ui/component-name"
            import { Markdown } from "@llamaindex/chat-ui"
            import { cn } from "@/lib/utils"
            ```
            - Ensure the code is idiomatic, production-ready, and includes necessary imports.
            - Only generate code relevant to the user's request—do not add extra boilerplate.
         **3. Don't be verbose on response**
            - No other text or comments only return the code which wrapped by ```language``` block.
            - If the user's request is to update the code, only return the updated code.
         **4. Only the following languages are allowed: "typescript", "python".**

        ## Context:
        {artifact_context}

        {plan_context}
         ```
        """)
        code_artifact: CodeArtifact = await self.llm.astructured_predict(
            output_cls=CodeArtifact,
            prompt=prompt,
            artifact_context=artifact_context,
            plan_context=event.context,
        )

        # Put the generated code to the memory
        memory: Memory = await ctx.get("memory")
        memory.put(
            ChatMessage(
                role="assistant",
                content=f"Updated the code: \n{code_artifact.code}",
            )
        )

        # To show the Canvas panel for the artifact in the UI
        ctx.write_event_to_stream(
            ArtifactEvent(
                data=Artifact(
                    type=ArtifactType.CODE,
                    created_at=int(time.time()),
                    data=CodeArtifactData(
                        language=code_artifact.language,
                        file_name=code_artifact.file_name,
                        code=code_artifact.code,
                    ),
                ),
            )
        )
        return SynthesizeAnswerEvent()

    @step
    async def synthesize_answer(
        self, ctx: Context, _: SynthesizeAnswerEvent
    ) -> StopEvent:
        """
        Synthesize the answer.
        """
        memory: Memory = await ctx.get("memory")
        chat_history = memory.get()
        chat_history.append(
            ChatMessage(
                role="system",
                content="""
                You are a helpful assistant who is responsible for explaining the work to the user.
                Based on the conversation history, provide an answer to the user's question. 
                The user has access to the code so avoid mentioning the whole code again in your response.
                """,
            )
        )
        response_stream = await self.llm.astream_chat(
            messages=chat_history,
        )
        ctx.write_event_to_stream(
            UIEvent(
                type="ui_event",
                data=UIEventData(
                    state="completed",
                ),
            )
        )
        response = ""

        # Stream response by sending stream events to the UI
        # Could push pressure to the platform
        # TODO: TBD how could we improve this?
        async for chunk in response_stream:
            ctx.write_event_to_stream(
                StreamEvent(
                    delta=chunk.delta or "",
                )
            )
            response += chunk.delta or ""
        return StopEvent(result=response)


workflow = CodeArtifactWorkflow(timeout=120)
