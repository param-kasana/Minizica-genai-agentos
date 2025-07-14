import threading
import pyttsx3
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_task, new_text_artifact
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)

MESSAGE = "I am an Agentic Operating System that can perform most of the daily tasks for you. "

def speak_background(tts_text):
    def run():
        engine = pyttsx3.init()
        engine.setProperty('rate', 180) 
        engine.setProperty('volume', 1.0)
        engine.say(tts_text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

class SpeakingAgent:
    async def invoke(self) -> str:
        return MESSAGE

class SpeakingAgentExecutor(AgentExecutor):
    def __init__(self):
        self.agent = SpeakingAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            if context.message is None:
                raise ValueError("No user message found")
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        message = await self.agent.invoke()

        # Send the content as an artifact update
        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                append=False,
                lastChunk=True,
                contextId=task.contextId,
                taskId=task.id,
                artifact=new_text_artifact(
                    name="spoken_message",
                    description="spoken message",
                    text=message,
                ),
            )
        )

        # Speak the message in the background
        speak_background(message)

        # Mark the task as completed
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                final=True,
                contextId=task.contextId,
                taskId=task.id,
                status=TaskStatus(state=TaskState.completed),
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")
