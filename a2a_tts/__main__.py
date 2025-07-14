import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from agent_executor import (
    SpeakingAgentExecutor,
)

if __name__ == '__main__':
    skill = AgentSkill(
            id='describe_capabilities',
            name='Describe Capabilities',
            description="Describes the agent's abilities, tools, and how it can be used within the master agent system.",
            tags=['help', 'capabilities', 'about', 'skills'],
            examples=['what can you do?', 'help', 'describe yourself', 'show skills'],
    )

    public_agent_card = AgentCard(
        name='Speaking Agent',
        description='An agent that speaks a predefined message out loud using text-to-speech.',
        url='http://host.docker.internal:9999/',
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill], 
        supportsAuthenticatedExtendedCard=True,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=SpeakingAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
    )

    uvicorn.run(server.build(), host='0.0.0.0', port=9999)
