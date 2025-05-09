from jupyter_ai.personas.base_persona import BasePersona
from jupyterlab_chat.models import Message

from agno.agent import Agent
from agno.models.message import Message as AgnoMessage


async def transform_iterator(source_iterator):
    async for item in source_iterator:
        yield item.content.replace('$',r'\\$')


class AgnoPersona(BasePersona):
    """
    The debug persona, the main persona provided by Jupyter AI.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = self.create_agent()

    def create_agent(self) -> Agent:
        raise NotImplementedError("Please implement create_agent()")

    def get_role(self, ychat_message):
        if ychat_message.sender in self.manager.personas:
            return 'assistant'
        else:
            return 'user'

    async def process_message(self, message: Message):
        history = self.ychat.get_messages()[-4:]
        messages = [AgnoMessage(content=h.body, name=h.sender, role=self.get_role(h)) for h in history]
        stream = await self.agent.arun(
            message.body,
            user_id=message.sender,
            session_id=self.ychat.get_id(),
            messages=messages
        )
        await self.forward_reply_stream(transform_iterator(stream))

