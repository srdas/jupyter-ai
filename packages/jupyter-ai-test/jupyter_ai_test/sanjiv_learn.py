from typing import Any

from jupyterlab_chat.models import Message
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from jupyter_core.paths import jupyter_data_dir

from jupyter_ai.history import YChatHistory
from jupyter_ai.personas import BasePersona, PersonaDefaults
from jupyter_ai.personas.jupyternaut.prompt_template import JUPYTERNAUT_PROMPT_TEMPLATE, JupyternautVariables
from jupyter_ai.chat_handlers.base import BaseChatHandler
from jupyter_ai.config_manager import DEFAULT_CONFIG_PATH
from jupyter_ai.chat_handlers import LearnChatHandler

from agno.agent import Agent
from agno.models.aws import AwsBedrock
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.models.openai import OpenAIChat
from agno.team import Team

import os
import json

# def env_api_keys_from_config(file_path=DEFAULT_CONFIG_PATH):
#     """
#     Reads a JSON file at 'file_path' and returns the 'api_keys' dictionary.
#     """
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#     for key, value in data.get('api_keys', {}).items():
#         os.environ[key] = value
#     pass


# Note: remove BasePersona and just use BaseChatHandler to try /learn
class SanjivLearnPersona(BasePersona, BaseChatHandler):
    """
    The Jupyternaut persona, the main persona provided by Jupyter AI.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def defaults(self):
        return PersonaDefaults(
            name="SanjivLearn",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="The standard agent provided by JupyterLab. Currently has no tools.",
            system_prompt="...",
        )

    async def process_message(self, message: Message):
        provider_name = self.config.lm_provider.name
        model_id = self.config.lm_provider_params["model_id"]

        runnable = self.build_runnable()
        variables = JupyternautVariables(
            input=message.body,
            model_id=model_id,
            provider_name=provider_name,
            persona_name=self.name,
        )
        if "/learn" in variables.input:
            msg = variables.input.split(" ", 1)[1].strip()
            if msg:
                self.reply(f"Learning: {msg}")
                print(f"====>self.config: {self.config._config}")
                # Code to call LearnChatHandler and learn the message
                learn_handler = LearnChatHandler(self.config._config)
                learn_handler.learn(msg)
            else:
                self.reply("Error: No ticker provided. Provide it in the format /ticker <ticker>")
        else:
            variables_dict = variables.model_dump()
            reply_stream = runnable.astream(variables_dict)
            await self.forward_reply_stream(reply_stream)

    def build_runnable(self) -> Any:
        # TODO: support model parameters. maybe we just add it to lm_provider_params in both 2.x and 3.x
        llm = self.config.lm_provider(**self.config.lm_provider_params)
        print(f"====> llm: {llm}")
        print(f"====> llm provider params: ", self.config.lm_provider_params)
        print(f"====> llm provider: ", self.config.lm_provider)
        print(f"====> llm provider name: ", self.config.lm_provider.name)
        print(f"====> llm provider name: ", self.config.lm_provider_params["model_id"])
        runnable = JUPYTERNAUT_PROMPT_TEMPLATE | llm | StrOutputParser()

        runnable = RunnableWithMessageHistory(
            runnable=runnable,  #  type:ignore[arg-type]
            get_session_history=lambda: YChatHistory(ychat=self.ychat, k=2),
            input_messages_key="input",
            history_messages_key="history",
        )

        return runnable
    
    
