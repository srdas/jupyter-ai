from typing import Any

from jupyterlab_chat.models import Message
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from jupyter_core.paths import jupyter_data_dir

from jupyter_ai.history import YChatHistory
from jupyter_ai.personas import BasePersona, PersonaDefaults
from .agno_persona import AgnoPersona
from jupyter_ai.personas.jupyternaut.prompt_template import JUPYTERNAUT_PROMPT_TEMPLATE, JupyternautVariables
from jupyter_ai.chat_handlers.base import BaseChatHandler
from jupyter_ai.config_manager import DEFAULT_CONFIG_PATH

from agno.agent import Agent
from agno.models.aws import AwsBedrock
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.models.openai import OpenAIChat
from agno.team import Team
from .fd import FinancialDatasetsTools
from .sanjiv_fintools import FinTools

import os
import json
import requests
import pandas as pd  # requires: pip install pandas
import torch
from chronos import BaseChronosPipeline # requires: pip install chronos-forecasting
import matplotlib.pyplot as plt  # requires: pip install matplotlib
import numpy as np



def env_api_keys_from_config(API_KEY_NAME, file_path=DEFAULT_CONFIG_PATH):
    """
    Reads a JSON file at 'file_path' and returns the 'api_keys' dictionary.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    for key, value in data.get('api_keys', {}).items():
        os.environ[key] = value
        if key == API_KEY_NAME:
            fin_key = value
    return fin_key

# Sample data for testing 
# This is a sample data structure to simulate the response from the API
def sample_prices(file_path=None):
    if file_path:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    else:
        print("====> Sample data not found.")
              

class FinancePersona(BasePersona):
    """
    The Jupyternaut persona, the main persona provided by Jupyter AI.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #https://www.financialdatasets.ai/

    @property
    def defaults(self):
        return PersonaDefaults(
            name="FinancePersona",
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
        # Check if the message contains specific commands
        if "/ticker" in variables.input:
            msg = variables.input.split("/ticker", 1)[1].strip()
            print(f"====> TICKER: {msg}")
            if msg:
                # Call the agno_ticker function to process the message
                self.agno_finance(msg)
            else:
                self.send_message("Error: No ticker provided. Provide it in the format /ticker <ticker>")
        else:
            variables_dict = variables.model_dump()
            reply_stream = runnable.astream(variables_dict)
            await self.stream_message(reply_stream)

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
    
    # Use Agno to process the /ticker command
    def agno_finance(self, message: Message):
        FINANCIAL_DATASETS_API_KEY = env_api_keys_from_config(API_KEY_NAME="TOGETHER_API_KEY", file_path=DEFAULT_CONFIG_PATH)
        ticker = message.strip()
        # llm = self.config.lm_provider(**self.config.lm_provider_params)
        # model_id = self.config.lm_provider_params["model_id"]
        stock_price_agent = Agent(
            role="Get stock prices for a given date range.",
            model=OpenAIChat(id="gpt-4.1"),
            description="Agent to get the stock price information for a ticker.",
            instructions="For a given ticker, please collect the latest stock prices for the date range provided.",
            tools = [FinancialDatasetsTools(enable_company_info=False, enable_prices=True, api_key=FINANCIAL_DATASETS_API_KEY)],
            show_tool_calls=False,
            markdown=True
        )
        # Save the response to a variable
        response = stock_price_agent.run(f"{message}")
        response = response.content
        # Print the response  
        self.send_message(f"====> ticker response: {response}")


