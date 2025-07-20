
import logging
import os
import re
from typing import List, Dict, Any
from openai import OpenAI

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig
import json
import resend
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
from .demo_data import demo_ceo_data

logger = logging.getLogger(__name__)


class CEOAgentConfig(FunctionBaseConfig, name="ceo_agent"):
    ceo_name: str
    llm_name: LLMRef

@register_function(config_type=CEOAgentConfig)
async def ceo_agent_as_tool(tool_config: CEOAgentConfig, builder: Builder):
  

    async def _arun(inputs: str) -> str:
        output = await research_ceo(inputs)
        logger.info("output from langchain_research_tool: %s", output)  # noqa: W293 E501
        return output

    yield FunctionInfo.from_fn(_arun, description="run ceo research")

async def research_ceo(inputs: str) -> str:
    if "launch" in inputs.lower():
        if demo_ceo_data["approve_launch"] and demo_ceo_data["runway_months"] > 3:
            return "Yes, we have enough financial stability to approve launch."
        else:
            return "No, financial runway or risk factors suggest holding off."