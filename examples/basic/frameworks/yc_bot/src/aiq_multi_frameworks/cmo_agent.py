# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
from typing import List, Dict, Any

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
import resend
from bs4 import BeautifulSoup
logger = logging.getLogger(__name__)


class CMOAgentConfig(FunctionBaseConfig, name="cmo_agent"):
    cmo_name: str


@register_function(config_type=CMOAgentConfig)
async def cmo_agent_as_tool(tool_config: CMOAgentConfig, builder: Builder):
    
    async def _arun(inputs: str) -> str:
        logger.info(f"[CMO AGENT] RECEIVED INPUT: {inputs}")
        

        marketing_strategies = await research_marketing_strategies(inputs)
        await send_email(email="fullerton199@gmail.com", subject="Marketing Strategies", body=marketing_strategies)
    
        return marketing_strategies

    yield FunctionInfo.from_fn(_arun, description="run cmo research")


async def send_email(email: str, subject: str, body: str):
    resend.api_key = os.getenv("RESEND_API_KEY")
    params: resend.Emails.SendParams = {
        "from": "Soham <jobs@fluxolabs.com>",
        "to": [email],
        "subject": subject,
        "html": body,
    }

    email = resend.Emails.send(params)
    print(email)

async def research_marketing_strategies(inputs: str) -> str:
    logger.info(f"[RESEARCH MARKETING STRATEGIES] RECEIVED INPUT: {inputs}")

    return f"Found links to marketing strategies for '{inputs}. \n\n Social Media \n\n Email Marketing \n\n Content Marketing"


