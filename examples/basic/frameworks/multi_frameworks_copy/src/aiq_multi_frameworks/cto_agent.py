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
from .demo_data import demo_cto_data

logger = logging.getLogger(__name__)


class CTOAgentConfig(FunctionBaseConfig, name="cto_agent"):
    cto_name: str


@register_function(config_type=CTOAgentConfig)
async def cto_agent_as_tool(tool_config: CTOAgentConfig, builder: Builder):
    async def _arun(inputs: str) -> str:
        return await research_technologies(inputs)
    yield FunctionInfo.from_fn(_arun, description="CTO mock tool")



async def research_technologies(inputs: str) -> str:
    logger.info(f"[CTO] INPUT: {inputs}")
    
    if "ready" in inputs.lower():
        if demo_cto_data["prod_ready"]:
            return "Yes, tech is production-ready. Stack includes: " + ", ".join(demo_cto_data["tech_stack"])
        else:
            return "No, the tech stack still needs work."

    return "Not sure how to respond to that without more context."