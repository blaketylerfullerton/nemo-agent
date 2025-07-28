# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

from . import haystack_agent  # noqa: F401
from . import langchain_research_tool  # noqa: F401
from . import llama_index_rag_tool  # noqa: F401
from . import job_application_agent  # noqa: F401
from . import ceo_agent  # noqa: F401
from . import cto_agent  # noqa: F401
from . import cmo_agent  # noqa: F401


import openai
import os
import json
import requests
import base64


logger = logging.getLogger(__name__)


class MultiFrameworksWorkflowConfig(FunctionBaseConfig, name="multi_frameworks"):
    llm: LLMRef = "nim_llm"
    cmo_agent: FunctionRef
    ceo_agent: FunctionRef
    cto_agent: FunctionRef


@register_function(config_type=MultiFrameworksWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def multi_frameworks_workflow(config: MultiFrameworksWorkflowConfig, builder: Builder):
    from typing import TypedDict

    from colorama import Fore
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.messages import BaseMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langgraph.graph import END, StateGraph

    from openai import OpenAI

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    logger.info("workflow config = %s", config)

    llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    cto_agent = builder.get_tool(fn_name=config.cto_agent, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    cmo_agent = builder.get_tool(fn_name=config.cmo_agent, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    ceo_agent = builder.get_tool(fn_name=config.ceo_agent, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    chat_hist = ChatMessageHistory()

    class AgentState(TypedDict):
        input: str
        chat_history: list[BaseMessage] | None
        plan: list[dict] | None
        current_step: int | None
        final_output: str | None
        current_result: str | None
        trace: list[str]



    async def planner(state: AgentState):
        query = state["input"]
        logger.info("🔷 Planner node: generating a plan for: %s", query)

        prompt = f"""
        You are a workflow planner agent. Based on the user question: "{query}", generate a step-by-step plan using the following agents:
        - cto_agent
        - cmo_agent
        - ceo_agent

        Each step should be a dictionary like:
        {{"agent": "agent_name", "input": "instruction for agent", "if": "optional condition"}}

        Output ONLY a JSON array of steps like this:
        [
        {{"agent": "cto_agent", "input": "some question"}},
        {{"agent": "cmo_agent", "input": "another question", "if": "yes"}},
        ...
        ]
        Do not include any explanation, just the JSON.
        """

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",  
                messages=[{"role": "system", "content": "You are a helpful planner agent."},
                        {"role": "user", "content": prompt}],
                temperature=0.3
            )
            response_text = response.choices[0].message.content.strip()

            # Safely parse JSON
            plan = json.loads(response_text)

            logger.info("Generated plan: %s", plan)

            return {
                **state,
                "plan": plan,
                "current_step": 0,
                "trace": ["🧠 Planner (OpenAI) created the execution plan."]
            }

        except Exception as e:
            logger.error("Error generating plan with OpenAI: %s", str(e))
            return {
                **state,
                "final_output": f"Planner failed to generate a plan: {str(e)}",
                "trace": ["❌ Planner failed."]
            }

    def mermaid_to_image(mermaid_code: str) -> str:
        # Using mermaid.ink service
        encoded = base64.b64encode(mermaid_code.encode()).decode()
        image_url = f"https://mermaid.ink/img/{encoded}"
        return f"![Workflow Diagram]({image_url})"
    
    async def execute_step(state: AgentState):
        step = state["plan"][state["current_step"]]
        step_input = step["input"]
        agent = step["agent"]

        logger.info("🚀 Executing step %s: calling %s with '%s'", state["current_step"], agent, step_input)

        if agent == "cto_agent":
            result = await cto_agent.ainvoke(step_input)
        elif agent == "cmo_agent":
            result = await cmo_agent.ainvoke(step_input)
        elif agent == "ceo_agent":
            result = await ceo_agent.ainvoke(step_input)
        else:
            result = "Unknown agent"

        trace = state.get("trace", [])
        trace.append(f"🔧 {agent} responded to '{step_input}' with: '{result}'")

        return {
            **state,
            "current_result": result,
            "trace": trace
        }

    async def decide_next(state: AgentState):
        current_step = state["current_step"]
        plan = state["plan"]
        result = state["current_result"]

        logger.info("🔍 Decision point after step %s: result = %s", current_step, result)

        trace = state.get("trace", [])
        trace.append(f"📍 Decision made after step {current_step}: {result}")

        if "no" in result.lower():
            return {
                **state,
                "final_output": f"Step {current_step} result blocked further action: {result}",
                "trace": trace
            }
        

        next_step = current_step + 1
        if next_step >= len(plan):
            return {
                **state,
                "final_output": f"Final decision: {result}",
                "trace": trace
            }

        return {
            **state,
            "current_step": next_step,
            "trace": trace
        }

    workflow = StateGraph(AgentState)
    workflow.set_entry_point("planner")
    workflow.add_node("planner", planner)
    workflow.add_node("execute_step", execute_step)
    workflow.add_node("decide_next", decide_next)

    workflow.add_edge("planner", "execute_step")
    workflow.add_edge("execute_step", "decide_next")

    workflow.add_conditional_edges(
        "decide_next",
        lambda state: "end" if state.get("final_output") else "execute_step",
        {
            "execute_step": "execute_step",
            "end": END
        }
    )

    app = workflow.compile()

    async def generate_mermaid_with_llm(trace: list[str], plan: list[dict], final_output: str) -> str:
        """Generate a proper Mermaid diagram using LLM to understand workflow structure"""
        
        # Prepare context for the LLM
        context = {
            "execution_trace": trace,
            "original_plan": plan,
            "final_result": final_output
        }
        
        prompt = f"""
        Create a Mermaid flowchart diagram representing this AI agent workflow execution.
        
        Execution Context:
        - Original Plan: {json.dumps(plan, indent=2)}
        - Execution Trace: {trace}
        - Final Result: {final_output}
        
        Requirements:
        1. Show the actual workflow structure with proper decision points
        2. Include agent names (cto_agent, cmo_agent, ceo_agent) in nodes
        3. Use decision diamonds for conditional flows
        4. Show parallel or sequential execution paths
        5. Indicate success/failure outcomes
        6. Use appropriate Mermaid syntax (flowchart TD)
        
        Return ONLY the Mermaid code, no explanations.
        
        Example format:
        flowchart TD
            A[Start: Planner] --> B{{Plan Generated?}}
            B -->|Yes| C[CTO Agent: Technical Analysis]
            B -->|No| D[Error: Planning Failed]
            C --> E{{Technical Feasible?}}
            E -->|Yes| F[CMO Agent: Marketing Strategy]
            E -->|No| G[End: Technical Blocked]
            F --> H[CEO Agent: Final Decision]
            H --> I[End: Complete]
 
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at creating Mermaid diagrams for workflow visualization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            mermaid_code = response.choices[0].message.content.strip()
            print("MERMAID CODE: ", mermaid_code)
            # Remove code block markers if present
            if mermaid_code.startswith("```"):
                mermaid_code = mermaid_code.split("\n", 1)[1]
                print("MERMAID CODE: ", mermaid_code)

            if mermaid_code.endswith("```"):
                mermaid_code = mermaid_code.rsplit("\n", 1)[0]
                print("MERMAID CODE: ", mermaid_code)

                
            return mermaid_code.strip()
            
        except Exception as e:
            logger.error("Error generating Mermaid with LLM: %s", str(e))
            # Fallback to a simple but better structure
            return f"""
            flowchart TD
                A[🧠 Planner] --> B[📋 Plan: {len(plan)} steps]
                B --> C[🚀 Execution Started]
                C --> D[✅ Result: {final_output[:50]}...]
            """

    async def _response_fn(input_message: str) -> str:
        try:
            logger.debug("Starting agent execution")
            out = await app.ainvoke({"input": input_message, "chat_history": chat_hist})
            output = out["final_output"]
            trace = out.get("trace", [])
            logger.info("final_output : %s ", output)

            mermaid_output = await generate_mermaid_with_llm(trace, out["plan"], output)
            image_url = mermaid_to_image(mermaid_output)
            return f"{output}\n\n**Workflow Execution Diagram:**\n{image_url}"

        finally:
            logger.debug("Finished agent execution")

    try:
        yield _response_fn
    except GeneratorExit:
        logger.exception("Exited early!", exc_info=True)
    finally:
        logger.debug("Cleaning up multi_frameworks workflow.")
