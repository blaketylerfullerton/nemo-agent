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

logger = logging.getLogger(__name__)


class MultiFrameworksWorkflowConfig(FunctionBaseConfig, name="multi_frameworks"):
    llm: LLMRef = "nim_llm"
    data_dir: str = "/home/coder/dev/ai-query-engine/examples/basic/frameworks/multi_frameworks/data/"
    research_tool: FunctionRef | None = None
    rag_tool: FunctionRef
    chitchat_agent: FunctionRef
    job_application_agent: FunctionRef
    cto_agent: FunctionRef
    cmo_agent: FunctionRef
    ceo_agent: FunctionRef


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

    logger.info("workflow config = %s", config)

    llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    research_tool = None
    if config.research_tool:
        research_tool = builder.get_tool(fn_name=config.research_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    rag_tool = builder.get_tool(fn_name=config.rag_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    chitchat_agent = builder.get_tool(fn_name=config.chitchat_agent, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    job_application_agent = builder.get_tool(fn_name=config.job_application_agent, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
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

        plan = [
            {"agent": "cto_agent", "input": "Is the AI product technically ready to launch next quarter?"},
            {
                "if": "yes",
                "agent": "cmo_agent",
                "input": "Can marketing support the AI product launch next quarter?"
            },
            {
                "if": "yes",
                "agent": "ceo_agent",
                "input": "Should we proceed with launch next quarter?"
            }
        ]

        return {
            **state,
            "plan": plan,
            "current_step": 0,
            "trace": ["🧠 Planner created the execution plan."]
        }

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

    def to_mermaid(trace: list[str]) -> str:
        lines = ["graph TD"]
        for i, step in enumerate(trace):
            step_id = f"Step{i}"
            next_id = f"Step{i+1}"
            lines.append(f'{step_id}["{step}"]')
            if i < len(trace) - 1:
                lines.append(f"{step_id} --> {next_id}")
        return "\n".join(lines)

    async def _response_fn(input_message: str) -> str:
        try:
            logger.debug("Starting agent execution")
            out = await app.ainvoke({"input": input_message, "chat_history": chat_hist})
            output = out["final_output"]
            trace = out.get("trace", [])
            logger.info("final_output : %s ", output)

            mermaid_output = to_mermaid(trace)
            return f"{output}\n\n```mermaid\n{mermaid_output}\n```"

        finally:
            logger.debug("Finished agent execution")

    try:
        yield _response_fn
    except GeneratorExit:
        logger.exception("Exited early!", exc_info=True)
    finally:
        logger.debug("Cleaning up multi_frameworks workflow.")
