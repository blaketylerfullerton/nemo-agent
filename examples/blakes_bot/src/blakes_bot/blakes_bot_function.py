import logging

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.builder.framework_enum import LLMFrameworkEnum

logger = logging.getLogger(__name__)

class BlakesBotFunctionConfig(FunctionBaseConfig, name="blakes_bot"):
    """
    Gets Blakes favorite food
    """
    # Add your custom configuration parameters here
    tool_name: list[FunctionRef] = Field(default=[], description="List of tools to use")
    llm_name: LLMRef = Field(description="The name of the LLM to use")
    max_history: int = Field(default=10, description="The maximum number of historical messages to pass to agent")
    description: str = Field(default="", description="The description of the tool")
    

@register_function(config_type=BlakesBotFunctionConfig)
async def blakes_bot_function(
    config: BlakesBotFunctionConfig, builder: Builder
):
    # Implement your function logic here
    from langchain import hub
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.schema import BaseMessage
    from langchain_core.messages import trim_messages
    from aiq.data_models.api_server import AIQChatRequest
    from aiq.data_models.api_server import AIQChatResponse


    tools = builder.get_tools(tool_names=config.tool_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    llm =  await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    prompt = hub.pull("khodak/react-agent-template")
    agent = create_react_agent(
        llm=llm,
        prompt=prompt,
        tools=tools,
        max_iterations=config.max_history,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=15, handle_parsing_errors=True, verbose=True, return_intermediate_steps=True)


    async def _response_fn(input_message: AIQChatRequest) -> AIQChatResponse:

        last_message = input_message.messages[-1].content
        chat_history = trim_messages(messages=[m.model_dump() for m in input_message.messages], 
                                     max_tokens=config.max_history,
                                     strategy="last",
                                     start_on = "human",
                                     include_system = True,
                                     )
        response = await agent_executor.invoke({"input": last_message, "chat_history": chat_history})
        return AIQChatResponse.from_string(response["output"])

    try:
        yield FunctionInfo.create(single_fn=_response_fn)
    except GeneratorExit:
        print("Function exited early!")
    finally:
        print("Cleaning up blakes_bot workflow.")