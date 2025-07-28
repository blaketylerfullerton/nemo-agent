

import logging
from typing import TypedDict, Dict, Any, List
from dataclasses import dataclass

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef, LLMRef
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.api_server import AIQChatRequest

# Import our individual agents
from . import github_agent  # noqa: F401, pylint: disable=unused-import
from . import code_analysis_agent  # noqa: F401, pylint: disable=unused-import
from . import file_modification_agent  # noqa: F401, pylint: disable=unused-import

logger = logging.getLogger(__name__)


# ==========================================
# 1. MAIN WORKFLOW CONFIGURATION
# ==========================================

class CodeEditorWorkflowConfig(FunctionBaseConfig, name="code_editor_workflow"):
    """Configuration for the code editor workflow"""
    llm: LLMRef = "nim_llm"
    project_temp_dir: str = "/tmp/code_editor_sessions/"
    
    # Agent references
    github_agent: FunctionRef
    code_analysis_agent: FunctionRef
    file_modification_agent: FunctionRef


# ==========================================
# 2. WORKFLOW STATE DEFINITION
# ==========================================

class CodeEditorState(TypedDict):
    """State that flows between agents"""
    input_query: str
    repo_url: str
    branch: str
    commit_message: str | None
    
    # Session management
    session_id: str | None
    current_phase: str | None  # "clone", "analyze", "modify", "commit"
    
    # Repository state
    repo_path: str | None
    
    # Analysis results
    relevant_files: List[str] | None
    files_to_modify: List[str] | None
    modification_plan: Dict[str, Any] | None
    
    # Modification results
    modified_files: List[str] | None
    modification_errors: List[str] | None
    
    # Final results
    commit_hash: str | None
    success: bool | None
    error_message: str | None
    final_output: str | None


# ==========================================
# 3. MAIN WORKFLOW ORCHESTRATOR
# ==========================================

@register_function(config_type=CodeEditorWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def code_editor_workflow(config: CodeEditorWorkflowConfig, builder: Builder):
    """Main workflow orchestrating code editing process"""
    
    from colorama import Fore
    from langgraph.graph import END, StateGraph
    import uuid
    
    logger.info("Code editor workflow config = %s", config)
    
    # Get agents from builder
    github_agent = builder.get_tool(fn_name=config.github_agent, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    code_analysis_agent = builder.get_tool(fn_name=config.code_analysis_agent, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    file_modification_agent = builder.get_tool(fn_name=config.file_modification_agent, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    # ==========================================
    # WORKFLOW NODES
    # ==========================================
    
    async def initialize_session(state: CodeEditorState):
        """Initialize a new editing session"""
        session_id = str(uuid.uuid4())[:8]
        logger.info("%s[%s] Initializing session for query: %s%s", 
                   Fore.BLUE, session_id, state["input_query"], Fore.RESET)
        
        return {
            **state,
            "session_id": session_id,
            "current_phase": "clone",
            "success": False
        }
    
    async def clone_repository(state: CodeEditorState):
        """Clone the target repository"""
        session_id = state["session_id"]
        logger.info("%s[%s] Cloning repository: %s%s", 
                   Fore.CYAN, session_id, state["repo_url"], Fore.RESET)
        
        clone_request = {
            "action": "clone",
            "repo_url": state["repo_url"],
            "branch": state["branch"],
            "session_id": session_id
        }
        
        result = await github_agent.ainvoke(clone_request)
        
        if result["success"]:
            return {
                **state,
                "repo_path": result["repo_path"],
                "current_phase": "analyze"
            }
        else:
            return {
                **state,
                "success": False,
                "error_message": f"Failed to clone repository: {result['error']}",
                "current_phase": "error"
            }
    
    async def analyze_code(state: CodeEditorState):
        """Analyze code to determine what needs modification"""
        session_id = state["session_id"]
        logger.info("%s[%s] Analyzing code for query: %s%s", 
                   Fore.YELLOW, session_id, state["input_query"], Fore.RESET)
        
        analysis_request = {
            "query": state["input_query"],
            "repo_path": state["repo_path"],
            "session_id": session_id
        }
        
        result = await code_analysis_agent.ainvoke(analysis_request)
        
        if result["success"]:
            return {
                **state,
                "relevant_files": result["relevant_files"],
                "files_to_modify": result["files_to_modify"],
                "modification_plan": result["modification_plan"],
                "current_phase": "modify"
            }
        else:
            return {
                **state,
                "success": False,
                "error_message": f"Code analysis failed: {result['error']}",
                "current_phase": "error"
            }
    
    async def modify_files(state: CodeEditorState):
        """Modify the identified files"""
        session_id = state["session_id"]
        files_to_modify = state["files_to_modify"]
        
        logger.info("%s[%s] Modifying %d files%s", 
                   Fore.GREEN, session_id, len(files_to_modify), Fore.RESET)
        
        modification_request = {
            "query": state["input_query"],
            "files_to_modify": files_to_modify,
            "modification_plan": state["modification_plan"],
            "repo_path": state["repo_path"],
            "session_id": session_id
        }
        
        result = await file_modification_agent.ainvoke(modification_request)
        
        if result["success"]:
            return {
                **state,
                "modified_files": result["modified_files"],
                "modification_errors": result.get("errors", []),
                "current_phase": "commit"
            }
        else:
            return {
                **state,
                "success": False,
                "error_message": f"File modification failed: {result['error']}",
                "current_phase": "error"
            }
    
    async def commit_and_push(state: CodeEditorState):
        """Commit and push changes"""
        session_id = state["session_id"]
        modified_files = state["modified_files"]
        
        logger.info("%s[%s] Committing %d modified files%s", 
                   Fore.MAGENTA, session_id, len(modified_files), Fore.RESET)
        
        commit_message = state["commit_message"] or f"Code Editor Agent: {state['input_query']}"
        
        commit_request = {
            "action": "commit_and_push",
            "modified_files": modified_files,
            "commit_message": commit_message,
            "session_id": session_id
        }
        
        result = await github_agent.ainvoke(commit_request)
        logger.info("Commit result: %s", result)
        if result["success"]:
            # Build final output message with PR URL if available
            final_message = f"Successfully modified {len(modified_files)} files and committed with hash: {result['commit_hash']}"
            if "pr_url" in result:
                final_message += f"\nPull Request created: {result['pr_url']}"
            
            return {
                **state,
                "commit_hash": result["commit_hash"],
                "success": True,
                "current_phase": "complete",
                "final_output": final_message
            }
        else:
            return {
                **state,
                "success": False,
                "error_message": f"Commit failed: {result['error']}",
                "current_phase": "error"
            }
    
    async def cleanup_session(state: CodeEditorState):
        """Clean up temporary resources"""
        session_id = state["session_id"]
        logger.info("%s[%s] Cleaning up session%s", 
                   Fore.RED, session_id, Fore.RESET)
        
        cleanup_request = {
            "action": "cleanup",
            "session_id": session_id
        }
        
        await github_agent.ainvoke(cleanup_request)
        
        if state["success"]:
            return {
                **state,
                "final_output": state.get("final_output", "Operation completed successfully")
            }
        else:
            return {
                **state,
                "final_output": f"Operation failed: {state.get('error_message', 'Unknown error')}"
            }
    
    # ==========================================
    # WORKFLOW ROUTER
    # ==========================================
    
    async def router(state: CodeEditorState):
        """Route to next node based on current phase"""
        current_phase = state.get("current_phase")
        
        logger.info("Router: current phase = %s", current_phase)
        
        if current_phase == "clone":
            return "clone_repository"
        elif current_phase == "analyze":
            return "analyze_code"
        elif current_phase == "modify":
            return "modify_files"
        elif current_phase == "commit":
            return "commit_and_push"
        elif current_phase in ["complete", "error"]:
            return "cleanup_session"
        else:
            return "end"
    
    # ==========================================
    # BUILD WORKFLOW GRAPH
    # ==========================================
    
    workflow = StateGraph(CodeEditorState)
    
    # Add nodes
    workflow.add_node("initialize_session", initialize_session)
    workflow.add_node("clone_repository", clone_repository)
    workflow.add_node("analyze_code", analyze_code)
    workflow.add_node("modify_files", modify_files)
    workflow.add_node("commit_and_push", commit_and_push)
    workflow.add_node("cleanup_session", cleanup_session)
    
    # Set entry point
    workflow.set_entry_point("initialize_session")
    
    # Add edges
    workflow.add_conditional_edges(
        "initialize_session",
        router,
        {
            "clone_repository": "clone_repository",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "clone_repository",
        router,
        {
            "analyze_code": "analyze_code",
            "cleanup_session": "cleanup_session",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "analyze_code",
        router,
        {
            "modify_files": "modify_files",
            "cleanup_session": "cleanup_session",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "modify_files",
        router,
        {
            "commit_and_push": "commit_and_push",
            "cleanup_session": "cleanup_session",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "commit_and_push",
        router,
        {
            "cleanup_session": "cleanup_session",
            "end": END
        }
    )
    
    workflow.add_edge("cleanup_session", END)
    
    # Compile the workflow
    app = workflow.compile()
    
    # ==========================================
    # MAIN RESPONSE FUNCTION
    # ==========================================
    
    async def _response_fn(chat_request: AIQChatRequest) -> str:
        """Process code editing request"""
        try:
            logger.debug("Starting code editor workflow")
            
            # Extract the message content from the chat request
            if chat_request.messages and len(chat_request.messages) > 0:
                input_message = chat_request.messages[-1].content
                if isinstance(input_message, list):
                    # Handle list of content items
                    input_message = " ".join([str(item) for item in input_message])
                elif isinstance(input_message, str):
                    # Handle string content
                    pass
                else:
                    input_message = str(input_message)
            else:
                input_message = "No message provided"
            
            # For now, use default values for repo_url and branch
            # In a real implementation, you might want to extract these from the message
            # or have them as additional fields in the chat request
            repo_url = "https://github.com/blaketylerfullerton/NeMo-Agent-Toolkit-UI"
            branch = "main"
            commit_message = None
            
            initial_state = {
                "input_query": input_message,
                "repo_url": repo_url,
                "branch": branch,
                "commit_message": commit_message
            }
            
            result = await app.ainvoke(initial_state)
            output = result["final_output"]
            
            logger.info("Code editor workflow completed: %s", output)
            return output
            
        except Exception as e:
            logger.exception("Code editor workflow failed", exc_info=True)
            return f"Workflow failed: {str(e)}"
        finally:
            logger.debug("Finished code editor workflow execution")
    
    try:
        yield _response_fn
    except GeneratorExit:
        logger.exception("Exited early!", exc_info=True)
    finally:
        logger.debug("Cleaning up code editor workflow.")

