

import logging
import re
import os
import git
import tempfile
from typing import Dict, List
import numpy as np
import faiss
from openai import OpenAI
from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

from . import brev_agent  # noqa: F401
from . import github_agent  # noqa: F401
from . import nemo_agent  # noqa: F401

logger = logging.getLogger(__name__)

# Global storage for embeddings and indices
_embedding_cache = {}

# Initialize OpenAI client for embeddings
_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(text: str) -> List[float]:
    """Return the embedding vector for text using the OpenAI embedding model."""
    response = _openai_client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding

def load_code_files(repo_path: str,
                   extensions: tuple[str, ...] = (".js", ".ts", ".tsx", ".html", ".py", ".css")) -> Dict[str, str]:
    """Recursively read all files from repo_path that match extensions and return a mapping of
    absolute file paths to their content."""
    code_files: Dict[str, str] = {}
    for root, _dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(extensions):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        code_files[full_path] = f.read()
                except Exception as exc:
                    logger.warning("Failed to read %s: %s", full_path, exc, exc_info=True)
    return code_files

def build_embedding_index(code_files: Dict[str, str]):
    """Build FAISS index from code files."""
    embeddings: List[List[float]] = []
    paths: List[str] = []

    for path, content in code_files.items():
        try:
            emb = embed_text(content)
            embeddings.append(emb)
            paths.append(path)
            logger.debug(f"Created embedding for {path}")
        except Exception as e:
            logger.warning(f"Failed to create embedding for {path}: {e}")
            continue

    if not embeddings:
        logger.warning("No embeddings could be created – repository is empty or unsupported file types.")
        return None, []

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    
    logger.info(f"Built FAISS index with {len(embeddings)} embeddings")
    return index, paths

def create_embeddings_for_repo(repo_path: str) -> None:
    """Create and cache embeddings for a repository."""
    try:
        logger.info(f"Creating embeddings for repository: {repo_path}")
        code_files = load_code_files(repo_path)
        
        if not code_files:
            logger.warning(f"No code files found in repository: {repo_path}")
            return
            
        index, paths = build_embedding_index(code_files)
        
        if index is not None:
            # Cache the embeddings using the repo path as key
            _embedding_cache[repo_path] = {
                'index': index,
                'paths': paths,
                'code_files': code_files
            }
            logger.info(f"✅ Successfully created and cached embeddings for {len(paths)} files in {repo_path}")
        else:
            logger.warning(f"Failed to create embeddings for repository: {repo_path}")
            
    except Exception as e:
        logger.error(f"Error creating embeddings for {repo_path}: {e}")

def get_cached_embeddings(repo_path: str):
    """Get cached embeddings for a repository."""
    return _embedding_cache.get(repo_path)

def update_cached_file(repo_path: str, file_path: str, new_content: str) -> None:
    """Update a specific file in the cached embeddings."""
    cached_data = _embedding_cache.get(repo_path)
    if cached_data:
        # Update the cached file content
        cached_data['code_files'][file_path] = new_content
        logger.info(f"Updated cached content for {file_path}")
        
        # Re-embed the specific file and update the FAISS index
        try:
            # Find the index of this file in the paths list
            paths = cached_data['paths']
            if file_path in paths:
                file_index = paths.index(file_path)
                
                # Create new embedding for the updated content
                new_embedding = embed_text(new_content)
                
                # Rebuild the FAISS index with updated embedding
                index = cached_data['index']
                all_embeddings = []
                
                # Reconstruct all embeddings, replacing the one for the updated file
                for i, path in enumerate(paths):
                    if i == file_index:
                        all_embeddings.append(new_embedding)
                    else:
                        # Get existing embedding from index
                        existing_emb = index.reconstruct(i)
                        all_embeddings.append(existing_emb.tolist())
                
                # Rebuild the index with all embeddings
                dim = len(all_embeddings[0])
                new_index = faiss.IndexFlatL2(dim)
                new_index.add(np.array(all_embeddings).astype("float32"))
                
                # Replace the old index
                cached_data['index'] = new_index
                logger.info(f"Successfully updated embedding and rebuilt index for {file_path}")
            else:
                logger.warning(f"File {file_path} not found in cached paths")
        except Exception as e:
            logger.error(f"Failed to update embedding for {file_path}: {e}")
            # Fallback: rebuild entire index
            logger.info(f"Falling back to full index rebuild for {repo_path}")
            code_files = cached_data['code_files']
            new_index, new_paths = build_embedding_index(code_files)
            if new_index is not None:
                cached_data['index'] = new_index
                cached_data['paths'] = new_paths

def invalidate_embeddings_cache(repo_path: str) -> None:
    """Remove cached embeddings for a repository."""
    if repo_path in _embedding_cache:
        del _embedding_cache[repo_path]
        logger.info(f"Invalidated embeddings cache for {repo_path}")

def search_similar_files(repo_path: str, query: str, k: int) -> List[str]:
    """Search for similar files using cached embeddings."""
    cached_data = get_cached_embeddings(repo_path)
    if not cached_data:
        logger.warning(f"No cached embeddings found for {repo_path}")
        return []
    
    try:
        index = cached_data['index']
        paths = cached_data['paths']
        
        query_emb = np.array([embed_text(query)]).astype("float32")
        _distances, indices = index.search(query_emb, k)
        return [paths[i] for i in indices[0] if i < len(paths)]
    except Exception as e:
        logger.error(f"Error searching similar files: {e}")
        return []


def extract_github_url(text: str) -> str | None:
    match = re.search(r"(https?://github\.com/[\w\-]+/[\w\-]+)", text)
    return match.group(1) if match else None
    
def clone_github_repo(repo_url: str) -> str:
    # Create a local repos directory if it doesn't exist
    local_repos_dir = os.path.join(os.getcwd(), "cloned_repos")
    os.makedirs(local_repos_dir, exist_ok=True)
    
    # Extract repo name from URL
    repo_name = repo_url.rstrip('/').split('/')[-1]
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]
    
    repo_path = os.path.join(local_repos_dir, repo_name)
    print("REPO PATH: ", repo_path)
    
    # Track if we need to create new embeddings
    needs_embedding = False
    
    # If repo already exists, pull latest changes
    if os.path.exists(repo_path):
        try:
            repo = git.Repo(repo_path)
            origin = repo.remotes.origin
            origin.pull()
            logger.info(f"Updated existing repository at: {repo_path}")
            needs_embedding = True  # Re-embed after update
        except Exception as e:
            logger.warning(f"Failed to update existing repo, re-cloning: {e}")
            # Remove and re-clone if update fails
            import shutil
            shutil.rmtree(repo_path)
            repo = git.Repo.clone_from(repo_url, repo_path)
            logger.info(f"Re-cloned repository to: {repo_path}")
            needs_embedding = True
    else:
        # Clone fresh repository
        repo = git.Repo.clone_from(repo_url, repo_path)
        logger.info(f"Cloned new repository to: {repo_path}")
        needs_embedding = True
    
    # Create embeddings if needed
    if needs_embedding:
        create_embeddings_for_repo(repo_path)
    
    return repo_path


class MultiFrameworksWorkflowConfig(FunctionBaseConfig, name="multi_frameworks"):
    llm: LLMRef = "nim_llm"
    github_agent: FunctionRef
    brev_agent: FunctionRef
    nemo_agent: FunctionRef
    
    verbose: bool = False
    max_retries: int = 3
    github_repo: str


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

    # Clone the configured GitHub repository on startup
    startup_repo_path = None
    if config.github_repo:
        try:
            startup_repo_path = clone_github_repo(config.github_repo)
            logger.info(f"✅ Cloned startup GitHub repo to: {startup_repo_path}")
        except Exception as e:
            logger.error(f"❌ Failed to clone startup repo {config.github_repo}: {e}")

    # Get the function instances first and update their configs
    nemo_fn = builder.get_function(config.nemo_agent)
    github_fn = builder.get_function(config.github_agent)
    logger.info("NEMO FN: %s", nemo_fn)
    logger.info("GITHUB FN: %s", github_fn)
    if startup_repo_path:
        nemo_fn.config.repo_path = startup_repo_path
        github_fn.config.github_repo = startup_repo_path

    # Now create the tool wrappers
    brev_agent = builder.get_tool(fn_name=config.brev_agent, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    github_agent = builder.get_tool(fn_name=config.github_agent, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    nemo_agent = builder.get_tool(fn_name=config.nemo_agent, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    chat_hist = ChatMessageHistory()

    class AgentState(TypedDict):
        input: str
        chat_history: list[BaseMessage] | None
        plan: list[dict] | None
        current_step: int | None
        final_output: str | None
        current_result: str | None
        trace: list[str]

    async def router_node(state: AgentState):
        """Route directly to nemo_then_github agent"""
        return {
            **state,
            "current_result": "Routing to nemo_then_github agent",
            "trace": ["Router selected nemo_then_github agent"]
        }

    async def nemo_then_github_node(state: AgentState):
        """Execute Nemo agent then GitHub agent with the changes"""
        try:
            # First run nemo agent
            nemo_result = await nemo_agent.ainvoke(state["input"])
            
            # Check if nemo agent actually made changes
            if "No files were updated" in nemo_result or "❌" in nemo_result:
                return {
                    **state,
                    "final_output": f"Nemo Agent Result: {nemo_result}\n\nSkipping GitHub operations - no changes were made.",
                    "trace": state.get("trace", []) + [
                        f"Nemo agent completed with no changes: {nemo_result[:100]}...",
                        "Skipped GitHub agent - no changes to commit"
                    ]
                }
            
            # Update the github agent function config with the changes
            github_fn = builder.get_function(config.github_agent)
            github_fn.config.changes = nemo_result
            
            # Generate a dynamic commit message based on the user input
            user_input = state["input"]
            # Extract key actions/topics from user input for commit message
            commit_message = f"AI Agent: {user_input[:50]}{'...' if len(user_input) > 50 else ''}"
            
            # Now run github agent with the dynamic commit message
            github_result = await github_agent.ainvoke(commit_message)
            
            return {
                **state,
                "final_output": f"Nemo Agent Results:\n{nemo_result}\n\nGitHub Agent Results:\n{github_result}",
                "trace": state.get("trace", []) + [
                    f"Nemo agent completed: {nemo_result[:100]}...",
                    f"GitHub agent completed: {github_result[:100]}..."
                ]
            }
        except Exception as e:
            return {
                **state,
                "final_output": f"Nemo→GitHub chain error: {str(e)}",
                "trace": state.get("trace", []) + [f"Nemo→GitHub chain failed: {str(e)}"]
            }

    # Build the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes - only router and nemo_then_github
    workflow.add_node("router", router_node)
    workflow.add_node("nemo_then_github", nemo_then_github_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Direct routing from router to nemo_then_github
    workflow.add_edge("router", "nemo_then_github")
    
    # End the workflow after nemo_then_github
    workflow.add_edge("nemo_then_github", END)
    # Compile the workflow
    app = workflow.compile()

    async def _response_fn(input_message: str) -> str:
        try:
            logger.debug("Starting agent execution")

            # Check if there's a GitHub URL in the input message
            repo_url = extract_github_url(input_message)
            print("EXTRACTED REPO URL: ", repo_url)
            
            current_repo_path = startup_repo_path  # Use startup repo by default
            
            if repo_url:
                # If a different repo URL is found in the message, clone it and update the function config
                if repo_url != config.github_repo:
                    current_repo_path = clone_github_repo(repo_url)
                    print("CLONED NEW REPO PATH: ", current_repo_path)
                    logger.info(f"✅ Cloned new GitHub repo from message to: {current_repo_path}")
                    # Update the function config directly
                    nemo_fn.config.repo_path = current_repo_path
                    github_fn.config.github_repo = current_repo_path
                else:
                    print("USING STARTUP REPO PATH: ", current_repo_path)
                    logger.info(f"✅ Using startup GitHub repo at: {current_repo_path}")
                    
            out = await app.ainvoke({"input": input_message, "chat_history": chat_hist})
            return out["final_output"]

        finally:
            logger.debug("Finished agent execution")
    try:
        yield _response_fn
    except GeneratorExit:
        logger.exception("Exited early!", exc_info=True)
    finally:
        logger.debug("Cleaning up multi_frameworks workflow.")
