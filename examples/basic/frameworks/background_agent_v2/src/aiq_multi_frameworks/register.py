

import logging
import re
import os
import git
import tempfile
import random
from typing import Dict, List, Tuple
import numpy as np
import faiss
from openai import OpenAI
from datetime import datetime
from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

from .git_functions import clone_github_repo, make_new_branch  # noqa: F401
from .embedding_functions import embed_query, embed_file, embed_file_chunked, retrieve_embeddings  # noqa: F401
from . import github_agent  # noqa: F401
from . import nemo_agent  # noqa: F401

logger = logging.getLogger(__name__)

# Global storage for embeddings and file mappings
_cached_embeddings: Dict[str, List[float]] = {}
_chunked_embeddings: Dict[str, List[List[float]]] = {}  # For files split into chunks
_file_paths: List[str] = []
_chunk_file_map: List[Tuple[str, int]] = []  # Maps chunk index to (file_path, chunk_number)
_embedding_matrix: np.ndarray = None
_faiss_index: faiss.Index = None

def get_cached_embeddings() -> Dict[str, List[float]]:
    """Get the cached embeddings dictionary"""
    return _cached_embeddings

def preprocess_query(query: str) -> str:
    """Preprocess query to improve search results"""
    # Remove common code artifacts that might confuse embedding
    query = re.sub(r'[{}()\[\];,]', ' ', query)
    
    # Add context words that help with semantic matching
    code_context_words = []
    if any(word in query.lower() for word in ['function', 'method', 'class']):
        code_context_words.append('code implementation')
    if any(word in query.lower() for word in ['component', 'react', 'tsx']):
        code_context_words.append('React component')
    if any(word in query.lower() for word in ['api', 'endpoint', 'route']):
        code_context_words.append('API endpoint')
    
    if code_context_words:
        query = f"{query} {' '.join(code_context_words)}"
    
    return query.strip()

def debug_search(query: str, top_k: int = 10) -> Dict:
    """Debug function to analyze search results"""
    global _faiss_index, _file_paths, _chunk_file_map, _cached_embeddings, _chunked_embeddings
    
    debug_info = {
        'original_query': query,
        'preprocessed_query': preprocess_query(query),
        'total_files': len(_cached_embeddings),
        'total_chunks': sum(len(chunks) for chunks in _chunked_embeddings.values()),
        'index_size': len(_file_paths) + len(_chunk_file_map) if _faiss_index else 0,
        'search_results': []
    }
    
    if _faiss_index is None:
        debug_info['error'] = 'No FAISS index available'
        return debug_info
    
    # Try both original and preprocessed queries
    for query_type, test_query in [('original', query), ('preprocessed', preprocess_query(query))]:
        try:
            query_embedding = embed_query(test_query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            scores, indices = _faiss_index.search(query_vector, min(top_k, _faiss_index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                    
                if idx < len(_file_paths):
                    file_path = _file_paths[idx]
                    results.append({
                        'file': os.path.basename(file_path),
                        'full_path': file_path,
                        'score': float(score),
                        'type': 'full_file'
                    })
                else:
                    chunk_idx = idx - len(_file_paths)
                    if chunk_idx < len(_chunk_file_map):
                        file_path, chunk_num = _chunk_file_map[chunk_idx]
                        results.append({
                            'file': os.path.basename(file_path),
                            'full_path': file_path,
                            'score': float(score),
                            'type': f'chunk_{chunk_num}',
                            'chunk_number': chunk_num
                        })
            
            debug_info['search_results'].append({
                'query_type': query_type,
                'query': test_query,
                'results': results[:5]  # Top 5 results
            })
            
        except Exception as e:
            debug_info['search_results'].append({
                'query_type': query_type,
                'query': test_query,
                'error': str(e)
            })
    
    return debug_info

def find_all_code_files(repo_path: str, max_depth: int = 3) -> List[str]:
    """Recursively find all code files in the repository"""
    code_files = []
    code_extensions = {'.tsx', '.ts', '.js', '.jsx', '.py', '.cpp', '.java', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs', '.kt', '.swift', '.dart', '.vue', '.html', '.css', '.scss', '.less'}
    
    # Directories and files to exclude
    exclude_dirs = {
        'node_modules', '.git', '.vscode', '.idea', 'dist', 'build', '__pycache__', 
        '.next', '.nuxt', 'coverage', '.pytest_cache', 'venv', 'env', '.env',
        'vendor', 'target', 'bin', 'obj', '.vs', 'Pods'
    }
    
    exclude_files = {
        'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', '.gitignore', 
        '.eslintrc', '.prettierrc', 'tsconfig.json', 'webpack.config.js',
        'vite.config.js', 'tailwind.config.js', 'next.config.js'
    }
    
    def _scan_directory(path: str, current_depth: int = 0) -> None:
        if current_depth > max_depth:
            return
            
        try:
            for item in os.listdir(path):
                if item.startswith('.') and item not in {'.env'}:
                    continue
                    
                item_path = os.path.join(path, item)
                
                if os.path.isdir(item_path):
                    if item not in exclude_dirs:
                        _scan_directory(item_path, current_depth + 1)
                elif os.path.isfile(item_path):
                    # Check file extension
                    _, ext = os.path.splitext(item)
                    if ext.lower() in code_extensions and item not in exclude_files:
                        # Additional size check - skip very large files
                        try:
                            if os.path.getsize(item_path) < 1024 * 1024:  # 1MB limit
                                code_files.append(item_path)
                        except OSError:
                            continue
        except PermissionError:
            logger.warning(f"Permission denied accessing {path}")
        except Exception as e:
            logger.warning(f"Error scanning directory {path}: {e}")
    
    _scan_directory(repo_path)
    return code_files

def search_similar_files(query: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """Search for files similar to the query using embeddings"""
    global _faiss_index, _file_paths, _chunk_file_map
    
    if _faiss_index is None or not _file_paths:
        return []
    
    # Preprocess query for better matching
    processed_query = preprocess_query(query)
    
    # Get query embedding
    query_embedding = embed_query(processed_query)
    query_vector = np.array([query_embedding], dtype=np.float32)
    
    # Normalize query vector for cosine similarity
    faiss.normalize_L2(query_vector)
    
    # Search using FAISS - get more results to account for chunks
    search_k = min(top_k * 3, len(_file_paths) + len(_chunk_file_map))
    scores, indices = _faiss_index.search(query_vector, search_k)
    
    # Aggregate results by file (in case of chunked files)
    file_scores = {}
    
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # Invalid index
            continue
            
        if idx < len(_file_paths):
            # Regular file embedding
            file_path = _file_paths[idx]
            file_scores[file_path] = max(file_scores.get(file_path, 0), float(score))
        else:
            # Chunked file embedding
            chunk_idx = idx - len(_file_paths)
            if chunk_idx < len(_chunk_file_map):
                file_path, chunk_num = _chunk_file_map[chunk_idx]
                file_scores[file_path] = max(file_scores.get(file_path, 0), float(score))
    
    # Sort by score and return top k
    results = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Log search results for debugging
    logger.info(f"Search query: '{query}' -> '{processed_query}' found {len(results)} files")
    for file_path, score in results[:3]:  # Log top 3
        logger.info(f"  {os.path.basename(file_path)}: {score:.4f}")
    
    return results

def update_cached_file(file_path: str, new_embedding: List[float]):
    """Update the embedding for a specific file"""
    global _cached_embeddings, _embedding_matrix, _faiss_index, _file_paths
    
    _cached_embeddings[file_path] = new_embedding
    
    # Rebuild the FAISS index
    _rebuild_faiss_index()

def _rebuild_faiss_index():
    """Rebuild the FAISS index from current embeddings"""
    global _cached_embeddings, _chunked_embeddings, _embedding_matrix, _faiss_index, _file_paths, _chunk_file_map
    
    if not _cached_embeddings and not _chunked_embeddings:
        return
    
    all_embeddings = []
    _file_paths = []
    _chunk_file_map = []
    
    # Add regular file embeddings
    for file_path, embedding in _cached_embeddings.items():
        all_embeddings.append(embedding)
        _file_paths.append(file_path)
    
    # Add chunked file embeddings
    for file_path, chunk_embeddings in _chunked_embeddings.items():
        for chunk_idx, embedding in enumerate(chunk_embeddings):
            all_embeddings.append(embedding)
            _chunk_file_map.append((file_path, chunk_idx))
    
    if all_embeddings:
        _embedding_matrix = np.array(all_embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(_embedding_matrix)
        
        # Create FAISS index with cosine similarity (inner product on normalized vectors)
        dimension = len(all_embeddings[0])
        _faiss_index = faiss.IndexFlatIP(dimension)
        _faiss_index.add(_embedding_matrix)
        
        logger.info(f"Built FAISS index with {len(_file_paths)} files and {len(_chunk_file_map)} chunks")

class MultiFrameworksWorkflowConfig(FunctionBaseConfig, name="multi_frameworks"):
    github_repo: str
    llm: LLMRef
    nemo_agent: FunctionRef
    github_agent: FunctionRef

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

    #Clone the repository
    repo_path = clone_github_repo(config.github_repo)
    logger.info(f"Cloned repository: {repo_path}")

    #make new branch
    branch_name = "Disbot" + str(random.randint(1, 100))
    make_new_branch(repo_path, branch_name)
    logger.info(f"Created new branch: {branch_name}")

    #store embeddings with FAISS index for efficient search
    global _cached_embeddings, _chunked_embeddings, _embedding_matrix, _faiss_index, _file_paths, _chunk_file_map
    
    _cached_embeddings.clear()
    _chunked_embeddings.clear()
    
    # Find all code files recursively
    all_code_files = find_all_code_files(repo_path)
    logger.info(f"Found {len(all_code_files)} code files to embed")
    
    # Embed all files
    embedded_count = 0
    for file_path in all_code_files:
        try:
            # Check file size to decide on chunking strategy
            file_size = os.path.getsize(file_path)
            
            if file_size > 16000:  # Large files - use chunking
                chunk_embeddings = embed_file_chunked(file_path)
                _chunked_embeddings[file_path] = chunk_embeddings
                logger.info(f"Embedded file in {len(chunk_embeddings)} chunks: {os.path.relpath(file_path, repo_path)}")
            else:
                # Small files - embed as single unit
                file_embedding = embed_file(file_path)
                _cached_embeddings[file_path] = file_embedding
                logger.info(f"Embedded file: {os.path.relpath(file_path, repo_path)}")
            
            embedded_count += 1
            
        except Exception as e:
            logger.error(f"Failed to embed file {file_path}: {e}")
    
    # Build FAISS index
    _rebuild_faiss_index()
    
    if embedded_count == 0:
        logger.warning("No files were embedded successfully")
    else:
        logger.info(f"Successfully embedded {embedded_count} files")

    llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # Clone the configured GitHub repository on startup
   
    # Get the function instances first and update their configs
    nemo_fn = builder.get_function(config.nemo_agent)
    github_fn = builder.get_function(config.github_agent)
    # Now create the tool wrappers
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
        """Execute nemo agent first, then github agent"""
        try:
            # First, use nemo agent to analyze and get code suggestions
            nemo_result = await nemo_agent.ainvoke({"user_prompt": state["input"]})
            logger.info(f"Nemo result: {nemo_result}")
            
            # Then, use github agent to implement the changes (commit & PR)
            github_result = await github_agent.ainvoke({"repo_path": repo_path, "changes": nemo_result})
            logger.info(f"GitHub result: {github_result}")
            return {
                **state,
                "final_output": f"Nemo Analysis: {nemo_result}\n\nGitHub Implementation: {github_result}",
                "trace": state.get("trace", []) + ["Executed nemo agent", "Executed github agent"]
            }
        except Exception as e:
            logger.error(f"Error in nemo_then_github_node: {e}")
            return {
                **state,
                "final_output": f"Error: {str(e)}",
                "trace": state.get("trace", []) + [f"Error: {str(e)}"]
            }

    
    # Build the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes - router and nemo_then_github
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
            
            # Create initial state
            initial_state = {
                "input": input_message,
                "chat_history": None,
                "plan": None,
                "current_step": None,
                "final_output": None,
                "current_result": None,
                "trace": []
            }
            
            # Execute the workflow
            result = await app.ainvoke(initial_state)
            
            return result.get("final_output", "No output generated")
            
        except Exception as e:
            logger.error(f"Error in _response_fn: {e}")
            return f"Error executing workflow: {str(e)}"
        finally:
            logger.debug("Finished agent execution")
    try:
        yield _response_fn
    except GeneratorExit:
        logger.exception("Exited early!", exc_info=True)
    finally:
        logger.debug("Cleaning up multi_frameworks workflow.")
