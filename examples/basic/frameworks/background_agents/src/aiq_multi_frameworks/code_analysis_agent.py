import logging
from typing import Dict, Any, List
from aiq.data_models.function import FunctionBaseConfig
from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef, LLMRef
from aiq.data_models.function import FunctionBaseConfig
from aiq.builder.function_info import FunctionInfo
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Code Analysis Agent Configuration
class CodeAnalysisAgentConfig(FunctionBaseConfig, name="code_analysis_agent"):
    """Configuration for code analysis agent"""
    llm: LLMRef = "nim_llm"
    max_files_to_analyze: int = 5


# Input schema for code analysis
class CodeAnalysisRequest(BaseModel):
    """Input schema for code analysis operations"""
    query: str = Field(description="The user query describing what code changes are needed")
    repo_path: str = Field(description="Path to the cloned repository")
    session_id: str = Field(description="Unique session identifier")


@register_function(config_type=CodeAnalysisAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def code_analysis_agent(config: CodeAnalysisAgentConfig, builder: Builder):
    """Code analysis agent: find relevant files and create modification plan"""
    
    import os
    from pathlib import Path
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    logger.info("Code analysis agent config = %s", config)
    
    llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    async def _analyze_code(request: CodeAnalysisRequest) -> Dict[str, Any]:
        """Analyze code and determine modification plan"""
        query = request.query
        repo_path = request.repo_path
        session_id = request.session_id
        
        try:
            # Find relevant files
            relevant_files = _find_relevant_files(query, repo_path)
            
            if not relevant_files:
                return {"success": False, "error": "No relevant files found"}
            
            # Analyze which files need modification
            files_to_modify = []
            modification_plan = {}
            
            for file_path in relevant_files[:config.max_files_to_analyze]:
                should_modify, plan = await _should_modify_file(llm, query, file_path, repo_path)
                
                if should_modify:
                    files_to_modify.append(file_path)
                    modification_plan[file_path] = plan
            
            return {
                "success": True,
                "relevant_files": relevant_files,
                "files_to_modify": files_to_modify,
                "modification_plan": modification_plan
            }
            
        except Exception as e:
            logger.exception("Code analysis failed")
            return {"success": False, "error": str(e)}
    
    def _find_relevant_files(query: str, repo_path: str) -> List[str]:
        """Find files relevant to the query, excluding those with "config" in them, sorted by most recent"""
        relevant_files = []
        code_extensions = { '.js', '.ts'}
        
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            logger.info("Files: %s", files)
            for file in files:
                if Path(file).suffix in code_extensions and "config" not in file.lower():
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, repo_path)
                    
                    if _is_file_relevant(query, rel_path, file_path):
                        logger.info("Relevant file: %s", rel_path)
                        # Store tuple of (relative_path, modification_time) for sorting
                        mod_time = os.path.getmtime(file_path)
                        relevant_files.append((rel_path, mod_time))
        
        # Sort by modification time (most recent first) and return only the paths
        relevant_files.sort(key=lambda x: x[1], reverse=True)
        return [file_path for file_path, _ in relevant_files]
    
    def _is_file_relevant(query: str, rel_path: str, full_path: str) -> bool:
        """Simple relevance check"""
        query_lower = query.lower()
        query_keywords = query_lower.split()
        
        # Check file path
        if any(keyword in rel_path.lower() for keyword in query_keywords):
            return True
        
        # Check file content
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
                if any(keyword in content for keyword in query_keywords[:3]):
                    return True
        except:
            pass
        
        return False
    
    async def _should_modify_file(llm, query: str, rel_path: str, repo_path: str) -> tuple[bool, str]:
        """Determine if file should be modified and create plan"""
        full_path = os.path.join(repo_path, rel_path)
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Truncate content for analysis
            if len(content) > 2000:
                content = content[:1000] + "\n... [TRUNCATED] ...\n" + content[-1000:]
            
            prompt = PromptTemplate.from_template("""
                Analyze if this file needs modification for the given task.

                TASK: {query}
                FILE: {file_path}
                CONTENT:
                {content}

                Respond in this format:
                MODIFY: YES/NO
                PLAN: Brief description of what changes are needed (if YES)

                Be conservative - only respond YES if changes are genuinely required.
                """)
            
            chain = prompt | llm | StrOutputParser()
            response = await chain.ainvoke({
                "query": query,
                "file_path": rel_path,
                "content": content
            })
            logger.info("LLM Response: %s", response)
            
            lines = response.strip().split('\n')
            modify_line = next((line for line in lines if line.startswith('MODIFY:')), 'MODIFY: NO')
            plan_line = next((line for line in lines if line.startswith('PLAN:')), 'PLAN: No changes needed')
            
            should_modify = 'YES' in modify_line.upper()
            plan = plan_line.replace('PLAN:', '').strip()
            
            return should_modify, plan
            
        except Exception as e:
            logger.error("Error analyzing file %s: %s", rel_path, e)
            return False, "Analysis failed"
    
    try:
        yield FunctionInfo.from_fn(
            _analyze_code,
            description="Code analysis agent: find relevant files and create modification plan",
            input_schema=CodeAnalysisRequest
        )
    except GeneratorExit:
        logger.exception("Code analysis agent exited early!")

