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

# File Modification Agent Configuration
class FileModificationAgentConfig(FunctionBaseConfig, name="file_modification_agent"):
    """Configuration for file modification agent"""
    llm: LLMRef = "nim_llm"
    temp_dir: str = "/tmp/code_editor_repos/"


# Input schema for file modification
class FileModificationRequest(BaseModel):
    """Input schema for file modification operations"""
    query: str = Field(description="The user query describing what code changes are needed")
    files_to_modify: List[str] = Field(description="List of file paths that need to be modified")
    modification_plan: Dict[str, Any] = Field(description="Plan for how each file should be modified")
    repo_path: str = Field(description="Path to the cloned repository")
    session_id: str = Field(description="Unique session identifier")


@register_function(config_type=FileModificationAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def file_modification_agent(config: FileModificationAgentConfig, builder: Builder):
    """File modification agent: actually modify the code files"""
    
    import os
    import re
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    logger.info("File modification agent config = %s", config)
    
    llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    async def _modify_files(request: FileModificationRequest) -> Dict[str, Any]:
        """Modify the specified files"""
        query = request.query
        files_to_modify = request.files_to_modify
        modification_plan = request.modification_plan
        repo_path = request.repo_path
        session_id = request.session_id
        
        modified_files = []
        errors = []
        
        try:
            for rel_path in files_to_modify:
                full_path = os.path.join(repo_path, rel_path)
                plan = modification_plan.get(rel_path, "Apply the requested changes")
                logger.info("Modifying file %s with plan: %s", rel_path, plan)
                success = await _modify_single_file(llm, query, plan, full_path, rel_path)
                logger.info("Success: %s", success)
                if success:
                    modified_files.append(rel_path)
                else:
                    errors.append(f"Failed to modify {rel_path}")
            
            return {
                "success": len(modified_files) > 0,
                "modified_files": modified_files,
                "errors": errors
            }
            
        except Exception as e:
            logger.exception("File modification failed")
            return {"success": False, "error": str(e)}
    
    async def _modify_single_file(llm, query: str, plan: str, full_path: str, rel_path: str) -> bool:
        """Modify a single file"""
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            prompt = PromptTemplate.from_template("""
                You are an expert code editor AI. Your task is to modify the given code file to accomplish the specified task.

TASK: {query}  
FILE PATH: {file_path}  
MODIFICATION PLAN: {plan}  

---  
CURRENT FILE CONTENT:  
{content}  
---

INSTRUCTIONS:  
- Apply only the necessary code modifications to complete the task.  
- If no changes are needed, return the **original file content exactly** as provided.  
- Do **not** include any comments, explanations, or annotations.  
- Your response must be the **full modified code file**, and nothing else.

OUTPUT FORMAT:  
Only return the complete, final code content. No markdown, no headers, no commentary.

                """)
            
            chain = prompt | llm | StrOutputParser()
            modified_content = await chain.ainvoke({
                "query": query,
                "file_path": rel_path,
                "plan": plan,
                "content": original_content
            })
            
            # Clean the response
            modified_content = _clean_llm_response(modified_content)
            
            # Only write if content changed
            if _content_changed(original_content, modified_content):
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                return True
            
            return False
            
        except Exception as e:
            logger.error("Error modifying file %s: %s", rel_path, e)
            return False
    
    def _clean_llm_response(response: str) -> str:
        """Clean LLM response"""
        response = re.sub(r'^```\w*\n', '', response, flags=re.MULTILINE)
        response = re.sub(r'\n```$', '', response, flags=re.MULTILINE)
        return response.strip()
    
    def _content_changed(original: str, modified: str) -> bool:
        """Check if content actually changed"""
        if original == modified:
            return False
        
        # Normalize whitespace
        orig_norm = re.sub(r'\s+', ' ', original.strip())
        mod_norm = re.sub(r'\s+', ' ', modified.strip())
        
        return orig_norm != mod_norm
    
    try:
        yield FunctionInfo.from_fn(
            _modify_files,
            description="File modification agent: modify code files based on analysis plan",
            input_schema=FileModificationRequest
        )
    except GeneratorExit:
        logger.exception("File modification agent exited early!")