import logging
import os
from typing import Dict, List

from openai import OpenAI
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.builder.function_info import FunctionInfo

# Import the embedding functions from the background agent
from .register import get_cached_embeddings, search_similar_files, update_cached_file

logger = logging.getLogger(__name__)


class NemoAgentConfig(FunctionBaseConfig, name="nemo_agent"):
    nvidia_api_key: str = Field(default_factory=lambda: os.getenv("NVIDIA_API_KEY"))
    max_similar_files: int = Field(default=5, description="Maximum number of similar files to include in analysis")


def write_updated_file(path: str, new_content: str) -> None:
    """Write new content to a file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        logger.info(f"Successfully wrote updated content to {path}")
    except Exception as e:
        logger.error(f"Failed to write to {path}: {e}")
        raise


@register_function(config_type=NemoAgentConfig)
async def nemo_agent_as_tool(tool_config: NemoAgentConfig, builder: Builder):
    
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    async def _arun(user_prompt: str) -> str:
        """Apply *user_prompt* to the files in the repository and return a summary of the changes made."""
        try:
            # Search for relevant files using embeddings
            similar_files = search_similar_files(user_prompt, top_k=tool_config.max_similar_files)
            logger.info(f"Similar files: {similar_files}")
            
            if not similar_files:
                return "❌ No relevant files found in the repository."
            
            changes_made = []
            
            # Process each relevant file
            for file_path, similarity_score in similar_files:
                try:
                    # Read the current file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original_content = f.read()

                    # Create a prompt for generating the updated file content
                    edit_prompt = f"""You are a precise code editor that makes minimal, targeted changes.

                    TASK: Modify the file below based on the user's request while preserving all existing functionality.

                    USER REQUEST: {user_prompt}

                    CURRENT FILE:
                    ```
                    {original_content}
                    ```

                    INSTRUCTIONS:
                    1. Analyze the user's request and identify EXACTLY what needs to change
                    2. Make ONLY the minimal changes required - do not refactor unrelated code
                    3. Preserve all existing imports, structure, formatting, and comments unless they conflict with the request
                    4. If the request is unclear or could break the code, make the safest interpretation
                    5. Maintain consistent code style with the existing file

                    CRITICAL REQUIREMENTS:
                    - Return the COMPLETE file content (not just changed sections)
                    - Include ALL original code that should remain unchanged
                    - Do not add explanatory comments about your changes
                    - Do not modify variable names, function signatures, or logic unless explicitly requested
                    - Preserve exact indentation and spacing patterns

                    OUTPUT FORMAT:
                    Return ONLY the complete updated code. No markdown blocks, no explanations, no additional text - just the raw code that can be written directly to the file.

                    If you cannot safely implement the request without more context, return the original code unchanged."""

                    # Get the updated content from the LLM
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": edit_prompt}],
                        temperature=0.1,
                        max_tokens=4000
                    )
                    
                    updated_content = response.choices[0].message.content.strip()
                    
                    # Check if the content actually changed
                    if original_content.strip() == updated_content.strip():
                        logger.info(f"No changes needed for {file_path} - content is already up to date")
                        continue
                    
                    # Write the updated content to the file
                    write_updated_file(file_path, updated_content)
                    
                    # Track the changes
                    relative_path = os.path.relpath(file_path)
                    changes_made.append({
                        'file': relative_path,
                        'similarity_score': similarity_score
                    })
                    
                    logger.info(f"Successfully updated {relative_path}")
                    
                except Exception as e:
                    logger.warning(f"Could not process file {file_path}: {e}")
                    continue
            
            if not changes_made:
                return "❌ No files were modified. The requested changes may already be implemented or no suitable files were found."
            
            # Format the response
            files_changed = [change['file'] for change in changes_made]
            return f"✅ Successfully modified {len(changes_made)} file(s):\n" + "\n".join([f"  - {file}" for file in files_changed])

        except Exception as exc:
            logger.exception("[Nemo Agent] Failed with exception: %s", exc, exc_info=True)
            return f"❌ Unexpected error: {exc}"

    # Expose the function as an AIQ Toolkit Function
    yield FunctionInfo.from_fn(_arun, description="Analyse a repository and apply code edits based on a natural-language prompt using cached embeddings.")
