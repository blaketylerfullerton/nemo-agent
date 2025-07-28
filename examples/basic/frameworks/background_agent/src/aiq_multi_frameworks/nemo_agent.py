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
    """Configuration for the Nemo Agent tool.

    Attributes
    ----------
    repo_path : str
        Path to the local repository to scan and edit.
    top_k : int
        Number of similar files to edit for a given prompt.
    """

    repo_path: str = Field(..., description="Path to the local code repository that will be scanned and edited.")
    top_k: int = Field(5, description="Number of similar files to edit for each user prompt.")


@register_function(config_type=NemoAgentConfig)
async def nemo_agent_as_tool(tool_config: NemoAgentConfig, builder: Builder):
    """Builds a function that analyses a code repository and applies code edits based on a natural-language prompt.

    The returned function (_arun) uses pre-computed embeddings from the background agent. It will:

    1.   Use cached embeddings and FAISS index from the background agent.
    2.   Search for the most similar files using the cached index.
    3.   For each similar file, ask GPT-4o to return the updated file content in accordance with the user prompt.
    4.   Overwrite the original files with the edited content and return a summary of the operation.
    """

    # Initialize the OpenAI client once so we can reuse it between calls
    client = OpenAI(api_key=os.getenv("NVIDIA_API_KEY"),
                     base_url = "https://integrate.api.nvidia.com/v1"
                     )

    def generate_code_edit(file_content: str, user_prompt: str) -> str:
        """Ask GPT-4o to apply *user_prompt* to *file_content* and return the edited file."""
        system_prompt = (
            "You are a code editor. Respond ONLY with the complete, final file content as plain code. "
            "DO NOT include code fences, language identifiers, comments, or explanations. "
            "Return the output exactly as it would appear in the source file."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Instruction: {user_prompt}\n\nFile content:\n{file_content}"},
        ]

        response = client.chat.completions.create(
            model="nvidia/llama-3.3-nemotron-super-49b-v1",
            messages=messages
        )
        response_text = response.choices[0].message.content.strip()
        return response_text

    def write_updated_file(path: str, new_content: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)

    # ---------------------------
    # The function that will be exposed as the tool
    # ---------------------------

    async def _arun(user_prompt: str) -> str:
        """Apply *user_prompt* to the files in *repo_path* and return a summary of the changes made."""
        try:
            logger.info("[Nemo Agent] Received prompt: %s", user_prompt)

            # 1. Check for cached embeddings
            repo_path = tool_config.repo_path
            cached_data = get_cached_embeddings(repo_path)
            
            if not cached_data:
                return f"❌ No cached embeddings found for repository '{repo_path}'. Please ensure the repository has been processed by the background agent first."

            # 2. Use cached code files and search for similar files
            code_files = cached_data['code_files']
            relevant_files = search_similar_files(repo_path, user_prompt, k=tool_config.top_k)

            if not relevant_files:
                return "No similar files found to apply the edit."

            # 3. Edit files and track changes
            changes_made = []
            for path in relevant_files:
                if path in code_files:
                    original = code_files[path]
                    edited = generate_code_edit(original, user_prompt)
                    
                    # Check if the content actually changed
                    if original.strip() == edited.strip():
                        logger.info(f"No changes needed for {path} - content is already up to date")
                        continue
                    
                    # Check if this edit might have already been applied
                    # by looking for keywords from the prompt in the original content
                    prompt_keywords = user_prompt.lower().split()
                    original_lower = original.lower()
                    
                    # Simple heuristic: if many prompt keywords are already in the file,
                    # it might already be modified
                    keyword_matches = sum(1 for keyword in prompt_keywords if keyword in original_lower)
                    if keyword_matches > len(prompt_keywords) * 0.6:  # 60% of keywords found
                        logger.info(f"File {path} may already contain requested changes - checking for actual differences")
                        
                        # Additional check: if the LLM returns nearly identical content, skip
                        similarity_ratio = len(set(original.split()) & set(edited.split())) / max(len(original.split()), len(edited.split()), 1)
                        if similarity_ratio > 0.95:  # 95% similar
                            logger.info(f"Skipping {path} - content appears to already have the requested changes")
                            continue
                    
                    write_updated_file(path, edited)
                    
                    # Update the cached content to keep it in sync
                    update_cached_file(repo_path, path, edited)
                    
                    # Track the before/after changes
                    relative_path = os.path.relpath(path, repo_path)
                    changes_made.append({
                        'file': relative_path,
                        'before': original,
                        'after': edited
                    })
                    
                    logger.info(f"Successfully updated {relative_path}")
                else:
                    logger.warning(f"File {path} not found in cached code files")

            # 4. Format the response with before/after changes
            if not changes_made:
                return "No files were updated."
            
            result_parts = [f"Successfully updated {len(changes_made)} file(s)\n"]
            
        
            logger.info("[Nemo Agent] Successfully updated %d files using cached embeddings.", len(changes_made))
            return "\n".join(result_parts)

        except Exception as exc:
            logger.exception("[Nemo Agent] Failed with exception: %s", exc, exc_info=True)
            return f"❌ Unexpected error: {exc}"

    # Expose the function as an AIQ Toolkit Function
    yield FunctionInfo.from_fn(_arun, description="Analyse a repository and apply code edits based on a natural-language prompt using cached embeddings.")
