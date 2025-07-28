import logging
import datetime
from typing import Dict, Any, List, Optional
from aiq.data_models.function import FunctionBaseConfig
from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.function import FunctionBaseConfig
from aiq.builder.function_info import FunctionInfo
from pydantic import BaseModel, Field
import os
logger = logging.getLogger(__name__)

# GitHub Agent Configuration
class GitHubAgentConfig(FunctionBaseConfig, name="github_agent"):
    """Configuration for GitHub operations agent"""
    temp_dir: str = "/tmp/code_editor_repos/"
    repo_owner: str = "NVIDIA"
    repo_name: str = "Nemo-Agent-Toolkit"
    base_branch: str = "main"
    github_token: Optional[str] = os.getenv("GITHUB_TOKEN")


# Input schema for GitHub operations
class GitHubOperationRequest(BaseModel):
    """Input schema for GitHub operations"""
    action: str = Field(description="The action to perform: 'clone', 'commit_and_push', or 'cleanup'")
    session_id: str = Field(description="Unique session identifier")
    repo_url: str = Field(None, description="Repository URL (required for clone action)")
    branch: str = Field(default="main", description="Branch name (optional for clone action)")
    commit_message: str = Field(None, description="Commit message (required for commit_and_push action)")
    modified_files: List[str] = Field(default_factory=list, description="List of modified files (optional for commit_and_push action)")
    pr_title: str = Field(None, description="Pull request title (optional, defaults to commit message)")
    pr_description: str = Field(default="", description="Pull request description (optional)")


@register_function(config_type=GitHubAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def github_agent(config: GitHubAgentConfig, builder: Builder):
    """GitHub operations agent: clone, commit, push, cleanup"""

    import subprocess
    import tempfile
    import shutil
    import os
    import requests
    import json
    from pathlib import Path
    
    logger.info("GitHub agent config = %s", config)
    
    # Store active sessions
    active_sessions = {}
    
    async def _github_operations(request: GitHubOperationRequest) -> Dict[str, Any]:
        """Handle GitHub operations"""
        action = request.action
        session_id = request.session_id
        
        try:
            if action == "clone":
                return await _clone_repository(request, session_id)
            elif action == "commit_and_push":
                return await _commit_and_push(request, session_id)
            elif action == "cleanup":
                return await _cleanup_session(session_id)
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
                
        except Exception as e:
            logger.exception("GitHub operation failed")
            return {"success": False, "error": str(e)}
    
    async def _clone_repository(request: GitHubOperationRequest, session_id: str) -> Dict[str, Any]:
        """Clone repository to temporary directory"""
        repo_url = request.repo_url
        if not repo_url:
            return {"success": False, "error": "repo_url is required for clone action"}
            
        logger.info("Cloning repository from %s", repo_url)
        branch = request.branch or "main"
        logger.info("Branch: %s", branch)
        
        # Create session directory
        temp_dir = tempfile.mkdtemp(prefix=f"code_editor_{session_id}_")
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        repo_path = os.path.join(temp_dir, repo_name)
        
        # Clone repository
        clone_cmd = ["git", "clone", "-b", branch, repo_url, repo_path]
        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            # Try without branch specification
            clone_cmd = ["git", "clone", repo_url, repo_path]
            result = subprocess.run(clone_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return {"success": False, "error": f"Clone failed: {result.stderr}"}
            
            # Create branch if needed
            if branch != "main" and branch != "master":
                subprocess.run(["git", "checkout", "-b", branch], cwd=repo_path)
        
        # Store session info
        active_sessions[session_id] = {
            "temp_dir": temp_dir,
            "repo_path": repo_path,
            "repo_url": repo_url,
            "branch": branch,
            "repo_name": repo_name,
            "repo_owner": _extract_repo_owner(repo_url)
        }
        
        return {
            "success": True,
            "repo_path": repo_path,
            "temp_dir": temp_dir
        }
    
    def _extract_repo_owner(repo_url: str) -> str:
        """Extract repository owner from URL"""
        # Handle both https and ssh URLs
        if "github.com/" in repo_url:
            parts = repo_url.split("github.com/")[1].split("/")
            return parts[0]
        return "unknown"
    
    def _generate_branch_name(session_id: str) -> str:
        """Generate a unique branch name"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"agent-update-{session_id}-{timestamp}"
    
    async def _commit_and_push(request: GitHubOperationRequest, session_id: str) -> Dict[str, Any]:
        """Commit and push changes to a new branch, then create PR"""
        if session_id not in active_sessions:
            return {"success": False, "error": "Session not found"}
        
        session_info = active_sessions[session_id]
        repo_path = session_info["repo_path"]
        commit_message = request.commit_message
        if not commit_message:
            return {"success": False, "error": "commit_message is required for commit_and_push action"}
            
        modified_files = request.modified_files or []
        
        # Generate unique branch name
        new_branch = _generate_branch_name(session_id)
        
        # Configure git
        subprocess.run(["git", "config", "user.name", "Code Editor Agent"], cwd=repo_path)
        subprocess.run(["git", "config", "user.email", "agent@example.com"], cwd=repo_path)
        
        # Create and checkout new branch
        checkout_result = subprocess.run(["git", "checkout", "-b", new_branch], 
                                       cwd=repo_path, capture_output=True, text=True)
        
        if checkout_result.returncode != 0:
            return {"success": False, "error": f"Failed to create branch: {checkout_result.stderr}"}
        
        # Add files
        if modified_files:
            for file_path in modified_files:
                subprocess.run(["git", "add", file_path], cwd=repo_path)
        else:
            subprocess.run(["git", "add", "."], cwd=repo_path)
        
        # Check for changes
        status_result = subprocess.run(["git", "status", "--porcelain"], 
                                     cwd=repo_path, capture_output=True, text=True)
        
        if not status_result.stdout.strip():
            return {"success": True, "message": "No changes to commit", "commit_hash": None}
        
        # Commit
        commit_result = subprocess.run(["git", "commit", "-m", commit_message], 
                                     cwd=repo_path, capture_output=True, text=True)
        
        if commit_result.returncode != 0:
            return {"success": False, "error": f"Commit failed: {commit_result.stderr}"}
        
        # Get commit hash
        hash_result = subprocess.run(["git", "rev-parse", "HEAD"], 
                                   cwd=repo_path, capture_output=True, text=True)
        commit_hash = hash_result.stdout.strip()
        
        # Push to new branch
        push_result = subprocess.run(["git", "push", "origin", new_branch], 
                                   cwd=repo_path, capture_output=True, text=True)
        
        if push_result.returncode != 0:
            return {"success": False, "error": f"Push failed: {push_result.stderr}", 
                   "commit_hash": commit_hash, "branch": new_branch}
        
        # Create pull request
        pr_result = await _create_pull_request(
            session_info, 
            new_branch, 
            request.pr_title or commit_message,
            request.pr_description
        )
        
        result = {
            "success": True, 
            "commit_hash": commit_hash,
            "branch": new_branch,
            "push_successful": True
        }
        
        if pr_result["success"]:
            result["pull_request"] = pr_result["pr_data"]
            result["pr_url"] = pr_result["pr_data"]["url"]  # Add PR URL at top level
        else:
            result["pr_error"] = pr_result["error"]
            
        return result
    
    async def _create_pull_request(session_info: Dict, branch: str, title: str, description: str) -> Dict[str, Any]:
        """Create a pull request using GitHub API"""
        if not config.github_token:
            logger.warning("GitHub token not configured, cannot create PR")
            return {"success": False, "error": "GitHub token not configured"}
        
        repo_owner = session_info["repo_owner"]
        repo_name = session_info["repo_name"]
        base_branch = "main"  # Always target main branch for PRs
        
        # GitHub API endpoint
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls"
        
        headers = {
            "Authorization": f"token {config.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json"
        }
        
        data = {
            "title": title,
            "body": description,
            "head": branch,
            "base": base_branch
        }
        
        logger.info("Creating PR: %s -> %s in %s/%s", branch, base_branch, repo_owner, repo_name)
        logger.info("PR data: %s", data)
        
        try:
            response = requests.post(url, headers=headers, json=data)
            logger.info("PR API Response Status: %s", response.status_code)
            
            # Safely handle JSON response
            try:
                response_data = response.json()
                logger.info("PR Response: %s", response_data)
            except ValueError as json_error:
                logger.error("Failed to parse PR response as JSON: %s", json_error)
                logger.error("Raw response content: %s", response.text)
                return {"success": False, "error": f"Invalid JSON response from GitHub API: {response.text}"}
            
            if response.status_code == 201:
                return {
                    "success": True,
                    "pr_data": {
                        "number": response_data["number"],
                        "url": response_data["html_url"],
                        "title": response_data["title"]
                    }
                }
            else:
                logger.error("GitHub API error: %s - %s", response.status_code, response.text)
                return {"success": False, "error": f"GitHub API error: {response.status_code} - {response.text}"}
                
        except Exception as e:
            logger.exception("Failed to create PR: %s", str(e))
            return {"success": False, "error": f"Failed to create PR: {str(e)}"}
    
    async def _cleanup_session(session_id: str) -> Dict[str, Any]:
        """Clean up session resources"""
        if session_id in active_sessions:
            session_info = active_sessions[session_id]
            temp_dir = session_info["temp_dir"]
            
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                del active_sessions[session_id]
                return {"success": True, "message": "Session cleaned up"}
            except Exception as e:
                return {"success": False, "error": f"Cleanup failed: {str(e)}"}
        
        return {"success": True, "message": "Session not found, nothing to clean"}
    
    try:
        yield FunctionInfo.from_fn(
            _github_operations,
            description="GitHub operations agent: clone, commit, push, cleanup repositories with PR creation",
            input_schema=GitHubOperationRequest
        )
    except GeneratorExit:
        logger.exception("GitHub agent exited early!")
    finally:
        # Cleanup all active sessions
        for session_id in list(active_sessions.keys()):
            await _cleanup_session(session_id) 