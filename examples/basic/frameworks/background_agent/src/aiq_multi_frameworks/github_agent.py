import logging
import os
import subprocess
import time
import tempfile
import git
import os
import requests
from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig
import re
from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv("NVIDIA_API_KEY")
)

logger = logging.getLogger(__name__)

class GithubAgentConfig(FunctionBaseConfig, name="github_agent"):
    github_repo: str
    changes: str


@register_function(config_type=GithubAgentConfig)
async def github_agent_as_tool(tool_config: GithubAgentConfig, builder: Builder):

    async def _arun(commit_message: str) -> str:
        repo_path = tool_config.github_repo
        
        github_token = os.getenv("GITHUB_TOKEN")
        print("GITHUB TOKEN: ", github_token)
        
        # Check if there are any actual changes to commit
        try:
            repo = git.Repo(repo_path)
            if not repo.is_dirty(untracked_files=True):
                return "ℹ️ No changes detected in the repository. Nothing to commit."
            
            # Get a summary of what files were changed
            changed_files = []
            for item in repo.index.diff(None):
                changed_files.append(item.a_path)
            for item in repo.untracked_files:
                changed_files.append(item)
            
            logger.info(f"Detected changes in files: {changed_files}")
            
        except Exception as e:
            logger.warning(f"Could not check git status: {e}")

        #create commit body
        body = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[{
                "role": "user", 
                "content": f"""Based on the following code changes, generate a concise and descriptive commit message and PR body. 
                
                Format your response as:
                COMMIT_MESSAGE: [brief one-line description]
                PR_BODY: [detailed description of changes]

                Code changes:
                {tool_config.changes}
                
                Changed files: {', '.join(changed_files) if 'changed_files' in locals() else 'Unknown'}"""
            }],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=False
        )
        commit_body = body.choices[0].message.content
        logger.info("COMMIT BODY: %s", commit_body)
        
        # Parse commit message and PR body more robustly
        try:
            parsed_commit_message = commit_body.split("COMMIT_MESSAGE:")[1].split("PR_BODY:")[0].strip()
            pr_body = commit_body.split("PR_BODY:")[1].strip()
        except IndexError:
            # Fallback if parsing fails
            parsed_commit_message = f"AI Agent: {commit_message}"
            pr_body = f"Automated changes:\n\n{tool_config.changes}"
       
        try:
            logger.info("COMMITTING AND PUSHING")
            branch = commit_and_push(repo_path, parsed_commit_message)
            logger.info("COMMIT AND PUSH SUCCESS: %s", branch)
        except Exception as e:
            logger.info("GIT COMMIT ERROR: %s", e)
            return f"❌ Git commit error: {str(e)}"
        
        # Step 3: Create PR
        try:
            # Get the default branch dynamically
            default_branch = get_default_branch(repo_path)
            pr_url = create_pull_request(
                repo_owner=get_repo_owner(repo_path),
                repo_name=get_repo_name(repo_path),
                head_branch=branch,
                base_branch=default_branch,
                title=parsed_commit_message,
                body=pr_body,
                github_token=github_token
            )
            # Return the PR URL prominently
            return f" {parsed_commit_message} - {pr_body}\n\n ✅ Pull Request Created: {pr_url} \n\n"
        except Exception as e:
            # Still return commit info even if PR creation fails
            repo_owner = get_repo_owner(repo_path)
            repo_name = get_repo_name(repo_path)
            commit_url = f"https://github.com/{repo_owner}/{repo_name}/commits/{branch}"
            return f"✅ Changes committed to branch '{branch}': {commit_url}\n❌ PR creation failed: {str(e)}"

    yield FunctionInfo.from_fn(_arun, description="Validate repo, commit changes, and open a pull request.")


def run_shell(command, cwd):
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=90)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "Command timed out"}


def validate_node_app(path):
    steps = [
        "npm install",
        "npm run build"
    ]
    for cmd in steps:
        result = run_shell(cmd, cwd=path)
        if not result["success"]:
            return False, f"`{cmd}` failed:\n{result['stderr']}"
    return True, "✅ Validation passed."


def commit_and_push(path, message):
    logger.info("COMMITTING AND PUSHING")
    repo = git.Repo(path)
    branch = f"agent-edit-{int(time.time())}"
    repo.git.checkout("HEAD", b=branch)
    repo.git.add("--all")
    repo.index.commit(message)
    origin = repo.remote(name="origin")
    origin.push(branch)
    return branch

def clone_github_repo(repo_url: str) -> str:
    logger.info("CLONING GITHUB REPO")
    # Use tempfile to create a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="agent-repo-")
    repo = git.Repo.clone_from(repo_url, temp_dir)
    return temp_dir


def extract_github_url(text: str) -> str | None:
    logger.info("EXTRACTING GITHUB URL")
    match = re.search(r"(https?://github\.com/[\w\-]+/[\w\-]+)", text)
    return match.group(1) if match else None

def get_repo_owner(repo_path):
    logger.info("GETTING REPO OWNER")
    # Use GitPython to extract origin URL and parse owner
    repo = git.Repo(repo_path)
    origin_url = next(repo.remote(name="origin").urls)
    parts = origin_url.rstrip(".git").split("/")
    return parts[-2]


def get_repo_name(repo_path):
    logger.info("GETTING REPO NAME")
    repo = git.Repo(repo_path)
    origin_url = next(repo.remote(name="origin").urls)
    return origin_url.rstrip(".git").split("/")[-1]


def get_default_branch(repo_path):
    """Get the default branch of the repository"""
    logger.info("GETTING DEFAULT BRANCH")
    repo = git.Repo(repo_path)
    try:
        # Try to get the default branch from remote HEAD
        origin = repo.remote(name="origin")
        return origin.refs.HEAD.ref.remote_head
    except Exception:
        # Fallback: check common branch names
        if "main" in [ref.name for ref in repo.refs]:
            return "main"
        elif "master" in [ref.name for ref in repo.refs]:
            return "master"
        else:
            # If neither main nor master exists, use the first branch
            branches = [ref.name for ref in repo.refs if not ref.name.startswith('origin/')]
            return branches[0] if branches else "main"


def check_existing_prs(repo_owner, repo_name, title, github_token):
    """Check if there are existing open PRs with similar titles"""
    logger.info("CHECKING FOR EXISTING PRS")
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json"
    }
    params = {"state": "open"}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            existing_prs = response.json()
            # Check for PRs with similar titles (simple similarity check)
            title_words = set(title.lower().split())
            
            for pr in existing_prs:
                pr_title_words = set(pr["title"].lower().split())
                # If 70% of words match, consider it similar
                similarity = len(title_words & pr_title_words) / max(len(title_words), len(pr_title_words), 1)
                if similarity > 0.7:
                    return pr["html_url"]
            return None
        else:
            logger.warning(f"Failed to check existing PRs: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error checking existing PRs: {e}")
        return None


def create_pull_request(repo_owner, repo_name, head_branch, base_branch, title, body, github_token):
    logger.info("CREATING PULL REQUEST")
    
    # Check for existing similar PRs first
    existing_pr = check_existing_prs(repo_owner, repo_name, title, github_token)
    if existing_pr:
        logger.info(f"Found existing similar PR: {existing_pr}")
        return existing_pr
    
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json"
    }
    data = {
        "title": title,
        "head": head_branch,
        "base": base_branch,
        "body": body
    }
    response = requests.post(url, headers=headers, json=data)
    logger.info("GITHUB RESPONSE: %s", response)
    if response.status_code == 201:
        return response.json()["html_url"]
    raise Exception(f"GitHub API error: {response.status_code} - {response.text}")
