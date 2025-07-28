def clone_github_repo(repo_url: str) -> str:
    """Clone a GitHub repository to a folder named cloned_repos and return the path to the local clone."""
    import subprocess
    import os

    # Get the owner and repo name from the URL
    owner, repo = repo_url.split('/')[-2:]
    repo_path = os.path.join(os.getcwd(), 'cloned_repos', repo)

    # Ensure the cloned_repos directory exists
    os.makedirs('cloned_repos', exist_ok=True)

    # Clone the repository
    subprocess.run(['git', 'clone', repo_url, repo_path])
    return repo_path


def make_new_branch(repo_path: str, branch_name: str) -> str:
    """Make a new branch in the repository and return the path to the local clone."""
    import subprocess
    import os
    import logging

    logger = logging.getLogger(__name__)
    
    # Change to the repository directory    
    os.chdir(repo_path)

    try:
        # First, try to checkout main branch, if it doesn't exist, try master
        try:
            subprocess.run(['git', 'checkout', 'main'], check=True, capture_output=True)
            logger.info("Checked out to main branch")
        except subprocess.CalledProcessError:
            try:
                subprocess.run(['git', 'checkout', 'master'], check=True, capture_output=True)
                logger.info("Checked out to master branch")
            except subprocess.CalledProcessError:
                logger.warning("Could not checkout main or master, staying on current branch")

        # Pull the latest changes from the remote
        try:
            subprocess.run(['git', 'pull'], check=True, capture_output=True)
            logger.info("Pulled latest changes")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not pull latest changes: {e}")

        # Create and checkout the new branch
        subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
        logger.info(f"Created and checked out new branch: {branch_name}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating branch {branch_name}: {e}")
        raise

    return repo_path
    
