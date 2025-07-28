import logging
import subprocess
from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.builder.function_info import FunctionInfo

logger = logging.getLogger(__name__)

class BrevAgentConfig(FunctionBaseConfig, name="brev_agent"):
    brev_repo: str  # Git repository URL to clone/pull

import os

@register_function(config_type=BrevAgentConfig)
async def brev_agent_as_tool(tool_config: BrevAgentConfig, builder: Builder):
    def run_cmd(cmd: str, allow_failure: bool = False) -> tuple[str, str, int]:
        """Run command and return (stdout, stderr, returncode)"""
        logger.info(f"Running command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        logger.info(f"Command return code: {result.returncode}")
        if result.stdout:
            logger.info(f"Command stdout: {result.stdout}")
        if result.stderr:
            logger.info(f"Command stderr: {result.stderr}")
        
        if not allow_failure and result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            logger.error(f"Command failed: {error_msg}")
            raise RuntimeError(error_msg)
        
        return result.stdout.strip(), result.stderr.strip(), result.returncode

    async def _arun(inputs: str) -> str:
        instance_name = "nvidia-demo"
        repo_url = tool_config.brev_repo
        
        # Extract repo name from URL for directory name
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        
        logger.info(f"Connecting to Brev instance: {instance_name}")
        logger.info(f"Repository URL: {repo_url}")
        logger.info(f"Repository directory: {repo_name}")
        
        try:
            # First, check if brev is installed and working
            logger.info("Checking if brev CLI is available...")
            try:
                stdout, stderr, returncode = run_cmd("brev --version", allow_failure=True)
                if returncode != 0:
                    return f"❌ Brev CLI not found or not working. Please ensure brev is installed and in your PATH.\nError: {stderr or stdout}"
                logger.info(f"Brev version: {stdout}")
            except Exception as e:
                return f"❌ Failed to check brev CLI: {str(e)}"
            
            # Check if the instance exists and is accessible
            logger.info(f"Checking if instance '{instance_name}' exists and is accessible...")
            try:
                stdout, stderr, returncode = run_cmd("brev ls", allow_failure=True)
                if returncode != 0:
                    return f"❌ Failed to list brev instances. Are you logged in?\nError: {stderr or stdout}"
                
                if instance_name not in stdout:
                    return f"❌ Instance '{instance_name}' not found in your brev instances.\nAvailable instances:\n{stdout}"
                
                logger.info(f"Instance '{instance_name}' found in brev list")
            except Exception as e:
                return f"❌ Failed to check brev instances: {str(e)}"
            
            # Test basic connectivity to the instance
            logger.info(f"Testing connectivity to instance '{instance_name}'...")
            cmd_format = None
            try:
                # Try different brev command formats to find the right one
                test_formats = [
                    f'brev shell {instance_name} --',
                    f'brev shell --name {instance_name} --',
                    f'brev ssh {instance_name}'
                ]
                
                for fmt in test_formats:
                    logger.info(f"Trying command format: {fmt}")
                    stdout, stderr, returncode = run_cmd(f'{fmt} echo CONNECTION_TEST_SUCCESS', allow_failure=True)
                    if returncode == 0 and "CONNECTION_TEST_SUCCESS" in stdout:
                        cmd_format = fmt
                        logger.info(f"Successfully found working command format: {fmt}")
                        break
                    else:
                        logger.info(f"Format failed: {fmt}, error: {stderr or stdout}")
                
                if not cmd_format:
                    return f"❌ Cannot connect to instance '{instance_name}' using any known command format.\nPlease check if the instance is running and accessible."
                
                logger.info(f"Successfully connected to instance '{instance_name}' using: {cmd_format}")
                    
            except Exception as e:
                return f"❌ Failed to test connection to instance: {str(e)}"
            
            # Now ensure Git is installed on the instance
            logger.info("Ensuring Git is installed on the instance...")
            try:
                stdout, stderr, returncode = run_cmd(f'{cmd_format} command -v git || (sudo apt update && sudo apt install -y git)')
                logger.info("Git check/installation completed successfully")
            except RuntimeError as e:
                return f"❌ Failed to ensure Git is installed on instance:\n{str(e)}"
            
            # Check if repo directory already exists
            logger.info(f"Checking if repository directory '{repo_name}' exists...")
            try:
                stdout, stderr, returncode = run_cmd(
                    f'{cmd_format} test -d {repo_name} && echo EXISTS || echo NOT_EXISTS',
                    allow_failure=True
                )
                
                if returncode != 0:
                    return f"❌ Failed to check directory on instance:\n{stderr or stdout}"
                
                if "EXISTS" in stdout:
                    # Directory exists, try to pull latest changes
                    logger.info(f"Repository directory exists, pulling latest changes...")
                    try:
                        output, _, _ = run_cmd(f'{cmd_format} cd {repo_name} && git pull')
                        return f"✅ `git pull` successful for {repo_name}\n📤 Output:\n{output}"
                    except RuntimeError as e:
                        return f"❌ Failed to pull repository:\n{str(e)}"
                else:
                    # Directory doesn't exist, clone the repo
                    logger.info(f"Repository directory doesn't exist, cloning...")
                    try:
                        output, _, _ = run_cmd(f'{cmd_format} git clone {repo_url}')
                        return f"✅ `git clone` successful for {repo_name}\n📤 Output:\n{output}"
                    except RuntimeError as e:
                        return f"❌ Failed to clone repository:\n{str(e)}"
                        
            except Exception as e:
                return f"❌ Unexpected error during repository operations: {str(e)}"
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Brev operation failed: {error_msg}")
            return f"❌ Unexpected error: {error_msg}"

    yield FunctionInfo.from_fn(_arun, description="SSH into Brev instance and clone/pull the specified Git repository")
