import platform
import datetime
import subprocess
import os

def get_current_datetime():
    """Returns the current date and time."""
    return datetime.datetime.now().isoformat()

def get_system_platform():
    """Returns the operating system platform (e.g., 'Windows', 'Linux')."""
    return platform.system()

def run_shell_command(command: str):
    """Runs a shell command and returns its output or error.
    Security Note: Be extremely cautious about the commands executed.
    Restrict allowed commands if possible.
    Args:
        command: The shell command to execute.
    Returns:
        A dictionary containing 'output' or 'error'.
    """
    # Basic security check: prevent potentially harmful commands (example)
    # You should implement a more robust allowlist/denylist based on your needs.
    forbidden_commands = ['rm', 'del', 'shutdown', 'reboot'] # Example denylist
    if any(cmd in command.split() for cmd in forbidden_commands):
        return {"error": "Execution of this command is forbidden."}

    try:
        # Use subprocess.run for better control and security
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False, timeout=30)
        if result.returncode == 0:
            return {"output": result.stdout.strip()}
        else:
            return {"error": result.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out after 30 seconds."}
    except Exception as e:
        return {"error": f"Failed to run command: {str(e)}"}

def list_directory_contents(path: str = '.', directory: str = None) :
    """Lists the contents of a specified directory.
    Args:
        path: The path to the directory (defaults to current directory).
        directory: Alias for path, for backward compatibility.
    Returns:
        A dictionary containing 'contents' (list of files/dirs) or 'error'.
    """
    # Use directory parameter if path is not provided (for backward compatibility)
    dir_path = directory if directory is not None else path
    
    try:
        if not os.path.isdir(dir_path):
            return {"error": f"Directory not found: {dir_path}"}
        contents = os.listdir(dir_path)
        return {"contents": contents}
    except Exception as e:
        return {"error": f"Failed to list directory contents: {str(e)}"}

# Add more tool functions here as needed