from rich.console import Console
from datetime import datetime
import inspect
import os

# Initialize console for colored output
console = Console()

# Log levels with their corresponding colors
LOG_LEVELS = {
    "DEBUG": "dim blue",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold red",
    "TOOL": "bright_magenta",
    "SYSTEM": "cyan",
    "MEMORY": "bright_blue",
    "MODEL": "bright_magenta",
    "USER": "bright_white",
    "SUCCESS": "bright_green",
    "PROCESSING": "bright_cyan"
}

class JarvisLogger:
    """A custom logger for the JARVIS assistant with colored output"""
    
    def __init__(self, log_to_file=False, log_file="jarvis.log"):
        self.console = Console()
        self.log_to_file = log_to_file
        self.log_file = log_file
        
        # Create log file if logging to file is enabled
        if self.log_to_file and not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write(f"JARVIS Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def _get_caller_info(self):
        """Get information about the caller function and file"""
        stack = inspect.stack()
        # Go back 3 frames to get the caller of the log method
        if len(stack) > 3:
            caller = stack[3]
            filename = os.path.basename(caller.filename)
            lineno = caller.lineno
            function = caller.function
            return f"{filename}:{function}:{lineno}"
        return "unknown:unknown:0"
    
    def _log(self, level, message, show_caller=True):
        """Internal logging method"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        caller = self._get_caller_info() if show_caller else ""
        
        # Format the log message
        if show_caller:
            log_message = f"[{LOG_LEVELS.get(level, 'white')}][{level}][/{LOG_LEVELS.get(level, 'white')}] [{timestamp}] [{caller}] {message}"
        else:
            log_message = f"[{LOG_LEVELS.get(level, 'white')}][{level}][/{LOG_LEVELS.get(level, 'white')}] [{timestamp}] {message}"
        
        # Print to console
        self.console.print(log_message)
        
        # Write to file if enabled
        if self.log_to_file:
            with open(self.log_file, 'a') as f:
                # Strip rich formatting for file logging
                plain_message = f"[{level}] [{timestamp}] [{caller}] {message}\n"
                f.write(plain_message)
    
    def debug(self, message, show_caller=True):
        """Log a debug message"""
        self._log("DEBUG", message, show_caller)
    
    def info(self, message, show_caller=True):
        """Log an info message"""
        self._log("INFO", message, show_caller)
    
    def warning(self, message, show_caller=True):
        """Log a warning message"""
        self._log("WARNING", message, show_caller)
    
    def error(self, message, show_caller=True):
        """Log an error message"""
        self._log("ERROR", message, show_caller)
    
    def critical(self, message, show_caller=True):
        """Log a critical message"""
        self._log("CRITICAL", message, show_caller)
    
    def tool(self, tool_name, action, show_caller=False):
        """Log a tool execution"""
        self._log("TOOL", f"Tool '{tool_name}' {action}", show_caller)
    
    def system(self, message, show_caller=False):
        """Log a system message"""
        self._log("SYSTEM", message, show_caller)
    
    def memory(self, action, context=None, show_caller=False):
        """Log a memory operation"""
        context_info = f" for context '{context}'" if context else ""
        self._log("MEMORY", f"{action}{context_info}", show_caller)
    
    def model(self, model_name, action, show_caller=False):
        """Log a model interaction"""
        self._log("MODEL", f"[{model_name}] {action}", show_caller)
    
    def user(self, message, show_caller=False):
        """Log user input"""
        self._log("USER", message, show_caller)
        
    def success(self, message, show_caller=False):
        """Log a success message"""
        self._log("SUCCESS", message, show_caller)
        
    def processing(self, message, show_caller=False):
        """Log a processing status message"""
        self._log("PROCESSING", message, show_caller)

# Create a global logger instance
logger = JarvisLogger()