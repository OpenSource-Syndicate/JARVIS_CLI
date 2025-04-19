import json
import re
import logging
from typing import Dict, Any, Optional

# Import custom logging if available
try:
    from TOOLS.logging_utils import logger
except ImportError:
    # Fallback to standard logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

def extract_tool_call(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract tool call JSON from LLM response using improved patterns.
    
    Args:
        response: The raw response from the LLM that might contain a tool call
        
    Returns:
        A validated tool call dictionary or None if no valid tool call found
    """
    patterns = [
        # Pattern 1: JSON in code block with possible whitespace
        r'(?s)```json\s*({.*?})\s*```',
        # Pattern 2: Bare JSON object with tool_call
        r'(?s){\s*"tool_call"\s*:\s*{\s*"name"\s*:\s*".*?",\s*"arguments"\s*:\s*{.*?}\s*}\s*}',
        # Pattern 3: Any valid JSON object containing tool_call
        r'(?s){.*?"tool_call".*?}'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, flags=re.DOTALL)
        if matches:
            for match in matches:
                json_str = match if pattern != patterns[0] else match
                try:
                    # Validate basic JSON structure
                    parsed = json.loads(json_str)
                    if "tool_call" in parsed and isinstance(parsed["tool_call"], dict):
                        # Validate tool call has required fields
                        tool_call = parsed["tool_call"]
                        if "name" in tool_call and isinstance(tool_call["name"], str):
                            # Ensure arguments is a dictionary (even if empty)
                            if "arguments" not in tool_call:
                                tool_call["arguments"] = {}
                            elif not isinstance(tool_call["arguments"], dict):
                                tool_call["arguments"] = {}
                            return parsed
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    logger.debug(f"Failed to parse potential tool call: {str(e)}")
                    continue
    
    return None

def validate_tool_call(tool_call: Dict[str, Any], available_tools: list) -> bool:
    """
    Validate that a tool call has the correct structure and references an available tool.
    
    Args:
        tool_call: The extracted tool call dictionary
        available_tools: List of available tool names
        
    Returns:
        True if the tool call is valid, False otherwise
    """
    if not isinstance(tool_call, dict) or "tool_call" not in tool_call:
        logger.error("Invalid tool call structure: missing 'tool_call' key")
        return False
    
    if not isinstance(tool_call["tool_call"], dict):
        logger.error("Invalid tool call structure: 'tool_call' is not a dictionary")
        return False
    
    if "name" not in tool_call["tool_call"]:
        logger.error("Invalid tool call structure: missing 'name' in tool_call")
        return False
    
    tool_name = tool_call["tool_call"]["name"]
    if not isinstance(tool_name, str):
        logger.error("Invalid tool call structure: 'name' is not a string")
        return False
    
    if tool_name not in available_tools:
        logger.error(f"Unknown tool: {tool_name}")
        return False
    
    if "arguments" not in tool_call["tool_call"]:
        logger.warning(f"Missing 'arguments' in tool call, using empty dict")
        tool_call["tool_call"]["arguments"] = {}
    elif not isinstance(tool_call["tool_call"]["arguments"], dict):
        logger.error("Invalid tool call structure: 'arguments' is not a dictionary")
        return False
    
    return True