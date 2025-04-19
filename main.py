import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.prompt import Prompt
import json
import re # Add regex import
import jsonschema

# Import custom logging
from TOOLS.logging_utils import logger

# Import Brain modules
from BRAIN.gemini import ask_gemini
from BRAIN.grok import get_groq_response
from BRAIN.reasoning_deepseek import get_reasoning

# Import Database modules
from TOOLS.DATABASE.data import ConversationDB, ShortTermMemory

# Import RAG modules
from TOOLS.RAG.pdfs.pdf_analysis import analyze_pdf
from TOOLS.RAG.CSV.csv_analysis import analyze_csv
from TOOLS.RAG.audios.audio_process import process_audio
from TOOLS.RAG.videos.video_analysis import analyze_video

# Import TTS module
from TOOLS.AUDIO.tts import speak

# Import Tool functions
from functions import system_tools

import pygame

# Load environment variables
load_dotenv()

# Get API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize rich console
console = Console()
# Disable standard logging
import logging
logging.disable(logging.CRITICAL)

class JarvisAssistant:
    def __init__(self):
        # Initialize database components
        logger.system("Initializing JARVIS Assistant")
        self.long_term_memory = ConversationDB()
        self.short_term_memory = ShortTermMemory()
        self.current_context = "default"
        self.last_archive_time = datetime.now()
        self.auto_archive_interval = timedelta(hours=1)  # Archive contexts after 1 hour of inactivity
        self.warning_threshold = timedelta(minutes=45)  # Warn when context is 45 minutes inactive
        self.context_last_used = {}  # Track when contexts were last used
        self.context_metadata = self._load_context_metadata()  # Load context metadata
        logger.info("Database components initialized")
        
        # Initialize command executor
        # self.command_executor = CommandExecutor() # TODO: Define or import CommandExecutor
        
        # Available tools
        self.tools = {
            "get_current_datetime": system_tools.get_current_datetime,
            "get_system_platform": system_tools.get_system_platform,
            "run_shell_command": system_tools.run_shell_command,
            "list_directory_contents": system_tools.list_directory_contents
        }

        # System message for the assistant, including tool instructions
        self.system_message = f"""You are JARVIS, an advanced AI assistant.
        You have access to various tools and capabilities to help the user.
        Be concise, helpful, and informative in your responses.

        Available tools:
        {self._get_tool_descriptions()}

        **Tool Usage Rules**:
        1. For tool calls, respond ONLY with JSON like:
        {{
          "tool_call": {{
            "name": "tool_name",
            "arguments": {{ "arg_name": "arg_value" }}
          }}
        }}
        2. Never combine JSON with text
        3. Tools available: {list(self.tools.keys())}
        4. No markdown formatting, only pure JSON

        **Examples**:
        User: What's the time?
        Response: {{
          "tool_call": {{
            "name": "get_current_datetime",
            "arguments": {{}}
          }}
        }}

        User: List Desktop files
        Response: {{
          "tool_call": {{
            "name": "list_directory_contents",
            "arguments": {{
              "path": "~/Desktop"
            }}
          }}
        }}

        If the user's request does NOT require using a tool (e.g., a general question, conversation), then respond directly to the user in a conversational manner. Do NOT use the JSON format in this case.
        """

    def _get_tool_descriptions(self):
        """Generates a string describing available tools for the system prompt."""
        descriptions = []
        for name, func in self.tools.items():
            docstring = func.__doc__.strip() if func.__doc__ else "No description available."
            # Simple parsing for args if needed, or just use the docstring
            descriptions.append(f"- `{name}`: {docstring.splitlines()[0]}") # First line of docstring
        return "\n".join(descriptions)

    def _load_context_metadata(self):
        """Load context metadata from file"""
        metadata_path = Path("context_metadata.json")
        logger.debug(f"Loading context metadata from {metadata_path}")
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    logger.info(f"Loaded metadata for {len(metadata)} contexts")
                    return metadata
            except json.JSONDecodeError:
                error_msg = "Error loading context metadata. Starting with empty metadata."
                logger.error(error_msg)
                console.print(f"[bold red]{error_msg}[/bold red]")
        else:
            logger.info("No context metadata file found. Starting with empty metadata.")
        return {}
    
    def _save_context_metadata(self):
        """Save context metadata to file"""
        metadata_path = Path("context_metadata.json")
        logger.debug(f"Saving context metadata to {metadata_path}")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.context_metadata, f, indent=4)
            logger.info(f"Saved metadata for {len(self.context_metadata)} contexts")
        except IOError as e:
            error_msg = f"Error saving context metadata: {e}"
            logger.error(error_msg)
            console.print(f"[bold red]{error_msg}[/bold red]")

    def process_command(self, command):
        """Process user commands that start with '/' and return the output/response."""
        
        if command.startswith("/context"):
            # Change the current conversation context
            parts = command.split(" ", 1)
            if len(parts) == 1:
                logger.warning("Context command used without specifying a context name")
                return "Please specify a context name."
            
            new_context = parts[1].strip()
            logger.info(f"Changing context from '{self.current_context}' to '{new_context}'")
            self.current_context = new_context
            self.context_last_used[self.current_context] = datetime.now()
            
            # Check if context exists in long-term memory and load it
            self._load_context_from_long_term_memory(self.current_context)
            
            # Initialize metadata for this context if it doesn't exist
            if self.current_context not in self.context_metadata:
                logger.info(f"Creating new metadata for context '{self.current_context}'")
                self.context_metadata[self.current_context] = {
                    "tags": [],
                    "description": "",
                    "created_at": datetime.now().isoformat()
                }
                self._save_context_metadata()
            
            return f"Context changed to: {self.current_context}"
            
        elif command == "/archive":
            # Archive current context to long-term memory
            logger.memory("Archiving context to long-term memory", self.current_context)
            self.short_term_memory.archive_context(self.current_context, self.long_term_memory)
            return f"Context '{self.current_context}' archived to long-term memory"
            
        elif command == "/history":
            # Show conversation history for current context
            conversation = self.short_term_memory.get_conversation(self.current_context)
            if not conversation:
                # Try to load from long-term memory if not in short-term
                long_term_conversation = self._get_long_term_conversation(self.current_context)
                if long_term_conversation:
                    # Create a rich table for the history
                    table = Table(title=f"Long-term Conversation History: {self.current_context}")
                    table.add_column("Sender", style="cyan")
                    table.add_column("Message", style="white")
                    table.add_column("Time", style="green")
                    
                    for _, sender, message, timestamp in long_term_conversation:
                        table.add_row(sender, message, timestamp)
                    
                    console.print(table)
                    return ""
                return "No conversation history found for this context."
            
            # Create a rich table for the history
            table = Table(title=f"Conversation History: {self.current_context}")
            table.add_column("Sender", style="cyan")
            table.add_column("Message", style="white")
            table.add_column("Time", style="green")
            
            for timestamp, sender, message in conversation:
                table.add_row(sender, message, timestamp)
            
            console.print(table)
            return ""
            
        elif command == "/clear":
            # Clear current context
            logger.memory("Clearing context", self.current_context)
            self.short_term_memory.clear_context(self.current_context)
            return f"Context '{self.current_context}' cleared"
            
        elif command.startswith("/tag"):
            # Add tags to the current context
            parts = command.split(" ", 1)
            if len(parts) == 1:
                # Show current tags
                if self.current_context in self.context_metadata:
                    tags = self.context_metadata[self.current_context].get("tags", [])
                    if tags:
                        return f"Tags for context '{self.current_context}': {', '.join(tags)}"
                    else:
                        return f"No tags set for context '{self.current_context}'."
                return f"No metadata for context '{self.current_context}'."
            
            # Add new tags
            tags = [tag.strip() for tag in parts[1].split(",")]
            if self.current_context not in self.context_metadata:
                self.context_metadata[self.current_context] = {
                    "tags": tags,
                    "description": "",
                    "created_at": datetime.now().isoformat()
                }
            else:
                # Add new tags without duplicates
                current_tags = set(self.context_metadata[self.current_context].get("tags", []))
                current_tags.update(tags)
                self.context_metadata[self.current_context]["tags"] = list(current_tags)
            
            self._save_context_metadata()
            return f"Tags added to context '{self.current_context}': {', '.join(tags)}"
            
        elif command.startswith("/describe"):
            # Add description to the current context
            parts = command.split(" ", 1)
            if len(parts) == 1:
                # Show current description
                if self.current_context in self.context_metadata:
                    desc = self.context_metadata[self.current_context].get("description", "")
                    if desc:
                        return f"Description for context '{self.current_context}': {desc}"
                    else:
                        return f"No description set for context '{self.current_context}'."
                return f"No metadata for context '{self.current_context}'."
            
            # Set description
            description = parts[1].strip()
            if self.current_context not in self.context_metadata:
                self.context_metadata[self.current_context] = {
                    "tags": [],
                    "description": description,
                    "created_at": datetime.now().isoformat()
                }
            else:
                self.context_metadata[self.current_context]["description"] = description
            
            self._save_context_metadata()
            return f"Description set for context '{self.current_context}'"
            
        elif command.startswith("/search"):
            # Search contexts by tags or text
            parts = command.split(" ", 1)
            if len(parts) == 1:
                return "Please specify search terms."
            
            search_terms = parts[1].lower().strip().split()
            results = []
            
            # Search in metadata (tags and descriptions)
            for context, metadata in self.context_metadata.items():
                tags = [tag.lower() for tag in metadata.get("tags", [])]
                description = metadata.get("description", "").lower()
                
                # Check if any search term is in tags or description
                if any(term in tags or term in description for term in search_terms):
                    results.append((context, metadata))
            
            if not results:
                return "No matching contexts found."
            
            # Display results in a table
            table = Table(title="Search Results")
            table.add_column("Context", style="cyan")
            table.add_column("Tags", style="green")
            table.add_column("Description", style="white")
            
            for context, metadata in results:
                table.add_row(
                    context,
                    ", ".join(metadata.get("tags", [])),
                    metadata.get("description", "")
                )
            
            console.print(table)
            return ""
            
        elif command.startswith("/analyze_pdf"):
            # Analyze PDF document
            _, pdf_path = command.split(" ", 1)
            logger.tool("analyze_pdf", f"Analyzing PDF: {pdf_path.strip()}")
            with console.status("[bold green]Analyzing PDF...[/bold green]"):
                result = analyze_pdf(pdf_path.strip(), "Analyze this document and provide key insights", GEMINI_API_KEY)
            
            console.print(Panel(Markdown(result), title="PDF Analysis Result"))
            return ""
            
        elif command.startswith("/analyze_csv"):
            # Analyze CSV document
            _, csv_path = command.split(" ", 1)
            logger.tool("analyze_csv", f"Analyzing CSV: {csv_path.strip()}")
            with console.status("[bold green]Analyzing CSV data...[/bold green]"):
                result = analyze_csv(csv_path.strip(), "Analyze this CSV data and provide key insights", GEMINI_API_KEY)
            
            console.print(Panel(Markdown(result), title="CSV Analysis Result"))
            return ""
            
        elif command.startswith("/analyze_audio"):
            # Process audio file
            try:
                _, audio_path = command.split(" ", 1)
                # Remove any extra quotes and normalize path
                audio_path = audio_path.strip().strip('"')
                logger.tool("analyze_audio", f"Processing audio: {audio_path}")
                with console.status("[bold green]Processing audio...[/bold green]"):
                    result = process_audio(audio_path, "Transcribe and analyze this audio", GEMINI_API_KEY)
                
                console.print(Panel(Markdown(result), title="Audio Analysis Result"))
                return ""
            except Exception as e:
                return f"Error processing audio command: {str(e)}"
        
        elif command.startswith("/analyze_video"):
            # Analyze video file
            _, video_path = command.split(" ", 1)
            logger.tool("analyze_video", f"Analyzing video: {video_path.strip()}")
            with console.status("[bold green]Analyzing video...[/bold green]"):
                result = analyze_video(video_path.strip(), GEMINI_API_KEY, "Analyze this video and describe what's happening")
            
            console.print(Panel(Markdown(result), title="Video Analysis Result"))
            return ""
            
        elif command == "/help":
            # Show available commands
            help_text = """
            # Available Commands
            
            ## Context Management
            - `/context [name]` - Change conversation context
            - `/archive` - Archive current context to long-term memory
            - `/history` - Show conversation history for current context
            - `/clear` - Clear current context
            
            ## Metadata & Search
            - `/tag [tag1, tag2, ...]` - Add tags to current context
            - `/describe [description]` - Add description to current context
            - `/search [terms]` - Search contexts by tags or description
            
            ## Analysis Tools
            - `/analyze_pdf [path]` - Analyze PDF document
            - `/analyze_csv [path]` - Analyze CSV data
            - `/analyze_audio [path]` - Process audio file
            - `/analyze_video [path]` - Analyze video file
            
            ## Model Selection
            - `!reasoning [message]` - Use DeepSeek for step-by-step reasoning
            - `!gemini [message]` - Use Gemini model for response
            - Default uses Groq model
            
            """
            console.print(Markdown(help_text))
            return ""
            
        elif command == "/tools":
            # Show available tools
            table = Table(title="Available Tools")
            table.add_column("Tool Name", style="cyan")
            table.add_column("Description", style="white")
            for name, func in self.tools.items():
                docstring = func.__doc__.strip() if func.__doc__ else "No description available."
                table.add_row(name, docstring.splitlines()[0])
            console.print(table)
            return ""
            
        # Handle tool execution requests from LLM
        elif command.startswith("{") and command.endswith("}"):
            try:
                logger.debug("Attempting to parse tool call JSON")
                
                # Enhanced JSON extraction with fallback patterns
                json_pattern = r'```json\\s*({.*?})\\s*```'
                matches = re.findall(json_pattern, command, flags=re.DOTALL)
                if not matches:
                    json_pattern = r'(?s)^\s*(\{.*?\})\s*$'
                    matches = re.findall(json_pattern, command, flags=re.DOTALL)
                if not matches:
                    raise ValueError("No JSON object found in command")
                
                try:
                    # Clean and validate JSON
                    cleaned_json = matches[0].replace('```json', '').replace('```', '').strip()
                    validated_data = self.validate_json(cleaned_json)
                    tool_call = validated_data
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Invalid JSON format: {str(e)}")
                    return json.dumps({"error": "Invalid JSON format in tool call", "details": str(e)})
                
                # Validate tool call structure
                if 'tool_call' not in tool_call:
                    raise KeyError("Missing 'tool_call' in JSON structure")
                
                tool_name = tool_call["tool_call"].get("name")
                if not tool_name:
                    raise KeyError("Missing tool name in tool_call")
                
                if 'arguments' not in tool_call["tool_call"]:
                    raise KeyError("Missing arguments in tool_call")
                
                logger.tool(tool_name, "Executing tool call")
                logger.debug(f"Tool call details: {json.dumps(tool_call, indent=2)}")
                
                if tool_name in self.tools:
                    logger.processing(f"Executing tool: {tool_name}")
                    logger.debug(f"Tool arguments: {json.dumps(tool_call['tool_call']['arguments'], indent=2)}")
                    try:
                        # Log before execution
                        logger.info(f"Starting execution of tool: {tool_name}")
                        
                        # Execute tool
                        result = self.tools[tool_name](**tool_call["tool_call"]["arguments"])
                        console.print(f"[bold green]Response:[/bold green] {result}")
                        
                        # Log after successful execution
                        logger.success(f"Tool '{tool_name}' executed successfully")
                        logger.debug(f"Tool result: {json.dumps(result, indent=2) if isinstance(result, dict) else result}")
                        return json.dumps({"result": result})
                    except Exception as e:
                        # Log detailed error information
                        logger.error(f"Tool execution failed: {str(e)}")
                        logger.debug(f"Full error details: {str(e)}")
                        logger.debug(f"Failed tool call: {json.dumps(tool_call, indent=2)}")
                        return json.dumps({"error": str(e)})
                else:
                    logger.error(f"Unknown tool: {tool_name}")
                    logger.debug(f"Available tools: {list(self.tools.keys())}")
                    return json.dumps({"error": f"Unknown tool: {tool_name}"})
            except json.JSONDecodeError as e:
                logger.error(f"Invalid tool call JSON format: {str(e)}")
                logger.debug(f"Original command: {command}")
                return f"Invalid tool call format: {str(e)}. Please ensure response is valid JSON without markdown formatting"
            except Exception as e:
                logger.error(f"Tool execution error: {str(e)}")
                logger.debug(f"Full error details: {str(e)}")
                return json.dumps({"error": str(e)})

        return "Unknown command. Type /help for available commands."
    
    def validate_json(self, json_str):
        """Validate JSON structure using schema"""
        schema = {
            "type": "object",
            "properties": {
                "tool_call": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "object"}
                    },
                    "required": ["name", "arguments"]
                }
            },
            "required": ["tool_call"]
        }
        try:
            data = json.loads(json_str)
            jsonschema.validate(instance=data, schema=schema)
            return data
        except Exception as e:
            logger.error(f"JSON validation failed: {str(e)}")
            raise ValueError(f"Invalid tool call format: {str(e)}")

    def _get_long_term_conversation(self, context):
        """Retrieve conversation from long-term memory"""
        return self.long_term_memory.fetch_conversation(context)
    
    def _load_context_from_long_term_memory(self, context):
        """Load context from long-term memory if it exists"""
        long_term_conversation = self._get_long_term_conversation(context)
        if long_term_conversation:
            # Check if we already have this context in short-term memory
            short_term_conversation = self.short_term_memory.get_conversation(context)
            if not short_term_conversation:
                # Load the last 10 messages from long-term memory into short-term
                for _, sender, message, _ in long_term_conversation[-10:]:
                    self.short_term_memory.add_message(context, sender, message)
                return True
        return False
    
    def _check_and_archive_inactive_contexts(self):
        """Archive contexts that haven't been used for a while and warn about expiring contexts"""
        now = datetime.now()
        if now - self.last_archive_time < timedelta(minutes=5):  # Check every 5 minutes
            return  # Not time to check yet
            
        self.last_archive_time = now
        contexts_to_archive = []
        contexts_to_warn = []
        
        logger.debug("Checking for inactive contexts")
        
        # Find inactive and warning contexts
        for context, last_used in self.context_last_used.items():
            time_inactive = now - last_used
            
            if time_inactive > self.auto_archive_interval:
                contexts_to_archive.append(context)
                logger.debug(f"Context '{context}' marked for archiving (inactive for {time_inactive})")
            elif time_inactive > self.warning_threshold and context == self.current_context:
                contexts_to_warn.append((context, time_inactive))
                logger.debug(f"Context '{context}' marked for warning (inactive for {time_inactive})")
                
        # Warn about contexts nearing expiration
        for context, time_inactive in contexts_to_warn:
            minutes_left = int((self.auto_archive_interval - time_inactive).total_seconds() / 60)
            warning_msg = f"Warning: Current context '{context}' will be archived in {minutes_left} minutes due to inactivity."
            logger.warning(warning_msg, False)
            console.print(f"[bold yellow]{warning_msg}[/bold yellow]")
                
        # Archive inactive contexts
        for context in contexts_to_archive:
            if context != self.current_context:  # Don't archive current context
                self.short_term_memory.archive_context(context, self.long_term_memory)
                archive_msg = f"Auto-archived inactive context: {context}"
                logger.memory(f"Auto-archived inactive context", context)
                console.print(f"[bold blue]{archive_msg}[/bold blue]")
    
    def get_response(self, user_message, model="groq"):
        """Get response from the selected AI model"""
        # Update context last used time
        self.context_last_used[self.current_context] = datetime.now()
        
        # Log user message with color
        logger.user(f"[bold cyan]{user_message}[/bold cyan]")
        console.print(f"[dim blue]Processing user message in context '{self.current_context}'...[/dim blue]")
        
        # Save user message to short-term memory
        self.short_term_memory.add_message(self.current_context, "user", user_message)
        logger.memory(f"Added user message to memory", self.current_context)
        
        # Get conversation history for context
        conversation = self.short_term_memory.get_conversation(self.current_context)
        
        # If conversation is short, try to supplement with long-term memory
        if len(conversation) < 3:
            self._load_context_from_long_term_memory(self.current_context)
            conversation = self.short_term_memory.get_conversation(self.current_context)
        
        conversation_text = "\n".join([f"{sender}: {message}" for _, sender, message in conversation[-5:]])
        
        # Get context metadata if available
        context_info = ""
        if self.current_context in self.context_metadata:
            metadata = self.context_metadata[self.current_context]
            tags = metadata.get("tags", [])
            description = metadata.get("description", "")
            if tags or description:
                context_info = f"\nContext tags: {', '.join(tags)}\nContext description: {description}"
        
        # Prepare prompt with conversation history and metadata
        prompt = f"""
        Current conversation context: {self.current_context}{context_info}
        
        Recent conversation history:
        {conversation_text}
        
        User's latest message: {user_message}
        
        Please respond using EXACTLY ONE of these formats:
        - For tool usage: Valid JSON with tool_call object
        - For normal responses: Plain text
        """
        
        # Show status while waiting for response
        model_name = "Gemini" if model == "gemini" else "DeepSeek" if model == "reasoning" else "Groq"
        with console.status(f"[bold green]Getting response from {model_name}...[/bold green]"):
            # Play calculating sound
            calculating_sound_path = r"ASSETS\\SOUNDS\\calculating_sound.mp3"
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(calculating_sound_path)
                pygame.mixer.music.play()
            except Exception as e:
                console.print(f"[bold yellow]Error playing calculating sound: {e}[/bold yellow]")
            
            # Get response from selected model
            try:
                if model == "gemini":
                    logger.model("Gemini", "Generating response")
                    console.print(f"[bright_magenta]Sending prompt to Gemini model...[/bright_magenta]")
                    response = ask_gemini(prompt, GEMINI_API_KEY)
                    console.print(f"[bright_green]✓ Received response from Gemini[/bright_green]")
                elif model == "reasoning":
                    logger.model("DeepSeek", "Generating reasoning response")
                    console.print(f"[bright_magenta]Sending prompt to DeepSeek reasoning model...[/bright_magenta]")
                    response = get_reasoning(prompt, GROQ_API_KEY)
                    console.print(f"[bright_green]✓ Received reasoning response from DeepSeek[/bright_green]")
                else:  # Default to groq
                    logger.model("Groq", "Generating response")
                    console.print(f"[bright_magenta]Sending prompt to Groq model...[/bright_magenta]")
                    response = get_groq_response(
                        prompt, 
                        GROQ_API_KEY, 
                        system_message=self.system_message
                    )
                    console.print(f"[bright_green]✓ Received response from Groq[/bright_green]")
            finally:
                try:
                    pygame.mixer.music.stop()
                    pygame.mixer.quit()
                except Exception as e:
                    console.print(f"[bold yellow]Error stopping sound: {e}[/bold yellow]")
        
        # Save assistant's response to short-term memory
        self.short_term_memory.add_message(self.current_context, "assistant", response)
        
        # Speak the response
        # speak(response) # <-- REMOVE THIS LINE
        
        # Check if we should archive any inactive contexts
        self._check_and_archive_inactive_contexts()
        
        # Import the enhanced JSON extraction utility
        from TOOLS.json_utils import extract_tool_call, validate_tool_call
        
        # --- Tool Handling Logic ---
        max_tool_iterations = 3  # Prevent infinite loops with a reasonable limit
        iterations = 0
        final_response = response  # Start with the initial response
        
        while iterations < max_tool_iterations:
            iterations += 1
            logger.debug(f"Tool handling iteration {iterations}/{max_tool_iterations}")
            
            # Extract tool call using the enhanced utility function
            tool_call_data = extract_tool_call(final_response)
            
            if not tool_call_data:
                logger.debug("No tool call found in response, using as-is")
                break  # No tool call found, exit loop
            
            # Validate the tool call structure and check if tool exists
            if not validate_tool_call(tool_call_data, list(self.tools.keys())):
                logger.warning("Invalid tool call structure or unknown tool")
                break
                
            # Extract tool details
            tool_name = tool_call_data["tool_call"]["name"]
            tool_args = tool_call_data["tool_call"].get("arguments", {})
            
            logger.tool(tool_name, f"Executing with args: {json.dumps(tool_args)}")
            console.print(f"[bright_magenta]⚙️ Running tool: {tool_name}...[/bright_magenta]")

            # Tool call is valid, proceed with execution
            if tool_name in self.tools:
                    console.print(f"[bold yellow]Executing tool: {tool_name} with args: {tool_args}[/bold yellow]")
                    # Execute the tool
                    try:
                        tool_function = self.tools[tool_name]
                        # Ensure args are passed correctly, handle potential errors
                        tool_result = tool_function(**tool_args)
                        logger.tool(tool_name, "Execution completed")
                        console.print(f"[bright_green]✓ Tool execution completed: {tool_name}[/bright_green]")
                    except Exception as e:
                        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                        logger.error(error_msg)
                        console.print(f"[bold red]❌ Tool execution failed: {tool_name} - {str(e)}[/bold red]")
                        tool_result = {"error": error_msg}

                    console.print(f"[bold yellow]Tool result: {tool_result}[/bold yellow]")

                    # Add tool call and result to memory
                    self.short_term_memory.add_message(self.current_context, "assistant", json.dumps(tool_call_data)) # Save the tool request
                    # Convert tool_result to JSON string if it's not already (e.g., if it's a dict)
                    tool_result_str = json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
                    self.short_term_memory.add_message(self.current_context, "tool", tool_result_str) # Save the tool result
                    logger.memory("Added tool call and result to memory", self.current_context)

                    # Get updated conversation history including the tool result
                    conversation = self.short_term_memory.get_conversation(self.current_context)
                    # Include more history for context, ensure messages are strings
                    conversation_text = "\n".join([f"{sender}: {str(message)}" for _, sender, message in conversation[-10:]])

                    # Prepare a new prompt for the LLM with the tool result
                    prompt = f"""
                    Current conversation context: {self.current_context}{context_info}

                    Conversation history (including tool execution):
                    {conversation_text}

                    A tool was just executed ({tool_name}). The result is in the last 'tool' message.
                    Formulate the final response to the user's original request: {user_message}
                    Provide ONLY the final natural language response for the user, do NOT include the tool result directly unless it's the answer, and do NOT output any tool call JSON.
                    """
                    
                    # Call the LLM again with the tool result
                    logger.info(f"Getting follow-up response after tool execution")
                    console.print(f"[bright_cyan]🔄 Processing tool results and generating final response...[/bright_cyan]")
                    with console.status(f"[bold green]Getting final response from {model_name}...[/bold green]"):
                        if model == "gemini":
                            logger.model("Gemini", "Generating follow-up response")
                            console.print(f"[bright_magenta]Sending follow-up prompt to Gemini model...[/bright_magenta]")
                            final_response = ask_gemini(prompt, GEMINI_API_KEY)
                            console.print(f"[bright_green]✓ Received follow-up response from Gemini[/bright_green]")
                        elif model == "reasoning":
                            logger.model("DeepSeek", "Generating follow-up reasoning response")
                            console.print(f"[bright_magenta]Sending follow-up prompt to DeepSeek reasoning model...[/bright_magenta]")
                            final_response = get_reasoning(prompt, GROQ_API_KEY)
                            console.print(f"[bright_green]✓ Received follow-up reasoning response from DeepSeek[/bright_green]")
                        else:  # Default to groq
                            logger.model("Groq", "Generating follow-up response")
                            console.print(f"[bright_magenta]Sending follow-up prompt to Groq model...[/bright_magenta]")
                            final_response = get_groq_response(
                                prompt,
                                GROQ_API_KEY,
                                system_message=self.system_message # Use the original system message
                            )
                            console.print(f"[bright_green]✓ Received follow-up response from Groq[/bright_green]")
                    # Save the intermediate LLM response (after tool use)
                    self.short_term_memory.add_message(self.current_context, "assistant", final_response)
                    logger.memory("Added assistant follow-up response to memory", self.current_context)

            else:
                    # Invalid tool name, treat as normal response
                    logger.warning(f"Tool '{tool_name}' not found")
                    console.print(f"[bold red]LLM tried to call unknown tool: {tool_name}[/bold red]")


        # --- End Tool Handling Logic ---

        # Save the final assistant response (after potential tool use)
        # self.short_term_memory.add_message(self.current_context, "assistant", final_response) # Already saved inside the loop if tool was used, or initial response if not
        logger.memory("Added final assistant response to memory", self.current_context)
        console.print(f"[bright_blue]💾 Response saved to memory context: '{self.current_context}'[/bright_blue]")

        # Speak the final response
        console.print(f"[bright_cyan]🔊 Converting response to speech...[/bright_cyan]")
        speak(final_response)
        console.print(f"[bright_green]✓ Speech generation complete[/bright_green]")

        return final_response

    def chat_loop(self):
        """Main chat loop for the assistant"""
        logger.system("Starting JARVIS chat loop")
        console.print(Panel.fit(
            "[bold cyan]JARVIS AI Assistant[/bold cyan] initialized.\n"
            "Type [bold green]/help[/bold green] for available commands.\n"
            "Type [bold red]exit[/bold red] to quit.",
            title="Welcome"
        ))
        
        # Initialize context tracking
        self.context_last_used[self.current_context] = datetime.now()
        logger.info(f"Initial context set to '{self.current_context}'")
        logger.memory("Initialized context tracking", self.current_context)
        console.print(f"[bright_blue]🔄 Active context: '{self.current_context}'[/bright_blue]")
        
        while True:
            # Show context metadata in prompt if available
            context_display = self.current_context
            if self.current_context in self.context_metadata:
                tags = self.context_metadata[self.current_context].get("tags", [])
                if tags:
                    context_display += f" [dim]({', '.join(tags)})[/dim]"
            
            user_input = Prompt.ask(f"[bold cyan][{context_display}][/bold cyan] You")
            
            if user_input.lower() == "exit":
                logger.system("User requested exit")
                console.print("[bold red]Goodbye![/bold red]")
                break
                
            if user_input.startswith("/"):
                # Process command
                logger.system(f"Processing command: {user_input.split()[0]}")
                response = self.process_command(user_input)
                if response:  # Only print if there's a response (some commands handle their own output)
                    console.print(f"[bold green]JARVIS:[/bold green] {response}")
            elif user_input.startswith("!gemini "):
                # Use Gemini model explicitly
                logger.system("Using Gemini model explicitly")
                user_message = user_input[8:]
                response = self.get_response(user_message, model="gemini")
                console.print(Panel(Markdown(response), title="[bold green]JARVIS (Gemini)[/bold green]"))
            elif user_input.startswith("!reasoning "):
                # Use DeepSeek reasoning model explicitly
                logger.system("Using DeepSeek reasoning model explicitly")
                user_message = user_input[11:]
                response = self.get_response(user_message, model="reasoning")
                console.print(Panel(Markdown(response), title="[bold green]JARVIS (DeepSeek Reasoning)[/bold green]"))
            else:
                # Default to Groq model
                logger.system("Using default Groq model")
                response = self.get_response(user_input)
                console.print(Panel(Markdown(response), title="[bold green]JARVIS[/bold green]"))


def main():
    parser = argparse.ArgumentParser(description="JARVIS AI Assistant")
    parser.add_argument("--context", help="Initial conversation context", default="default")
    args = parser.parse_args()
    
    logger.system("Starting JARVIS AI Assistant")
    jarvis = JarvisAssistant()
    jarvis.current_context = args.context
    logger.info(f"Setting initial context to '{args.context}'")
    
    # Initialize context tracking
    jarvis.context_last_used[jarvis.current_context] = datetime.now()
    
    # Try to load context from long-term memory if it exists
    jarvis._load_context_from_long_term_memory(jarvis.current_context)
    logger.memory("Attempted to load context from long-term memory", jarvis.current_context)
    
    try:
        jarvis.chat_loop()
    except KeyboardInterrupt:
        logger.system("Received keyboard interrupt, shutting down")
        console.print("\n[bold red]JARVIS shutting down...[/bold red]")
    except Exception as e:
        error_msg = f"Error: {e}"
        logger.critical(error_msg)
        console.print(f"[bold red]{error_msg}[/bold red]")
    finally:
        logger.system("Performing cleanup before exit")
        # Save any pending changes before exit
        for context in jarvis.short_term_memory.memory:
            jarvis.short_term_memory._save_context(context)
            logger.memory(f"Saved context to disk", context)
        
        # Archive all active contexts to long-term memory
        for context in list(jarvis.short_term_memory.memory.keys()):
            jarvis.short_term_memory.archive_context(context, jarvis.long_term_memory)
            archive_msg = f"Archived context '{context}' to long-term memory"
            logger.memory("Archived context to long-term memory", context)
            console.print(f"[bold blue]{archive_msg}[/bold blue]")
        
        # Save context metadata
        jarvis._save_context_metadata()
        logger.system("JARVIS shutdown complete")

if __name__ == "__main__":
    main()