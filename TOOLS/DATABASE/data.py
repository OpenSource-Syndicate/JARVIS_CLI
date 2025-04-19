import sqlite3
from typing import List, Tuple, Optional
from pathlib import Path
import logging
import os
from datetime import datetime
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set DB_PATH relative to the TOOLS/DATABASE directory
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "conversations.db"

class ConversationDB:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the database and create tables if they don't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        context TEXT NOT NULL,
                        sender TEXT NOT NULL,
                        message TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                # Create index for faster context-based queries
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_context 
                    ON conversations (context, timestamp)
                ''')
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")

    def save_message(
        self, 
        context: str, 
        sender: str, 
        message: str
    ) -> bool:
        """Save a message to the database with error handling"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO conversations (context, sender, message)
                    VALUES (?, ?, ?)
                ''', (context, sender, message))
            return True
        except sqlite3.Error as e:
            logger.error(f"Error saving message: {e}")
            return False

    def fetch_conversation(
        self, 
        context: str
    ) -> List[Tuple[int, str, str, str]]:
        """Retrieve conversation history for a context"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, sender, message, timestamp
                    FROM conversations
                    WHERE context = ?
                    ORDER BY timestamp ASC
                ''', (context,))
                return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error fetching conversation: {e}")
            return []

    def delete_conversation(
        self, 
        context: str
    ) -> bool:
        """Delete all messages for a specific context"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    DELETE FROM conversations 
                    WHERE context = ?
                ''', (context,))
            return True
        except sqlite3.Error as e:
            logger.error(f"Error deleting conversation: {e}")
            return False

class ShortTermMemory:
    def __init__(self, storage_dir: Path = Path("short_term_memory")):
        self.storage_dir = storage_dir
        self.memory = {}  # In-memory cache for active contexts
        self._init_storage()
        
    def _init_storage(self):
        """Create storage directory if it doesn't exist"""
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create storage directory: {e}")

    def _get_context_path(self, context: str) -> Path:
        """Get file path for a given context"""
        sanitized = context.replace('/', '_').replace('..', '')
        return self.storage_dir / f"{sanitized}.md"

    def _load_context(self, context: str) -> List[Tuple[str, str, str]]:
        """Load context from markdown file into memory"""
        file_path = self._get_context_path(context)
        messages = []
        
        if not file_path.exists():
            return messages
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Parse markdown content
            in_messages = False
            for line in content.split('\n'):
                if line.startswith("## Messages"):
                    in_messages = True
                    continue
                if in_messages and line.startswith("- ["):
                    parts = line[2:-1].split('](', 1)
                    if len(parts) >= 2:
                        sender_part, message_part = parts
                        timestamp = sender_part.split(', ')[-1]
                        sender = ', '.join(sender_part.split(', ')[:-1])
                        message = message_part.split(') ')[-1]
                        messages.append((timestamp, sender, message))
            return messages
            
        except Exception as e:
            logger.error(f"Error loading context {context}: {e}")
            return []

    def _save_context(self, context: str):
        """Save in-memory cache to markdown file"""
        if context not in self.memory:
            return
            
        file_path = self._get_context_path(context)
        messages = self.memory.get(context, [])
        
        try:
            with open(file_path, 'w') as f:
                f.write(f"# Context: {context}\n\n")
                f.write("## Messages\n")
                for msg in messages:
                    timestamp, sender, message = msg
                    f.write(f"- [{sender}, {timestamp}] {message}\n")
        except Exception as e:
            logger.error(f"Error saving context {context}: {e}")

    def add_message(self, context: str, sender: str, message: str):
        """Add message to short-term memory"""
        timestamp = datetime.now().isoformat()
        message_entry = (timestamp, sender, message)
        
        # Load from disk if not in memory
        if context not in self.memory:
            self.memory[context] = self._load_context(context)
            
        # Append to in-memory cache
        self.memory[context].append(message_entry)
        
        # Auto-save every 5 messages (adjust as needed)
        if len(self.memory[context]) % 5 == 0:
            self._save_context(context)

    def get_conversation(self, context: str) -> List[Tuple[str, str, str]]:
        """Get conversation history from short-term memory"""
        # Ensure context is loaded
        if context not in self.memory:
            self.memory[context] = self._load_context(context)
            
        return self.memory.get(context, [])

    def clear_context(self, context: str):
        """Clear short-term memory for a context"""
        if context in self.memory:
            del self.memory[context]
        file_path = self._get_context_path(context)
        if file_path.exists():
            file_path.unlink()

    def archive_context(self, context: str, db: ConversationDB):
        """Move context to long-term memory and clear short-term"""
        conversation = self.get_conversation(context)
        for msg in conversation:
            _, sender, message = msg
            db.save_message(context, sender, message)
            
        self.clear_context(context)
        logger.info(f"Archived {context} to long-term memory")

# Example usage integration
if __name__ == "__main__":
    # Initialize both memory systems
    db = ConversationDB()
    stm = ShortTermMemory()
    
    # Short-term interaction
    stm.add_message("weather", "user", "Will it rain today?")
    stm.add_message("weather", "assistant", "No rain expected")
    
    # Check short-term memory
    print("Short-term memory:")
    for msg in stm.get_conversation("weather"):
        print(f"{msg[0]} - {msg[1]}: {msg[2]}")
        
    # Archive to long-term
    stm.archive_context("weather", db)
    
    # Verify in long-term memory
    print("\nLong-term memory:")
    for msg in db.fetch_conversation("weather"):
        print(f"{msg[3]} - {msg[1]}: {msg[2]}")