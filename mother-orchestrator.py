"""
MOTHER Orchestrator - Master Orchestrator for Human-LLM Conversation

This module implements the MOTHER orchestration layer that enables:
1. Context-aware conversations via database history
2. Inter-LLM communication and coordination
3. Voice identity management
4. Dynamic intent routing

The MOTHER component sits between webgui.py and the AI worker nodes.
"""
import asyncio
import json
import logging
import sqlite3
import uuid
from typing import Dict, List, Optional, Union, Any, Set

from fastapi import WebSocket
from pydantic import BaseModel

# Setup logging
logger = logging.getLogger(__name__)

# --- Models ---
class LLMWorker(BaseModel):
    name: str
    address: str
    type: str
    voice_id: Optional[str] = None
    is_active: bool = True
    is_processing: bool = False
    current_user_guid: Optional[str] = None
    capabilities: List[str] = []

class User(BaseModel):
    guid: str
    nickname: str
    active_session: bool = True
    
class MotherMessage(BaseModel):
    sender: str  # User GUID or LLM name
    recipient: Optional[str] = None  # If None, broadcast to all
    content: str
    message_type: str = "text"  # text, audio, intent, system
    voice_output: bool = False
    debug_only: bool = False
    timestamp: float = 0.0

# --- MOTHER Orchestrator ---
class MOTHEROrchestrator:
    def __init__(self, db_path: str):
        """Initialize the MOTHER orchestrator."""
        self.db_path = db_path
        self.active_llms: Dict[str, LLMWorker] = {}
        self.active_users: Dict[str, User] = {}
        self.connections: Dict[str, WebSocket] = {}
        self.intent_router = None  # Will be assigned to an LLM
        self.default_voice = "v2/en_speaker_6"  # Default Bark voice
        
        # Message bus for inter-LLM communication
        self.mtor_bus: List[MotherMessage] = []
        self.mtor_subscriptions: Dict[str, Set[str]] = {}  # recipient -> set of subscribers
        
    async def start(self):
        """Start the MOTHER orchestrator service."""
        logger.info("Starting MOTHER orchestrator")
        
        # Initialize database connection
        self._init_db()
        
        # Designate an intent router
        await self._select_intent_router()
        
        # Start the heartbeat to monitor system health
        asyncio.create_task(self._heartbeat())
    
    def _init_db(self):
        """Initialize database for context tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS mother_context (
            id INTEGER PRIMARY KEY,
            user_guid TEXT,
            llm_name TEXT,
            context_data TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS mother_voices (
            llm_name TEXT PRIMARY KEY,
            voice_id TEXT,
            voice_params TEXT
        )
        """)
        
        conn.commit()
        conn.close()
    
    async def _select_intent_router(self):
        """Select an LLM to act as the intent router."""
        # Prefer models with better reasoning capabilities
        for name, llm in self.active_llms.items():
            if "reasoning" in llm.capabilities:
                self.intent_router = name
                logger.info(f"Selected {name} as intent router")
                return
                
        # Fallback to any available LLM
        if self.active_llms:
            self.intent_router = next(iter(self.active_llms.keys()))
            logger.info(f"Selected {self.intent_router} as intent router (fallback)")
    
    async def _heartbeat(self):
        """Periodic health check and maintenance."""
        while True:
            try:
                # Check LLM health
                for name, llm in list(self.active_llms.items()):
                    if not await self._check_llm_health(name):
                        logger.warning(f"LLM {name} is unhealthy, marking inactive")
                        llm.is_active = False
                
                # Process any pending messages
                await self._process_mtor_bus()
                
                # Clean up stale sessions
                self._cleanup_stale_sessions()
                
            except Exception as e:
                logger.error(f"Error in MOTHER heartbeat: {str(e)}")
            
            await asyncio.sleep(30)  # 30-second heartbeat
    
    async def _check_llm_health(self, llm_name: str) -> bool:
        """Check if an LLM worker is healthy."""
        # This would make a health check request to the worker
        # Simplified for this example
        return llm_name in self.active_llms and self.active_llms[llm_name].is_active
    
    async def _process_mtor_bus(self):
        """Process pending messages on the MTOR bus."""
        # Process messages in order
        while self.mtor_bus:
            message = self.mtor_bus.pop(0)
            
            if message.message_type == "intent":
                # Route to intent router
                await self._route_intent(message)
            elif message.recipient:
                # Direct message
                await self._deliver_message(message)
            else:
                # Broadcast
                await self._broadcast_message(message)
    
    async def _route_intent(self, message: MotherMessage):
        """Route an intent to the appropriate handler."""
        if not self.intent_router:
            await self._select_intent_router()
        
        if not self.intent_router:
            logger.error("No intent router available")
            return
        
        # Forward to the intent router LLM
        router_message = MotherMessage(
            sender="MOTHER",
            recipient=self.intent_router,
            content=f"INTENT_ROUTING_REQUEST: {message.content}",
            message_type="system"
        )
        
        await self._deliver_message(router_message)
    
    async def _deliver_message(self, message: MotherMessage):
        """Deliver a message to its recipient."""
        if message.recipient in self.active_llms:
            # Message to an LLM
            await self._deliver_to_llm(message)
        elif message.recipient in self.active_users:
            # Message to a user
            await self._deliver_to_user(message)
        elif message.recipient == "MOTHER":
            # Command for MOTHER
            await self._process_mother_command(message)
        else:
            logger.warning(f"Unknown recipient: {message.recipient}")
    
    async def _deliver_to_llm(self, message: MotherMessage):
        """Deliver a message to an LLM worker."""
        llm = self.active_llms.get(message.recipient)
        if not llm or not llm.is_active:
            logger.warning(f"LLM {message.recipient} is not available")
            return
        
        # Get context for this conversation
        context = self._get_context(message.sender, message.recipient)
        
        # Prepare the API request
        payload = {
            "prompt": message.content,
            "context": context,
            "type": "chat",
            "sender": message.sender
        }
        
        # This would make an API request to the LLM worker
        # Simplified for this example - in reality, you'd use your worker API
        response = {"response": f"Response from {message.recipient} to {message.content}"}
        
        # Create a response message
        response_message = MotherMessage(
            sender=message.recipient,
            recipient=message.sender,
            content=response["response"],
            voice_output=message.voice_output
        )
        
        # Update context
        self._update_context(message.sender, message.recipient, 
                            message.content, response["response"])
        
        # If voice output is requested, generate speech
        if message.voice_output:
            await self._generate_speech(response_message)
        
        # Deliver the response
        self.mtor_bus.append(response_message)
    
    async def _deliver_to_user(self, message: MotherMessage):
        """Deliver a message to a user via WebSocket."""
        user_guid = message.recipient
        if user_guid not in self.connections:
            logger.warning(f"User {user_guid} is not connected")
            return
        
        connection = self.connections[user_guid]
        
        # Prepare WebSocket message
        ws_message = {
            "type": "message",
            "sender": message.sender,
            "content": message.content
        }
        
        # If this is a voice message, include the audio data
        if hasattr(message, "audio_data") and message.audio_data:
            ws_message["audio"] = message.audio_data
            ws_message["type"] = "speech_result"
        
        # Send the message
        await connection.send_json(ws_message)
    
    async def _process_mother_command(self, message: MotherMessage):
        """Process a command directed at MOTHER."""
        command = message.content
        parts = command.split(":", 1)
        
        if len(parts) != 2 or parts[0] != "MOTHERREALM":
            logger.warning(f"Invalid MOTHER command: {command}")
            return
        
        action = parts[1].split("(", 1)[0]
        params_str = parts[1].split("(", 1)[1].rstrip(")")
        
        if action == "SPEECHOUT":
            # Enable speech output for specified entities
            recipients = params_str.split(",")
            for recipient in recipients:
                if recipient in self.active_users or recipient in self.active_llms:
                    # Mark for speech output
                    self._set_speech_preference(recipient, True)
                    
        elif action == "debugwindowoutONLYLLMONLYPRIVATECHAT":
            # Private debug chat between LLMs
            llms = params_str.split(",")
            await self._setup_private_llm_chat(llms, message.sender)
        
        else:
            logger.warning(f"Unknown MOTHER command action: {action}")
    
    def _set_speech_preference(self, entity_id: str, enabled: bool):
        """Set speech output preference for a user or LLM."""
        if entity_id in self.active_users:
            # TODO: Store user preference in database
            pass
        elif entity_id in self.active_llms:
            self.active_llms[entity_id].voice_enabled = enabled
    
    async def _setup_private_llm_chat(self, llms: List[str], initiator: str):
        """Set up a private chat between LLMs."""
        valid_llms = [llm for llm in llms if llm in self.active_llms]
        
        if not valid_llms:
            logger.warning("No valid LLMs for private chat")
            return
        
        # Create a group chat ID
        chat_id = f"private_chat_{uuid.uuid4()}"
        
        # Notify all participants
        for llm_name in valid_llms:
            init_message = MotherMessage(
                sender="MOTHER",
                recipient=llm_name,
                content=f"SYSTEM: You are now in a private debug chat with these LLMs: {', '.join(valid_llms)}. Initiated by: {initiator}. ChatID: {chat_id}",
                message_type="system",
                debug_only=True
            )
            
            await self._deliver_to_llm(init_message)
            
            # Add subscription to the group
            if llm_name not in self.mtor_subscriptions:
                self.mtor_subscriptions[llm_name] = set()
            
            for other_llm in valid_llms:
                if other_llm != llm_name:
                    self.mtor_subscriptions[llm_name].add(other_llm)
    
    async def _generate_speech(self, message: MotherMessage):
        """Generate speech for a message."""
        sender = message.sender
        voice_id = self.default_voice
        
        # Check if sender has a specific voice
        if sender in self.active_llms and self.active_llms[sender].voice_id:
            voice_id = self.active_llms[sender].voice_id
        
        # This would call your existing text-to-speech service
        # Simplified for this example
        audio_result = "base64_encoded_audio_data"  # In reality, call your TTS service
        
        # Attach the audio data to the message
        message.audio_data = audio_result
    
    async def _broadcast_message(self, message: MotherMessage):
        """Broadcast a message to all recipients."""
        # If debug only, send only to LLMs
        recipients = []
        if message.debug_only:
            recipients = list(self.active_llms.keys())
        else:
            recipients = list(self.active_llms.keys()) + list(self.active_users.keys())
        
        for recipient in recipients:
            if recipient != message.sender:  # Don't send to self
                msg_copy = message.copy()
                msg_copy.recipient = recipient
                await self._deliver_message(msg_copy)
    
    def _cleanup_stale_sessions(self):
        """Clean up stale user sessions and contexts."""
        # This would remove inactive users and clean up old contexts
        # Implementation depends on your session management strategy
        pass
    
    def _get_context(self, user_guid: str, llm_name: str) -> str:
        """Get conversation context for a user-LLM pair."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT context_data FROM mother_context
        WHERE user_guid = ? AND llm_name = ?
        ORDER BY timestamp DESC LIMIT 10
        """, (user_guid, llm_name))
        
        contexts = cursor.fetchall()
        conn.close()
        
        if not contexts:
            return ""
        
        # Combine contexts into a single string
        return ";\n".join([ctx[0] for ctx in contexts])
    
    def _update_context(self, user_guid: str, llm_name: str, query: str, response: str):
        """Update conversation context."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        context_data = json.dumps({
            "query": query,
            "response": response,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        cursor.execute("""
        INSERT INTO mother_context (user_guid, llm_name, context_data)
        VALUES (?, ?, ?)
        """, (user_guid, llm_name, context_data))
        
        conn.commit()
        conn.close()
    
    # --- Public API ---
    async def register_llm(self, llm: LLMWorker):
        """Register an LLM worker with MOTHER."""
        self.active_llms[llm.name] = llm
        
        # If no intent router yet, select one
        if not self.intent_router:
            await self._select_intent_router()
        
        # Announce new LLM to all active users
        announce = MotherMessage(
            sender="MOTHER",
            content=f"SYSTEM: New LLM {llm.name} has joined the system.",
            message_type="system"
        )
        
        await self._broadcast_message(announce)
        
        return True
    
    async def register_user(self, user: User, connection: WebSocket):
        """Register a user with MOTHER."""
        self.active_users[user.guid] = user
        self.connections[user.guid] = connection
        
        # Welcome message
        welcome = MotherMessage(
            sender="MOTHER",
            recipient=user.guid,
            content=f"Welcome, {user.nickname}! I am MOTHER, your AI orchestration system. "
                   f"There are {len(self.active_llms)} LLMs available to assist you.",
            message_type="system"
        )
        
        await self._deliver_to_user(welcome)
        
        return True
    
    async def unregister_user(self, user_guid: str):
        """Unregister a user from MOTHER."""
        if user_guid in self.active_users:
            del self.active_users[user_guid]
        
        if user_guid in self.connections:
            del self.connections[user_guid]
    
    async def process_message(self, message: MotherMessage):
        """Process an incoming message."""
        # Check for MOTHER commands
        if message.content.startswith("MOTHERREALM:"):
            message.recipient = "MOTHER"
            await self._process_mother_command(message)
            return
        
        # If no explicit recipient, route via intent
        if not message.recipient:
            message.message_type = "intent"
        
        # Add to MTOR bus for processing
        self.mtor_bus.append(message)
        
        # Process immediately
        await self._process_mtor_bus()
        
    async def set_llm_voice(self, llm_name: str, voice_id: str, voice_params: dict = None):
        """Set a specific voice for an LLM."""
        if llm_name not in self.active_llms:
            return False
            
        self.active_llms[llm_name].voice_id = voice_id
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT OR REPLACE INTO mother_voices (llm_name, voice_id, voice_params)
        VALUES (?, ?, ?)
        """, (llm_name, voice_id, json.dumps(voice_params or {})))
        
        conn.commit()
        conn.close()
        
        return True

# --- Integration with WebGUI ---
# This section shows how to integrate MOTHER with your existing webgui.py

mother = None  # Global MOTHER instance

async def initialize_mother(db_path):
    """Initialize the MOTHER orchestrator."""
    global mother
    mother = MOTHEROrchestrator(db_path)
    await mother.start()
    return mother

async def handle_websocket_connection(websocket: WebSocket, user_guid: str):
    """Handle WebSocket connection with MOTHER integration."""
    # Existing connection acceptance code...
    
    # Register user with MOTHER
    user = User(guid=user_guid, nickname=f"user_{user_guid[:8]}")
    await mother.register_user(user, websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            # Handle standard message types as before...
            
            # Check for MOTHER messages
            if "content" in data and isinstance(data["content"], str):
                content = data["content"]
                
                # Create a MOTHER message
                mother_msg = MotherMessage(
                    sender=user_guid,
                    content=content,
                    voice_output="voice_output" in data and data["voice_output"]
                )
                
                # Process through MOTHER
                await mother.process_message(mother_msg)
            
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {str(e)}")
    finally:
        # Unregister from MOTHER
        await mother.unregister_user(user_guid)

# --- Sample Usage Example ---

async def example_usage():
    """Example of how to use MOTHER."""
    # Initialize MOTHER
    mother = await initialize_mother("llm_broker.db")
    
    # Register LLMs
    llm1 = LLMWorker(
        name="LLAMA3",
        address="localhost:8000",
        type="chat",
        capabilities=["reasoning", "code"]
    )
    
    llm2 = LLMWorker(
        name="LLAVA",
        address="localhost:8001", 
        type="vision"
    )
    
    await mother.register_llm(llm1)
    await mother.register_llm(llm2)
    
    # Set voices
    await mother.set_llm_voice("LLAMA3", "v2/en_speaker_6")
    await mother.set_llm_voice("LLAVA", "v2/en_speaker_9")
    
    # User sends a message
    user_message = MotherMessage(
        sender="user_123",
        content="Can you help me understand how photosynthesis works?",
        voice_output=True
    )
    
    await mother.process_message(user_message)
    
    # LLM wants to collaborate with another LLM
    llm_message = MotherMessage(
        sender="LLAMA3",
        content="MOTHERREALM:debugwindowoutONLYLLMONLYPRIVATECHAT(LLAVA)",
        message_type="system"
    )
    
    await mother.process_message(llm_message)
    
    # Then the debug conversation can happen...

if __name__ == "__main__":
    asyncio.run(example_usage())
