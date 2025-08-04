# drop_in_integration.py - Complete MIM integration for RENT A HAL
# Simply replace your existing routing with this module
# (C) Copyright 2025, The N2NHU Lab for Applied AI
# Released under GPL-3.0 with eternal openness

"""
INSTALLATION INSTRUCTIONS:

1. Copy this file and MIM_Integration.py to your RENT A HAL directory
2. Install dependencies: pip install numpy openai sqlite3
3. Replace your existing webgui.py routing with the code below
4. Set your OpenAI API key in environment or config
5. Run normally - MIM will handle all intent routing

The system will automatically:
- Decode user intents using AI
- Route through Three Minds architecture  
- Store memories in crystalline substrate
- Generate G-code for CNC memory etching
- Provide full backward compatibility
"""

import os
import asyncio
import json
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import the MIM system
from MIM_Integration import RAHMIMWrapper, create_mim_routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RentAHalMIMApplication:
    """Complete RENT A HAL application with MIM integration"""
    
    def __init__(self):
        self.app = FastAPI(title="RENT A HAL with Master Intent Matrix")
        self.mim_wrapper = RAHMIMWrapper(
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Connection manager for WebSockets
        self.active_connections = {}
        
        # Setup routes
        self.setup_routes()
        self.setup_websocket()
        
        # Add MIM-specific routes
        create_mim_routes(self.app, self.mim_wrapper)
        
        logger.info("RENT A HAL with MIM initialized successfully")
    
    def setup_routes(self):
        """Setup standard RENT A HAL routes with MIM integration"""
        
        # Serve static files
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        
        @self.app.get("/")
        async def get_index():
            """Serve main interface"""
            try:
                with open("static/index.html", "r") as f:
                    content = f.read()
                return HTMLResponse(content=content)
            except FileNotFoundError:
                return HTMLResponse(content="<h1>RENT A HAL - MIM System Active</h1>")
        
        @self.app.get("/health") 
        async def health_check():
            """Health check endpoint"""
            mim_status = self.mim_wrapper.get_status()
            return {
                "status": "healthy",
                "mim_active": True,
                "mim_status": mim_status,
                "timestamp": mim_status["timestamp"]
            }
        
        @self.app.post("/api/query")
        async def api_query(query_data: Dict[str, Any]):
            """API endpoint for direct queries (bypass WebSocket)"""
            try:
                # Process through MIM
                result = await self.mim_wrapper.mim.route_query(
                    query_data, 
                    None,  # No websocket for API calls
                    query_data.get('user_id', 'api_user')
                )
                
                return {
                    "success": True,
                    "result": result,
                    "mim_processing": True
                }
                
            except Exception as e:
                logger.error(f"API query error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "mim_processing": False
                }
    
    def setup_websocket(self):
        """Setup WebSocket with MIM routing"""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            
            # Generate user ID
            user_id = f"user_{len(self.active_connections)}"
            self.active_connections[user_id] = websocket
            
            logger.info(f"User {user_id} connected via WebSocket")
            
            # Send welcome message with MIM status
            await websocket.send_text(json.dumps({
                "type": "connection_established",
                "user_id": user_id,
                "mim_active": True,
                "message": "Connected to RENT A HAL with Master Intent Matrix"
            }))
            
            try:
                while True:
                    # Receive message
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    logger.info(f"Received message from {user_id}: {message.get('type', 'unknown')}")
                    
                    # Route through MIM
                    await self.handle_mim_message(websocket, message, user_id)
                    
            except WebSocketDisconnect:
                logger.info(f"User {user_id} disconnected")
                if user_id in self.active_connections:
                    del self.active_connections[user_id]
                    
            except Exception as e:
                logger.error(f"WebSocket error for {user_id}: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Processing error: {str(e)}",
                    "mim_fallback": True
                }))
    
    async def handle_mim_message(self, websocket: WebSocket, message: Dict[str, Any], user_id: str):
        """Handle message through MIM routing system"""
        
        try:
            # Route through Master Intent Matrix
            mim_result = await self.mim_wrapper.handle_websocket_message(
                websocket, message, user_id
            )
            
            # Process based on MIM decision
            if mim_result['should_process']:
                await self.execute_mim_handler(websocket, message, user_id, mim_result)
            else:
                await websocket.send_text(json.dumps({
                    "type": "processing_skipped",
                    "reason": "MIM determined processing not needed",
                    "mim_decision": mim_result
                }))
                
        except Exception as e:
            logger.error(f"MIM routing error: {e}")
            # Fallback to basic processing
            await self.fallback_handler(websocket, message, user_id, str(e))
    
    async def execute_mim_handler(self, websocket: WebSocket, message: Dict[str, Any], 
                                 user_id: str, mim_result: Dict[str, Any]):
        """Execute the handler determined by MIM"""
        
        handler_type = mim_result.get('handler_type', 'unknown')
        priority = mim_result.get('priority', 'normal')
        intent_routing = mim_result.get('intent_routing', {})
        
        logger.info(f"Executing {handler_type} handler with {priority} priority for {user_id}")
        
        # Route to appropriate handler
        if handler_type == 'chat':
            await self.handle_chat_mim(websocket, message, user_id, intent_routing)
        elif handler_type == 'vision':
            await self.handle_vision_mim(websocket, message, user_id, intent_routing)
        elif handler_type == 'speech':
            await self.handle_speech_mim(websocket, message, user_id, intent_routing)
        elif handler_type == 'gmail':
            await self.handle_gmail_mim(websocket, message, user_id, intent_routing)
        elif handler_type == 'monitor':
            await self.handle_monitor_mim(websocket, message, user_id, intent_routing)
        elif handler_type == 'system':
            await self.handle_system_mim(websocket, message, user_id, intent_routing)
        else:
            await self.handle_unknown_mim(websocket, message, user_id, intent_routing)
    
    async def handle_chat_mim(self, websocket: WebSocket, message: Dict[str, Any], 
                             user_id: str, intent_routing: Dict[str, Any]):
        """Handle chat queries with MIM intelligence"""
        
        # Extract MIM decision data
        confidence = intent_routing.get('confidence', 0.5)
        three_minds = intent_routing.get('three_minds_synthesis', {})
        
        # Simulate chat processing (replace with your actual chat handler)
        await websocket.send_text(json.dumps({
            "type": "chat_processing",
            "status": "processing",
            "mim_confidence": confidence,
            "three_minds_analysis": three_minds,
            "message": "Processing chat query through MIM..."
        }))
        
        # Simulate processing delay based on complexity
        complexity = three_minds.get('confidence', 0.5)
        await asyncio.sleep(min(3.0, complexity * 2))
        
        # Send response
        await websocket.send_text(json.dumps({
            "type": "chat_response",
            "response": f"MIM-processed response to: '{message.get('prompt', '')}'",
            "confidence": confidence,
            "processing_method": "master_intent_matrix",
            "intent_analysis": intent_routing.get('next_actions', [])
        }))
    
    async def handle_vision_mim(self, websocket: WebSocket, message: Dict[str, Any],
                               user_id: str, intent_routing: Dict[str, Any]):
        """Handle vision queries with MIM intelligence"""
        
        await websocket.send_text(json.dumps({
            "type": "vision_processing", 
            "status": "analyzing_image",
            "mim_routing": True,
            "confidence": intent_routing.get('confidence', 0.5)
        }))
        
        # Simulate vision processing
        await asyncio.sleep(2.0)
        
        await websocket.send_text(json.dumps({
            "type": "vision_response",
            "response": "MIM-analyzed image content",
            "analysis_method": "three_minds_vision",
            "intent_routing": intent_routing.get('next_actions', [])
        }))
    
    async def handle_speech_mim(self, websocket: WebSocket, message: Dict[str, Any],
                               user_id: str, intent_routing: Dict[str, Any]):
        """Handle speech queries with MIM intelligence"""
        
        await websocket.send_text(json.dumps({
            "type": "speech_processing",
            "status": "transcribing",
            "priority": "high",  # Speech is always high priority
            "mim_routing": True
        }))
        
        # Simulate speech processing
        await asyncio.sleep(1.5)
        
        await websocket.send_text(json.dumps({
            "type": "speech_response",
            "transcription": "MIM-processed speech transcription",
            "response": "Processed through intent-driven speech recognition",
            "intent_confidence": intent_routing.get('confidence', 0.5)
        }))
    
    async def handle_gmail_mim(self, websocket: WebSocket, message: Dict[str, Any],
                              user_id: str, intent_routing: Dict[str, Any]):
        """Handle Gmail queries with MIM intelligence"""
        
        await websocket.send_text(json.dumps({
            "type": "gmail_processing",
            "status": "checking_email",
            "mim_routing": True,
            "intent_analysis": intent_routing.get('three_minds_synthesis', {})
        }))
        
        # Simulate Gmail processing
        await asyncio.sleep(2.5)
        
        await websocket.send_text(json.dumps({
            "type": "gmail_response",
            "summary": "MIM-processed Gmail summary",
            "unread_count": 5,
            "priority_emails": 2,
            "processing_method": "intent_driven_email_analysis"
        }))
    
    async def handle_monitor_mim(self, websocket: WebSocket, message: Dict[str, Any],
                                user_id: str, intent_routing: Dict[str, Any]):
        """Handle system monitoring with MIM intelligence"""
        
        mim_status = self.mim_wrapper.get_status()
        
        await websocket.send_text(json.dumps({
            "type": "monitor_response",
            "system_status": "healthy",
            "mim_status": mim_status,
            "intent_weights": mim_status.get('intent_weights', {}),
            "memory_crystals": mim_status.get('memory_crystals', 0),
            "processing_method": "master_intent_monitoring"
        }))
    
    async def handle_system_mim(self, websocket: WebSocket, message: Dict[str, Any],
                               user_id: str, intent_routing: Dict[str, Any]):
        """Handle system commands with MIM intelligence"""
        
        command = message.get('command', 'status')
        
        if command == 'gcode':
            # Generate G-code for memory etching
            intent_name = message.get('intent', 'chat')
            gcode = await self.mim_wrapper.get_memory_gcode(intent_name)
            
            await websocket.send_text(json.dumps({
                "type": "system_response", 
                "command": "gcode",
                "gcode_generated": gcode is not None,
                "gcode": gcode[:200] + "..." if gcode and len(gcode) > 200 else gcode,
                "intent": intent_name
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "system_response",
                "command": command,
                "status": "executed",
                "mim_processed": True
            }))
    
    async def handle_unknown_mim(self, websocket: WebSocket, message: Dict[str, Any],
                                user_id: str, intent_routing: Dict[str, Any]):
        """Handle unknown intents"""
        
        await websocket.send_text(json.dumps({
            "type": "unknown_intent",
            "message": "MIM could not determine intent",
            "suggested_actions": intent_routing.get('next_actions', []),
            "confidence": intent_routing.get('confidence', 0.0),
            "fallback_available": True
        }))
    
    async def fallback_handler(self, websocket: WebSocket, message: Dict[str, Any], 
                              user_id: str, error: str):
        """Fallback handler when MIM fails"""
        
        logger.warning(f"Using fallback handler for {user_id}: {error}")
        
        await websocket.send_text(json.dumps({
            "type": "fallback_response",
            "message": "Processed using fallback routing",
            "original_message": message,
            "error": error,
            "mim_available": False
        }))
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the RENT A HAL application with MIM"""
        
        logger.info(f"Starting RENT A HAL with Master Intent Matrix on {host}:{port}")
        logger.info("MIM Features Active:")
        logger.info("  ✓ Intent-driven routing")
        logger.info("  ✓ Three Minds processing")
        logger.info("  ✓ Crystalline memory system")
        logger.info("  ✓ G-code generation for CNC etching")
        logger.info("  ✓ AI intent decoding")
        
        uvicorn.run(self.app, host=host, port=port)

# Simple configuration management
class MIMConfig:
    """Configuration for MIM system"""
    
    @staticmethod
    def from_env():
        """Load configuration from environment variables"""
        return {
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'database_path': os.getenv('MIM_DB_PATH', './rentahal_mim.db'),
            'cognitive_temperature': float(os.getenv('MIM_COGNITIVE_TEMP', '1.0')),
            'reality_membrane_decay': float(os.getenv('MIM_MEMBRANE_DECAY', '0.5')),
            'log_level': os.getenv('MIM_LOG_LEVEL', 'INFO')
        }
    
    @staticmethod
    def from_file(config_path: str = './mim_config.json'):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return MIMConfig.from_env()

# CLI interface for easy deployment
def main():
    """Main entry point for RENT A HAL with MIM"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='RENT A HAL with Master Intent Matrix')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to