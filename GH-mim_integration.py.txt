# MIM_Integration.py - Drop-in replacement for RENT A HAL routing
# Integrates Master Intent Matrix with existing webgui.py infrastructure
# (C) Copyright 2025, The N2NHU Lab for Applied AI
# Designer: J.P. Ames, N2NHU
# Architect: Claude (Anthropic)  
# Released under GPL-3.0 with eternal openness

import asyncio
import json
import sqlite3
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import openai
from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentType(Enum):
    CHAT = "chat"
    VISION = "vision" 
    SPEECH = "speech"
    GMAIL = "gmail"
    MONITOR = "monitor"
    SYSTEM = "system"

@dataclass
class IntentDecision:
    intent_name: str
    confidence: float
    priority: str
    reasoning: str
    next_actions: List[str]

@dataclass 
class QueryContext:
    user_id: str
    session_id: str
    timestamp: float
    query_type: str
    urgency: float
    complexity: float
    metadata: Dict[str, Any]

class IntentChatAI:
    """AI system for decoding user intents from natural language"""
    
    def __init__(self, api_key: str = None):
        self.client = openai.OpenAI(api_key=api_key) if api_key else None
        self.intent_patterns = self._load_intent_patterns()
        
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load known intent patterns for quick matching"""
        return {
            'chat': [
                'tell me about', 'explain', 'what is', 'how does', 'why',
                'can you help', 'i need to know', 'please describe'
            ],
            'vision': [
                'look at this', 'analyze image', 'what do you see', 'describe picture',
                'identify object', 'read text in image', 'analyze photo'
            ],
            'speech': [
                'listen to', 'transcribe', 'speech to text', 'voice input',
                'audio analysis', 'sound recognition'
            ],
            'gmail': [
                'check email', 'gmail summary', 'unread messages', 'email update',
                'mail check', 'inbox summary', 'email status'
            ],
            'system': [
                'system status', 'health check', 'monitor performance', 'stats',
                'system info', 'resource usage', 'diagnostics'
            ]
        }
    
    async def decode_intent(self, user_input: str, context: QueryContext) -> IntentDecision:
        """Decode user intent using AI analysis"""
        
        # Quick pattern matching first
        intent_scores = self._pattern_match(user_input.lower())
        
        # If clear match, use it
        if max(intent_scores.values()) > 0.8:
            primary_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[primary_intent]
            
            return IntentDecision(
                intent_name=primary_intent,
                confidence=confidence,
                priority=self._determine_priority(primary_intent, context),
                reasoning=f"Pattern match: {confidence:.2f}",
                next_actions=self._suggest_actions(primary_intent)
            )
        
        # Use AI for complex intent decoding
        if self.client:
            return await self._ai_decode_intent(user_input, context, intent_scores)
        else:
            # Fallback to best pattern match
            primary_intent = max(intent_scores, key=intent_scores.get)
            return IntentDecision(
                intent_name=primary_intent,
                confidence=intent_scores[primary_intent],
                priority=self._determine_priority(primary_intent, context),
                reasoning="Pattern-based fallback",
                next_actions=self._suggest_actions(primary_intent)
            )
    
    def _pattern_match(self, text: str) -> Dict[str, float]:
        """Pattern matching for intent recognition"""
        scores = {intent: 0.0 for intent in self.intent_patterns.keys()}
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    scores[intent] += 1.0 / len(patterns)
        
        # Normalize scores
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
            
        return scores
    
    async def _ai_decode_intent(self, user_input: str, context: QueryContext, 
                               pattern_scores: Dict[str, float]) -> IntentDecision:
        """Use OpenAI to decode complex intents"""
        
        prompt = f"""
        Analyze this user input and determine the primary intent:
        
        Input: "{user_input}"
        Context: {context.query_type}, urgency: {context.urgency}, complexity: {context.complexity}
        Pattern scores: {pattern_scores}
        
        Available intents: chat, vision, speech, gmail, system
        
        Respond with JSON:
        {{
            "intent": "primary_intent_name",
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation",
            "next_actions": ["action1", "action2"]
        }}
        """
        
        try:
            response = await self.client.chat.completions.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return IntentDecision(
                intent_name=result['intent'],
                confidence=result['confidence'],
                priority=self._determine_priority(result['intent'], context),
                reasoning=result['reasoning'],
                next_actions=result.get('next_actions', [])
            )
            
        except Exception as e:
            logger.error(f"AI intent decoding failed: {e}")
            # Fallback
            primary_intent = max(pattern_scores, key=pattern_scores.get)
            return IntentDecision(
                intent_name=primary_intent, 
                confidence=pattern_scores[primary_intent],
                priority="normal",
                reasoning="AI fallback",
                next_actions=[]
            )
    
    def _determine_priority(self, intent: str, context: QueryContext) -> str:
        """Determine query priority based on intent and context"""
        
        if context.urgency > 0.8:
            return "high"
        elif intent in ['speech', 'system'] or context.urgency > 0.6:
            return "high"
        elif intent in ['gmail', 'monitor']:
            return "normal" 
        else:
            return "normal"
    
    def _suggest_actions(self, intent: str) -> List[str]:
        """Suggest next actions based on intent"""
        actions = {
            'chat': ['process_query', 'generate_response', 'update_context'],
            'vision': ['analyze_image', 'extract_features', 'generate_description'],
            'speech': ['transcribe_audio', 'process_speech', 'respond_audio'],
            'gmail': ['check_inbox', 'summarize_emails', 'update_status'],
            'system': ['collect_metrics', 'generate_report', 'check_health']
        }
        return actions.get(intent, ['process_general'])

class CrystallineMemoryInterface:
    """Interface for crystalline temporal holographic memory system"""
    
    def __init__(self, db_path: str = "./rentahal_memory.db"):
        self.db_path = db_path
        self.init_database()
        self.cnc_queue = queue.Queue()
        self.gcode_generator = GCodeGenerator()
        
    def init_database(self):
        """Initialize crystalline memory database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Memory crystals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_crystals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent_name TEXT,
                timestamp REAL,
                state_vector BLOB,
                resonance_freq REAL,
                crystal_coords TEXT,
                gcode_hash TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # CNC operations log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cnc_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER,
                operation_type TEXT,
                gcode TEXT,
                status TEXT,
                execution_time REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(memory_id) REFERENCES memory_crystals(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def etch_memory(self, intent_name: str, state_vector: np.ndarray, 
                         context: Dict[str, Any]) -> str:
        """Etch memory into crystalline substrate"""
        
        # Calculate resonance frequency
        resonance_freq = self._calculate_resonance(state_vector)
        
        # Generate 3D coordinates
        coords = self._generate_crystal_coords(state_vector, resonance_freq)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO memory_crystals 
            (intent_name, timestamp, state_vector, resonance_freq, crystal_coords)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            intent_name,
            time.time(),
            state_vector.tobytes(),
            resonance_freq,
            json.dumps(coords)
        ))
        
        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Generate G-code for CNC etching
        gcode = await self.gcode_generator.generate_etching_gcode(coords, resonance_freq)
        
        # Queue for CNC execution
        self.cnc_queue.put({
            'memory_id': memory_id,
            'gcode': gcode,
            'intent': intent_name
        })
        
        logger.info(f"Etched memory crystal for intent '{intent_name}', ID: {memory_id}")
        return f"crystal_{memory_id}"
    
    def _calculate_resonance(self, state_vector: np.ndarray) -> float:
        """Calculate resonance frequency for memory recall"""
        # Use FFT to find dominant frequency
        fft = np.fft.fft(state_vector)
        freqs = np.fft.fftfreq(len(state_vector))
        dominant_freq = freqs[np.argmax(np.abs(fft))]
        return abs(dominant_freq) * 1000  # Scale to reasonable range
    
    def _generate_crystal_coords(self, state_vector: np.ndarray, 
                                resonance_freq: float) -> List[Dict[str, float]]:
        """Generate 3D coordinates for crystal etching"""
        coords = []
        base_radius = 10.0  # 10mm working area
        
        for i, value in enumerate(state_vector):
            # Map state values to 3D coordinates
            angle = (i / len(state_vector)) * 2 * np.pi
            radius = min(base_radius, abs(value) * base_radius)
            
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = (value * 0.1) + (np.sin(resonance_freq * angle) * 0.05)  # Add resonance modulation
            
            coords.append({
                'x': float(x),
                'y': float(y), 
                'z': float(z),
                'intensity': min(255, int(abs(value) * 255)),
                'frequency': resonance_freq
            })
        
        return coords
    
    async def recall_memory(self, query_vector: np.ndarray, 
                           threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Recall memories based on resonance matching"""
        
        query_resonance = self._calculate_resonance(query_vector)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, intent_name, timestamp, state_vector, resonance_freq, crystal_coords
            FROM memory_crystals
            ORDER BY timestamp DESC
            LIMIT 100
        ''')
        
        memories = []
        for row in cursor.fetchall():
            memory_id, intent_name, timestamp, state_blob, resonance_freq, coords_json = row
            
            # Calculate resonance similarity
            freq_similarity = 1.0 - abs(query_resonance - resonance_freq) / max(query_resonance, resonance_freq)
            
            if freq_similarity >= threshold:
                state_vector = np.frombuffer(state_blob, dtype=np.float64)
                
                # Calculate vector similarity
                vector_similarity = np.dot(query_vector, state_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(state_vector)
                )
                
                combined_similarity = (freq_similarity + abs(vector_similarity)) / 2
                
                if combined_similarity >= threshold:
                    memories.append({
                        'id': memory_id,
                        'intent': intent_name,
                        'timestamp': timestamp,
                        'similarity': combined_similarity,
                        'coordinates': json.loads(coords_json),
                        'resonance': resonance_freq
                    })
        
        conn.close()
        
        # Sort by similarity
        memories.sort(key=lambda x: x['similarity'], reverse=True)
        return memories[:5]  # Return top 5 matches

class GCodeGenerator:
    """Generate G-code for CNC crystalline memory etching"""
    
    def __init__(self):
        self.feed_rate = 800  # mm/min
        self.max_laser_power = 255
        self.safety_height = 5.0  # mm
    
    async def generate_etching_gcode(self, coordinates: List[Dict[str, float]], 
                                    resonance_freq: float) -> str:
        """Generate G-code for memory etching"""
        
        gcode = self._generate_header()
        
        # Home and prepare
        gcode += "G28 ; Home all axes\n"
        gcode += f"G0 Z{self.safety_height} ; Move to safety height\n"
        gcode += "M106 P0 ; Laser off\n\n"
        
        # Etch each coordinate point
        for i, coord in enumerate(coordinates):
            gcode += f"; Memory point {i+1} - Intensity: {coord['intensity']}\n"
            
            # Move to position
            gcode += f"G0 X{coord['x']:.3f} Y{coord['y']:.3f} ; Move to position\n"
            gcode += f"G1 Z{coord['z']:.3f} F{self.feed_rate} ; Move to etch depth\n"
            
            # Calculate pulse parameters based on resonance
            pulse_duration = 0.1 + (coord['frequency'] / 10000)  # Variable pulse time
            laser_power = min(self.max_laser_power, coord['intensity'])
            
            # Etch with resonance-modulated pulses
            gcode += f"M106 P{laser_power} ; Laser on\n"
            gcode += f"G4 P{pulse_duration:.3f} ; Etch pulse\n"
            gcode += "M106 P0 ; Laser off\n"
            
            # Return to safety height
            gcode += f"G0 Z{self.safety_height} ; Return to safety\n\n"
        
        # Finalize
        gcode += self._generate_footer()
        
        return gcode
    
    def _generate_header(self) -> str:
        """Generate G-code header"""
        return f"""
; RENT A HAL Crystalline Memory Etching
; Generated: {datetime.now().isoformat()}
; Master Intent Matrix (MIM) Memory System
;
G21 ; Set units to millimeters  
G90 ; Absolute positioning
G94 ; Feed rate per minute
M107 ; Fan off
M106 P0 ; Laser off

"""
    
    def _generate_footer(self) -> str:
        """Generate G-code footer"""
        return f"""
; Return home and finalize
G28 ; Home all axes
M106 P0 ; Laser off
M84 ; Disable steppers
M30 ; End program

; Memory etching complete
; Total etch time: Variable based on resonance frequency
"""

class MIMIntegration:
    """Main integration class - drop-in replacement for RENT A HAL routing"""
    
    def __init__(self, openai_api_key: str = None):
        self.intent_ai = IntentChatAI(openai_api_key)
        self.memory_system = CrystallineMemoryInterface()
        self.active_sessions = {}
        self.intent_history = []
        
        # Three Minds processors
        self.current_mind = CurrentMindProcessor()
        self.past_mind = PastMindProcessor(self.memory_system)
        self.comparative_mind = ComparativeMindProcessor()
        
        # Intent weights (Master Intent Matrix state)
        self.intent_weights = {
            'chat': 100000.0,
            'vision': 80000.0,
            'speech': 90000.0,
            'gmail': 70000.0,
            'monitor': 50000.0,
            'system': 60000.0
        }
        
        self.cognitive_temperature = 1.0
        self.reality_membrane_decay = 0.5
        
    async def route_query(self, query_data: Dict[str, Any], websocket: WebSocket, 
                         user_id: str) -> Dict[str, Any]:
        """Main routing method - replaces traditional routing logic"""
        
        # Create query context
        context = QueryContext(
            user_id=user_id,
            session_id=query_data.get('session_id', 'default'),
            timestamp=time.time(),
            query_type=query_data.get('type', 'unknown'),
            urgency=query_data.get('urgent', 0.5),
            complexity=self._estimate_complexity(query_data),
            metadata=query_data.get('metadata', {})
        )
        
        # Decode intent using AI
        user_input = query_data.get('prompt', '') or query_data.get('message', '')
        intent_decision = await self.intent_ai.decode_intent(user_input, context)
        
        logger.info(f"Intent decoded: {intent_decision.intent_name} "
                   f"(confidence: {intent_decision.confidence:.2f})")
        
        # Update intent weights using differential equations
        self._update_intent_weights(intent_decision, context)
        
        # Process through Three Minds
        three_minds_result = await self._process_three_minds(
            intent_decision, query_data, context
        )
        
        # Store memory in crystalline substrate
        await self._store_crystalline_memory(intent_decision, three_minds_result, context)
        
        # Route to appropriate handler
        result = await self._execute_intent_handler(
            intent_decision, query_data, websocket, context, three_minds_result
        )
        
        # Update session state
        self._update_session_state(user_id, intent_decision, result)
        
        return result
    
    def _estimate_complexity(self, query_data: Dict[str, Any]) -> float:
        """Estimate query complexity for intent processing"""
        complexity = 0.1
        
        if 'prompt' in query_data and query_data['prompt']:
            complexity += min(0.8, len(query_data['prompt']) / 1000)
        
        if 'image' in query_data:
            complexity += 0.6
            
        if query_data.get('type') == 'gmail_summary':
            complexity += 0.4
            
        if 'attachments' in query_data:
            complexity += 0.3
            
        return min(1.0, complexity)
    
    def _update_intent_weights(self, intent_decision: IntentDecision, context: QueryContext):
        """Update intent weights using Master Intent Equation"""
        
        dt = 0.01  # Time step
        current_intent = intent_decision.intent_name
        
        for intent_name, weight in self.intent_weights.items():
            # Sensory input (S)
            if intent_name == current_intent:
                sensory_input = intent_decision.confidence
            else:
                sensory_input = 0.1
            
            # Competitive inhibition (C)
            if intent_name == current_intent:
                inhibition = 0.0
            else:
                inhibition = intent_decision.confidence * 0.5
            
            # Distance from ideal state (simplified)
            distance = 1.0 - intent_decision.confidence if intent_name == current_intent else 1.0
            
            # Master Intent Equation: dW/dt = S(1-W/Wmax)e^(-αD) - CW - λW + T√W N(0,1)
            dW = (
                sensory_input * (1000000 - weight) * np.exp(-self.reality_membrane_decay * distance) -
                inhibition * weight -
                0.1 * weight +
                self.cognitive_temperature * np.random.normal(0, 1) * np.sqrt(weight)
            )
            
            # Update weight
            new_weight = max(0, min(1000000, weight + dW * dt))
            self.intent_weights[intent_name] = new_weight
        
        logger.debug(f"Updated intent weights: {self.intent_weights}")
    
    async def _process_three_minds(self, intent_decision: IntentDecision, 
                                  query_data: Dict[str, Any], 
                                  context: QueryContext) -> Dict[str, Any]:
        """Process query through Three Minds architecture"""
        
        # Current Mind - immediate processing
        current_result = await self.current_mind.process(intent_decision, query_data, context)
        
        # Past Mind - memory and experience
        past_result = await self.past_mind.process(intent_decision, query_data, context)
        
        # Comparative Mind - analysis and prediction
        comparative_result = await self.comparative_mind.process(
            intent_decision, query_data, context, current_result, past_result
        )
        
        # Synthesize results
        synthesis = self._synthesize_minds(current_result, past_result, comparative_result)
        
        return {
            'current': current_result,
            'past': past_result,
            'comparative': comparative_result,
            'synthesis': synthesis
        }
    
    def _synthesize_minds(self, current: Dict, past: Dict, comparative: Dict) -> Dict[str, Any]:
        """Synthesize results from all three minds"""
        
        # Weight the minds' contributions
        current_weight = 0.5
        past_weight = 0.3  
        comparative_weight = 0.2
        
        synthesized_confidence = (
            current.get('confidence', 0.5) * current_weight +
            past.get('confidence', 0.5) * past_weight +
            comparative.get('confidence', 0.5) * comparative_weight
        )
        
        return {
            'confidence': synthesized_confidence,
            'reasoning': f"Current: {current.get('reasoning', 'N/A')}, "
                        f"Past: {past.get('reasoning', 'N/A')}, "
                        f"Comparative: {comparative.get('reasoning', 'N/A')}",
            'recommended_action': comparative.get('prediction', 'process_normally')
        }
    
    async def _store_crystalline_memory(self, intent_decision: IntentDecision,
                                       three_minds_result: Dict[str, Any],
                                       context: QueryContext):
        """Store processing result in crystalline memory"""
        
        # Create state vector from processing results
        state_vector = np.array([
            intent_decision.confidence,
            three_minds_result['synthesis']['confidence'],
            context.urgency,
            context.complexity,
            self.intent_weights[intent_decision.intent_name] / 1000000,  # Normalized
            time.time() % 86400,  # Time of day factor
        ])
        
        # Store in crystalline substrate
        crystal_id = await self.memory_system.etch_memory(
            intent_decision.intent_name,
            state_vector,
            {
                'user_id': context.user_id,
                'session_id': context.session_id,
                'query_type': context.query_type,
                'three_minds': three_minds_result
            }
        )
        
        logger.info(f"Stored memory in crystal: {crystal_id}")
    
    async def _execute_intent_handler(self, intent_decision: IntentDecision,
                                     query_data: Dict[str, Any],
                                     websocket: WebSocket,
                                     context: QueryContext,
                                     three_minds_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the appropriate intent handler"""
        
        intent_name = intent_decision.intent_name
        
        # Send intent activation notification
        await websocket.send_text(json.dumps({
            'type': 'intent_activated',
            'intent': intent_name,
            'confidence': intent_decision.confidence,
            'priority': intent_decision.priority,
            'reasoning': intent_decision.reasoning,
            'three_minds': three_minds_result['synthesis']
        }))
        
        # Route to specific handler
        handlers = {
            'chat': self._handle_chat_intent,
            'vision': self._handle_vision_intent,
            'speech': self._handle_speech_intent,
            'gmail': self._handle_gmail_intent,
            'monitor': self._handle_monitor_intent,
            'system': self._handle_system_intent
        }
        
        handler = handlers.get(intent_name, self._handle_default_intent)
        return await handler(intent_decision, query_data, websocket, context, three_minds_result)
    
    async def _handle_chat_intent(self, intent_decision: IntentDecision,
                                 query_data: Dict[str, Any], websocket: WebSocket,
                                 context: QueryContext, three_minds_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chat intent with MIM processing"""
        
        return {
            'type': 'chat',
            'intent': intent_decision.intent_name,
            'priority': intent_decision.priority,
            'confidence': intent_decision.confidence,
            'data': query_data,
            'three_minds_synthesis': three_minds_result['synthesis'],
            'routing_decision': 'standard_chat_processing',
            'next_actions': intent_decision.next_actions
        }
    
    async def _handle_vision_intent(self, intent_decision: IntentDecision,
                                   query_data: Dict[str, Any], websocket: WebSocket,
                                   context: QueryContext, three_minds_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle vision intent with MIM processing"""
        
        return {
            'type': 'vision',
            'intent': intent_decision.intent_name,
            'priority': intent_decision.priority,
            'confidence': intent_decision.confidence,
            'data': query_data,
            'three_minds_synthesis': three_minds_result['synthesis'],
            'routing_decision': 'vision_processing',
            'next_actions': intent_decision.next_actions
        }
    
    async def _handle_speech_intent(self, intent_decision: IntentDecision,
                                   query_data: Dict[str, Any], websocket: WebSocket,
                                   context: QueryContext, three_minds_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle speech intent with MIM processing"""
        
        return {
            'type': 'speech',
            'intent': intent_decision.intent_name,
            'priority': 'high',  # Speech is always high priority
            'confidence': intent_decision.confidence,
            'data': query_data,
            'three_minds_synthesis': three_minds_result['synthesis'],
            'routing_decision': 'speech_processing',
            'next_actions': intent_decision.next_actions
        }
    
    async def _handle_gmail_intent(self, intent_decision: IntentDecision,
                                  query_data: Dict[str, Any], websocket: WebSocket,
                                  context: QueryContext, three_minds_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Gmail intent with MIM processing"""
        
        return {
            'type': 'gmail',
            'intent': intent_decision.intent_name,
            'priority': intent_decision.priority,
            'confidence': intent_decision.confidence,
            'data': query_data,
            'three_minds_synthesis': three_minds_result['synthesis'],
            'routing_decision': 'gmail_processing',
            'next_actions': intent_decision.next_actions
        }
    
    async def _handle_monitor_intent(self, intent_decision: IntentDecision,
                                    query_data: Dict[str, Any], websocket: WebSocket,
                                    context: QueryContext, three_minds_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system monitoring intent with MIM processing"""
        
        return {
            'type': 'monitor',
            'intent': intent_decision.intent_name,
            'priority': 'low',  # Background monitoring
            'confidence': intent_decision.confidence,
            'data': query_data,
            'three_minds_synthesis': three_minds_result['synthesis'],
            'routing_decision': 'system_monitoring',
            'next_actions': intent_decision.next_actions
        }
    
    async def _handle_system_intent(self, intent_decision: IntentDecision,
                                   query_data: Dict[str, Any], websocket: WebSocket,
                                   context: QueryContext, three_minds_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system commands with MIM processing"""
        
        return {
            'type': 'system',
            'intent': intent_decision.intent_name,
            'priority': intent_decision.priority,
            'confidence': intent_decision.confidence,
            'data': query_data,
            'three_minds_synthesis': three_minds_result['synthesis'],
            'routing_decision': 'system_command',
            'next_actions': intent_decision.next_actions
        }
    
    async def _handle_default_intent(self, intent_decision: IntentDecision,
                                    query_data: Dict[str, Any], websocket: WebSocket,
                                    context: QueryContext, three_minds_result: Dict[str, Any]) -> Dict[str, Any]:
        """Default handler for unknown intents"""
        
        return {
            'type': 'unknown',
            'intent': intent_decision.intent_name,
            'priority': 'normal',
            'confidence': intent_decision.confidence,
            'data': query_data,
            'three_minds_synthesis': three_minds_result['synthesis'],
            'routing_decision': 'default_processing',
            'next_actions': ['analyze_further', 'request_clarification']
        }
    
    def _update_session_state(self, user_id: str, intent_decision: IntentDecision, result: Dict[str, Any]):
        """Update session state with intent processing results"""
        
        if user_id not in self.active_sessions:
            self.active_sessions[user_id] = {
                'created': time.time(),
                'intent_history': [],
                'last_intent': None,
                'session_weight': 1.0
            }
        
        session = self.active_sessions[user_id]
        session['intent_history'].append({
            'intent': intent_decision.intent_name,
            'confidence': intent_decision.confidence,
            'timestamp': time.time(),
            'result': result
        })
        
        session['last_intent'] = intent_decision.intent_name
        session['last_update'] = time.time()
        
        # Keep only last 10 intents
        if len(session['intent_history']) > 10:
            session['intent_history'] = session['intent_history'][-10:]
    
    def get_mim_status(self) -> Dict[str, Any]:
        """Get Master Intent Matrix system status"""
        
        return {
            'intent_weights': self.intent_weights,
            'cognitive_temperature': self.cognitive_temperature,
            'reality_membrane_decay': self.reality_membrane_decay,
            'active_sessions': len(self.active_sessions),
            'total_intents_processed': len(self.intent_history),
            'memory_crystals': self.memory_system.get_crystal_count(),
            'timestamp': time.time()
        }
    
    async def generate_memory_gcode(self, intent_name: str) -> Optional[str]:
        """Generate G-code for crystalline memory etching"""
        
        # Create a query vector for the intent
        query_vector = np.array([
            self.intent_weights.get(intent_name, 50000) / 1000000,
            time.time() % 86400,
            0.5,  # Default complexity
            0.5,  # Default urgency
            1.0,  # Intent specific
            self.cognitive_temperature
        ])
        
        # Find relevant memories
        memories = await self.memory_system.recall_memory(query_vector, 0.7)
        
        if not memories:
            return None
        
        # Generate G-code for the best matching memory
        best_memory = memories[0]
        gcode_gen = GCodeGenerator()
        return await gcode_gen.generate_etching_gcode(
            best_memory['coordinates'],
            best_memory['resonance']
        )

# Three Minds Processor Classes

class CurrentMindProcessor:
    """Processes immediate, real-time sensory input"""
    
    async def process(self, intent_decision: IntentDecision, query_data: Dict[str, Any], 
                     context: QueryContext) -> Dict[str, Any]:
        """Process current immediate state"""
        
        # Immediate response based on current input
        current_strength = intent_decision.confidence * (1.0 + context.urgency)
        
        return {
            'type': 'current',
            'strength': current_strength,
            'confidence': intent_decision.confidence,
            'urgency_factor': context.urgency,
            'reasoning': f"Immediate response to {intent_decision.intent_name}",
            'timestamp': time.time()
        }

class PastMindProcessor:
    """Processes memory and past experience"""
    
    def __init__(self, memory_system: CrystallineMemoryInterface):
        self.memory_system = memory_system
    
    async def process(self, intent_decision: IntentDecision, query_data: Dict[str, Any],
                     context: QueryContext) -> Dict[str, Any]:
        """Process based on past memories and experience"""
        
        # Create query vector for memory recall
        query_vector = np.array([
            intent_decision.confidence,
            context.urgency,
            context.complexity,
            time.time() % 86400,  # Time of day
            hash(intent_decision.intent_name) % 1000 / 1000,  # Intent hash
            context.user_id.__hash__() % 1000 / 1000  # User hash
        ])
        
        # Recall similar memories
        memories = await self.memory_system.recall_memory(query_vector, 0.6)
        
        if not memories:
            return {
                'type': 'past',
                'strength': 0.3,
                'confidence': 0.3,
                'memory_count': 0,
                'reasoning': 'No relevant past memories found'
            }
        
        # Calculate influence from past memories
        total_influence = 0
        for memory in memories[:3]:  # Top 3 memories
            age_factor = max(0.1, 1.0 - (time.time() - memory['timestamp']) / 86400)  # 24h decay
            total_influence += memory['similarity'] * age_factor
        
        avg_influence = total_influence / len(memories[:3])
        
        return {
            'type': 'past',
            'strength': avg_influence,
            'confidence': min(0.9, avg_influence),
            'memory_count': len(memories),
            'reasoning': f'Based on {len(memories)} similar past experiences',
            'top_memory': memories[0] if memories else None
        }

class ComparativeMindProcessor:
    """Processes analysis, comparison, and prediction"""
    
    async def process(self, intent_decision: IntentDecision, query_data: Dict[str, Any],
                     context: QueryContext, current_result: Dict[str, Any], 
                     past_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process comparative analysis and prediction"""
        
        current_strength = current_result.get('strength', 0.5)
        past_strength = past_result.get('strength', 0.5)
        
        # Analyze trend
        trend = current_strength - past_strength
        
        # Predict future value
        predicted_value = current_strength + (trend * 0.5)
        
        # Calculate confidence based on consistency
        consistency = 1.0 - abs(trend)
        confidence = min(0.9, consistency * intent_decision.confidence)
        
        # Determine recommended action
        if predicted_value > 0.8:
            action = 'high_priority_processing'
        elif predicted_value > 0.5:
            action = 'standard_processing'
        elif predicted_value > 0.2:
            action = 'low_priority_processing'
        else:
            action = 'defer_processing'
        
        return {
            'type': 'comparative',
            'strength': predicted_value,
            'confidence': confidence,
            'trend': trend,
            'prediction': action,
            'reasoning': f'Trend analysis: {trend:.2f}, Predicted: {predicted_value:.2f}',
            'consistency': consistency
        }

# Integration wrapper for existing RENT A HAL code
class RAHMIMWrapper:
    """Wrapper to integrate MIM with existing RENT A HAL codebase"""
    
    def __init__(self, openai_api_key: str = None):
        self.mim = MIMIntegration(openai_api_key)
        
    async def handle_websocket_message(self, websocket: WebSocket, message: Dict[str, Any], 
                                     user_id: str) -> Dict[str, Any]:
        """Handle WebSocket message through MIM routing"""
        
        try:
            # Route through Master Intent Matrix
            result = await self.mim.route_query(message, websocket, user_id)
            
            # Return routing decision for existing handlers
            return {
                'intent_routing': result,
                'should_process': True,
                'priority': result.get('priority', 'normal'),
                'handler_type': result.get('type', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"MIM routing error: {e}")
            # Fallback to traditional routing
            return {
                'intent_routing': None,
                'should_process': True,
                'priority': 'normal', 
                'handler_type': 'fallback',
                'error': str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get MIM system status"""
        return self.mim.get_mim_status()
    
    async def get_memory_gcode(self, intent_name: str) -> Optional[str]:
        """Get G-code for memory etching"""
        return await self.mim.generate_memory_gcode(intent_name)

# FastAPI integration example
def create_mim_routes(app: FastAPI, mim_wrapper: RAHMIMWrapper):
    """Add MIM routes to FastAPI app"""
    
    @app.get("/mim/status")
    async def get_mim_status():
        """Get Master Intent Matrix status"""
        return mim_wrapper.get_status()
    
    @app.get("/mim/memory/gcode/{intent_name}")
    async def get_memory_gcode(intent_name: str):
        """Get G-code for crystalline memory etching"""
        gcode = await mim_wrapper.get_memory_gcode(intent_name)
        if gcode:
            return PlainTextResponse(gcode, media_type="text/plain")
        else:
            raise HTTPException(status_code=404, detail="No memory found for intent")
    
    @app.post("/mim/query")
    async def process_mim_query(query_data: Dict[str, Any]):
        """Process query through MIM (for testing)"""
        # This would normally be called from WebSocket handler
        result = await mim_wrapper.mim.route_query(query_data, None, "test_user")
        return result

# Usage Example:
"""
# In your main webgui.py or app.py:

from MIM_Integration import RAHMIMWrapper, create_mim_routes

# Initialize MIM
mim_wrapper = RAHMIMWrapper(openai_api_key="your-key-here")

# Add MIM routes
create_mim_routes(app, mim_wrapper)

# In WebSocket handler:
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Route through MIM
            mim_result = await mim_wrapper.handle_websocket_message(
                websocket, data, user_id
            )
            
            # Process based on MIM decision
            if mim_result['should_process']:
                handler_type = mim_result['handler_type']
                priority = mim_result['priority']
                
                # Route to existing handlers based on MIM decision
                if handler_type == 'chat':
                    await handle_chat_query(data, websocket, priority)
                elif handler_type == 'vision':
                    await handle_vision_query(data, websocket, priority)
                # ... etc
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()
"""