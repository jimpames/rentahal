# TERMS AND CONDITIONS

# ## 🔐 Supplemental License Terms (RENT A HAL Specific)

# In addition to the terms of the GNU General Public License v3.0 (GPL-3.0), the following conditions **explicitly apply** to this project and all derivative works:

# - 🚫 **No Closed Source Derivatives**: Any derivative, fork, or modified version of RENT A HAL must **remain fully open source** under a GPL-compatible license.
  
# - 🧬 **No Patents**: You **may not patent** RENT A HAL or any part of its original or derived code, design, architecture, or functional implementations.

# - 🔁 **License Must Propagate**: Any distribution of modified versions must include this exact clause, in addition to the GPL-3.0, to ensure **eternal openness**.

# - ⚖️ **Enforcement**: Violation of these conditions terminates your rights under this license and may be pursued legally.

# This clause is intended to **protect the freedom and integrity** of this AI system for all present and future generations. If you use it — respect it.

# > "This project is free forever. If you change it — it stays free too."

# this notice must remain in all copies / derivatives of the work forever and must not be removed.

import time
import asyncio
import aiofiles
from asyncio import TimeoutError as AsyncTimeoutError
import json
import uuid
import logging
import configparser
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Callable, Union
import sqlite3
from datetime import datetime, timedelta
import aiohttp
from aiohttp import ClientConnectorError, ClientResponseError
from contextlib import asynccontextmanager
import os
from huggingface_hub import InferenceClient
import functools
import redis
import base64
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor
import librosa
import subprocess
import random 
import torch
import whisper
import shelve
from bark import generate_audio, SAMPLE_RATE, preload_models

from scipy.io.wavfile import write as write_wav
import numpy as np

import pyttsx3

from scipy.io import wavfile

from fastapi import APIRouter
from logging.handlers import RotatingFileHandler

class QueueProcessorStatus:
    def __init__(self):
        self.last_heartbeat = time.time()
        self.is_running = False

queue_processor_status = QueueProcessorStatus()






# Pydantic models
class User(BaseModel):
    guid: str
    nickname: str
    is_sysop: bool = False
    total_query_time: float = 0.0
    total_cost: float = 0.0
    is_banned: bool = False
    query_count: int = 0


class Query(BaseModel):
    prompt: str
    query_type: str
    model_type: str
    model_name: str
    image: Optional[str] = None
    audio: Optional[str] = None  # New field for audio data

class AIWorker(BaseModel):
    name: str
    address: str
    type: str
    health_score: float = 100.0
    is_blacklisted: bool = False
    last_active: str = datetime.now().isoformat()

class HuggingFaceModel(BaseModel):
    name: str
    type: str


# Create an API router
api_router = APIRouter()


tts_engine = pyttsx3.init()



# from logging.handlers import RotatingFileHandler

log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = 'webgui_detailed.log'
log_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(log_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

logger.info("Starting webgui.py")



# Debug decorator
def debug(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f"Entering {func_name}")
        logger.debug(f"Args: {args}")
        logger.debug(f"Kwargs: {kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func_name}")
            logger.debug(f"Result: {result}")
            return result
        except Exception as e:
            logger.exception(f"Exception in {func_name}: {str(e)}")
            raise
    
    return wrapper




















# Add this after the initial imports and logging setup
logger.info("Preloading BARK model...")
preload_models(text_use_small=True, text_use_gpu=True, coarse_use_small=True, coarse_use_gpu=True, fine_use_gpu=True, fine_use_small=True)
logger.info("BARK model preloaded successfully")

# Global setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
whisper_model = whisper.load_model("base").to(device)


total_costs_lifetime = 0.0
system_stats = {
    "total_queries": 0,
    "chat_time": [],
    "vision_time": [],
    "imagine_time": [],
    "speech_in_time": [],
    "speech_out_time": [],
    "max_connected_users": 0
}

@debug
def insert_query(user: User, query: Query, processing_time: float, cost: float):
    logger.debug(f"Inserting query for user {user.guid}")
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
    INSERT INTO queries (user_guid, query_type, model_type, model_name, prompt, processing_time, cost)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user.guid, query.query_type, query.model_type, query.model_name, query.prompt, processing_time, cost))
    db.commit()
    db.close()
    logger.info(f"Query inserted for user {user.guid}")


def load_persistent_stats():
    global total_costs_lifetime, system_stats
    with shelve.open('persistent_stats') as db:
        total_costs_lifetime = db.get('total_costs_lifetime', 0.0)
        system_stats = db.get('system_stats', system_stats)

def save_persistent_stats():
    with shelve.open('persistent_stats') as db:
        db['total_costs_lifetime'] = total_costs_lifetime
        db['system_stats'] = system_stats

def get_avg_time(time_list):
    return sum(time_list) / len(time_list) if time_list else 0


def reset_stats_if_zero():
    global system_stats
    if all(not times for times in system_stats.values() if isinstance(times, list)):
        logger.info("Resetting system stats as all values are zero")
        system_stats = {
            "total_queries": 0,
            "chat_time": [],
            "vision_time": [],
            "imagine_time": [],
            "speech_in_time": [],
            "speech_out_time": [],
            "max_connected_users": system_stats["max_connected_users"]
        }
        save_persistent_stats()

# Call this function at the start of your application
load_persistent_stats()





# Log GPU information
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"Device being used: {device}")
if torch.cuda.is_available():
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"Current GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Current GPU memory allocated: {torch.cuda.memory_allocated(0)}")
    logger.info(f"Current GPU memory reserved: {torch.cuda.memory_reserved(0)}")

# Log initial system information
logger.info(f"Operating System: {os.name}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"Device being used: {device}")
if torch.cuda.is_available():
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"Current GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0)}")
    logger.info(f"Initial GPU memory reserved: {torch.cuda.memory_reserved(0)}")

# Redis setup
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    redis_client.ping()
    logger.info("Redis connection established")
except redis.ConnectionError:
    logger.error("Failed to connect to Redis. Ensure Redis server is running.")
    redis_client = None


# Load and validate configuration
@debug
def load_config():
    logger.info("Loading configuration")
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Default configuration
    default_config = {
        'Settings': {
            'debug': 'True',
            'idle_watt_rate': '500',
            'premium_watt_rate': '1000',
            'electricity_cost_per_kwh': '0.25'
        },
        'Database': {
            'database_name': 'llm_broker.db'
        },
        'Server': {
            'host': '0.0.0.0',
            'port': '5000',
            'debug_port': '5001'
        },
        'Websocket': {
            'max_message_size': '1048576'
        },
        'Workers': {
            'default_worker_address': 'localhost:8000',
            'health_check_interval': '60',
            'NO_BLACKLIST_IMAGINE': '1'
        },
        'HuggingFace': {
            'default_models': 'gpt2,gpt2-medium,gpt2-large',
            'api_key': 'YOUR_HUGGINGFACE_API_KEY'
        },
        'Claude': {
            'api_key': 'YOUR_CLAUDE_API_KEY_HERE',
            'endpoint': 'https://api.anthropic.com/v1/messages',
            'model_name': 'claude-2.1'
        },
        'Security': {
            'secret_key': 'your_secret_key_here',
            'token_expiration': '3600'
        },
        'Performance': {
            'max_connections': '100',
            'query_timeout': '30'
        },
        'Costs': {
            'base_cost_per_query': '0.01',
            'cost_per_second': '0.001'
        },
        'Queue': {
            'max_queue_size': '100',
            'queue_timeout': '300'
        },
        'Chunking': {
            'chunk_size': '1048576'  # 1MB default chunk size
        }
    }

    # Update config with default values for missing keys
    for section, options in default_config.items():
        if section not in config:
            config[section] = {}
        for option, value in options.items():
            if option not in config[section]:
                config[section][option] = value

    # Write updated config back to file
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    logger.info("Configuration loaded and validated successfully")
    return config

config = load_config()

# Settings
DEBUG = config.getboolean('Settings', 'debug')
IDLE_WATT_RATE = config.getfloat('Settings', 'idle_watt_rate')
PREMIUM_WATT_RATE = config.getfloat('Settings', 'premium_watt_rate')
ELECTRICITY_COST_PER_KWH = config.getfloat('Settings', 'electricity_cost_per_kwh')
DATABASE_NAME = config.get('Database', 'database_name')
HOST = config.get('Server', 'host')
PORT = config.getint('Server', 'port')
DEBUG_PORT = config.getint('Server', 'debug_port')
MAX_MESSAGE_SIZE = config.getint('Websocket', 'max_message_size')
DEFAULT_WORKER_ADDRESS = config.get('Workers', 'default_worker_address')
HEALTH_CHECK_INTERVAL = config.getint('Workers', 'health_check_interval')
NO_BLACKLIST_IMAGINE = config.getboolean('Workers', 'NO_BLACKLIST_IMAGINE')
DEFAULT_HUGGINGFACE_MODELS = config.get('HuggingFace', 'default_models').split(',')
HUGGINGFACE_API_KEY = config.get('HuggingFace', 'api_key')
CLAUDE_API_KEY = config.get('Claude', 'api_key')
CLAUDE_ENDPOINT = config.get('Claude', 'endpoint')
CLAUDE_MODEL = config.get('Claude', 'model_name')
SECRET_KEY = config.get('Security', 'secret_key')
TOKEN_EXPIRATION = config.getint('Security', 'token_expiration')
MAX_CONNECTIONS = config.getint('Performance', 'max_connections')
QUERY_TIMEOUT = config.getint('Performance', 'query_timeout')
BASE_COST_PER_QUERY = config.getfloat('Costs', 'base_cost_per_query')
COST_PER_SECOND = config.getfloat('Costs', 'cost_per_second')
MAX_QUEUE_SIZE = config.getint('Queue', 'max_queue_size')
QUEUE_TIMEOUT = config.getint('Queue', 'queue_timeout')
CHUNK_SIZE = config.getint('Chunking', 'chunk_size')

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Base directory: {BASE_DIR}")


class CancellableQuery:
    def __init__(self, query_data: Dict[str, Any]):
        self.query_data = query_data
        self.task: Optional[asyncio.Task] = None
        self.cancelled = False

    async def run(self):
        self.task = asyncio.create_task(self._process())
        try:
            return await self.task
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def _process(self):
        result = await process_query(self.query_data['query'])
        return result

    async def cancel(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

class SafeQueue:
    def __init__(self):
        self._queue = asyncio.Queue()
        self._processing: Dict[str, CancellableQuery] = {}
        self._lock = asyncio.Lock()

    async def put(self, item: Dict[str, Any]):
        async with self._lock:
            await self._queue.put(item)

    async def get(self) -> CancellableQuery:
        async with self._lock:
            item = await self._queue.get()
            cancellable_query = CancellableQuery(item)
            self._processing[item['user'].guid] = cancellable_query
            return cancellable_query

    async def remove_by_guid(self, guid: str):
        async with self._lock:
            new_queue = asyncio.Queue()
            while not self._queue.empty():
                item = await self._queue.get()
                if item['user'].guid != guid:
                    await new_queue.put(item)
            self._queue = new_queue
            if guid in self._processing:
                await self._processing[guid].cancel()
                del self._processing[guid]

    def qsize(self) -> int:
        return self._queue.qsize() + len(self._processing)

    async def clear_processing(self, guid: str):
        async with self._lock:
            if guid in self._processing:
                del self._processing[guid]

# State management
class State:
    def __init__(self):
        self.query_queue: SafeQueue = SafeQueue()
        self.total_workers: int = 0

state = State()
logger.info("State initialized")

# Global variables
ai_workers: Dict[str, AIWorker] = {}
huggingface_models: Dict[str, HuggingFaceModel] = {}

# Create a thread pool for CPU-bound tasks
thread_pool = ThreadPoolExecutor(max_workers=4)

# Helper functions
@debug
def get_db():
    logger.debug("Getting database connection")
    db = sqlite3.connect(DATABASE_NAME)
    db.row_factory = sqlite3.Row
    return db

@debug
def init_db():
    logger.info("Initializing database...")
    db = get_db()
    cursor = db.cursor()
    
    # Create tables
    tables = [
        ("users", """
        CREATE TABLE IF NOT EXISTS users (
            guid TEXT PRIMARY KEY,
            nickname TEXT UNIQUE,
            is_sysop BOOLEAN,
            total_query_time REAL DEFAULT 0,
            total_cost REAL DEFAULT 0,
            is_banned BOOLEAN DEFAULT 0,
            query_count INTEGER DEFAULT 0
        )
        """),
  
        ("queries", """
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY,
            user_guid TEXT,
            query_type TEXT,
            model_type TEXT,
            model_name TEXT,
            prompt TEXT,
            processing_time REAL,
            cost REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_guid) REFERENCES users (guid)
        )
        """),
        ("ai_workers", """
        CREATE TABLE IF NOT EXISTS ai_workers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            address TEXT,
            type TEXT,
            health_score REAL,
            is_blacklisted BOOLEAN,
            last_active DATETIME
        )
        """),
        ("huggingface_models", """
        CREATE TABLE IF NOT EXISTS huggingface_models (
            id INTEGER PRIMARY KEY,
            name TEXT,
            type TEXT
        )
        """),
        ("system_stats", """
        CREATE TABLE IF NOT EXISTS system_stats (
            id INTEGER PRIMARY KEY,
            total_queries INTEGER DEFAULT 0,
            total_processing_time REAL DEFAULT 0,
            total_cost REAL DEFAULT 0,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
    ]
    
    for table_name, create_table_sql in tables:
        logger.debug(f"Creating table: {table_name}")
        cursor.execute(create_table_sql)
    
    db.commit()
    db.close()
    logger.info("Database initialized successfully")

@debug
def load_ai_workers():
    logger.info("Loading AI workers")
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM ai_workers")
    workers = cursor.fetchall()
    if not workers:
        logger.warning("No AI workers found in database. Adding default workers.")
        default_workers = [
            ('default_worker', DEFAULT_WORKER_ADDRESS, 'chat', 100.0, False, datetime.now().isoformat()),
            ('claude', CLAUDE_ENDPOINT, 'chat', 100.0, False, datetime.now().isoformat())
        ]
        cursor.executemany("""
        INSERT INTO ai_workers (name, address, type, health_score, is_blacklisted, last_active)
        VALUES (?, ?, ?, ?, ?, ?)
        """, default_workers)
        db.commit()
        workers = [dict(zip(['name', 'address', 'type', 'health_score', 'is_blacklisted', 'last_active'], w)) for w in default_workers]
    for worker in workers:
        ai_workers[worker['name']] = AIWorker(**dict(worker))
    db.close()
    state.total_workers = len(ai_workers)
    logger.info(f"Loaded {len(ai_workers)} AI workers")



@debug
def ensure_query_count_column():
    logger.info("Ensuring query_count column exists in users table")
    db = get_db()
    cursor = db.cursor()
    try:
        # Check if the column exists
        cursor.execute("SELECT query_count FROM users LIMIT 1")
    except sqlite3.OperationalError:
        # Column doesn't exist, so add it
        logger.info("Adding query_count column to users table")
        cursor.execute("ALTER TABLE users ADD COLUMN query_count INTEGER DEFAULT 0")
        db.commit()
    finally:
        db.close()


@debug
def load_huggingface_models():
    logger.info("Loading Hugging Face models")
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM huggingface_models")
    models = cursor.fetchall()
    if not models:
        logger.warning("No Hugging Face models found in database. Adding default models.")
        default_models = [(model, 'chat') for model in DEFAULT_HUGGINGFACE_MODELS]
        cursor.executemany("INSERT INTO huggingface_models (name, type) VALUES (?, ?)", default_models)
        db.commit()
        models = [{'name': name, 'type': type} for name, type in default_models]
    for model in models:
        huggingface_models[model['name']] = HuggingFaceModel(**dict(model))
    db.close()
    logger.info(f"Loaded {len(huggingface_models)} Hugging Face models")

@debug
def get_or_create_user(db: sqlite3.Connection, guid: str) -> User:
    logger.debug(f"Getting or creating user with GUID: {guid}")
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE guid = ?", (guid,))
    user = cursor.fetchone()
    if user is None:
        logger.info(f"Creating new user with GUID: {guid}")
        cursor.execute("SELECT COUNT(*) FROM users")
        is_sysop = cursor.fetchone()[0] == 0  # First user becomes sysop
        nickname = f"user_{guid[:8]}"
        cursor.execute("INSERT INTO users (guid, nickname, is_sysop, total_query_time, total_cost, is_banned, query_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (guid, nickname, is_sysop, 0.0, 0.0, False, 0))
        db.commit()
        return User(guid=guid, nickname=nickname, is_sysop=is_sysop, total_query_time=0.0, total_cost=0.0, is_banned=False)
    return User(**dict(user))





def write_wav(file_path, sample_rate, audio_data):
    wavfile.write(file_path, sample_rate, audio_data)

@debug
def update_system_stats(db: sqlite3.Connection, processing_time: float, cost: float):
    logger.debug(f"Updating system stats: processing_time={processing_time}, cost={cost}")
    cursor = db.cursor()
    cursor.execute("""
    INSERT INTO system_stats (total_queries, total_processing_time, total_cost, last_updated)
    VALUES (1, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
    total_queries = total_queries + 1,
    total_processing_time = total_processing_time + ?,
    total_cost = total_cost + ?,
    last_updated = ?
    """, (processing_time, cost, datetime.now().isoformat(), processing_time, cost, datetime.now().isoformat()))
    db.commit()
    logger.info("System stats updated successfully")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def send_request_to_worker(session, url, payload, timeout):
    try:
        async with session.post(url, json=payload, timeout=timeout) as response:
            if response.status != 200:
                logger.error(f"Worker returned non-200 status: {response.status}")
                logger.error(f"Response text: {await response.text()}")
                raise HTTPException(status_code=response.status, detail=await response.text())
            return await response.json()
    except asyncio.TimeoutError:
        logger.error(f"Request to worker timed out: {url}")
        raise
    except aiohttp.ClientError as e:
        logger.error(f"Client error when contacting worker: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in send_request_to_worker: {str(e)}")
        raise

# Vision processing classes and functions
class VisionChunker:
    def __init__(self):
        self.chunks: Dict[str, List[str]] = {}

    async def receive_chunk(self, data: Dict[str, Any]) -> Optional[str]:
        chunk_id = data['chunk_id']
        total_chunks = data['total_chunks']
        chunk_data = data['chunk_data']
        image_id = data['image_id']

        if image_id not in self.chunks:
            self.chunks[image_id] = [''] * total_chunks

        self.chunks[image_id][chunk_id] = chunk_data

        if all(chunk != '' for chunk in self.chunks[image_id]):
            complete_image = ''.join(self.chunks[image_id])
            del self.chunks[image_id]
            return complete_image
        
        return None

vision_chunker = VisionChunker()

async def process_image(image_data: str) -> str:
    def _process_image():
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        
            if image.mode == 'RGBA':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])
                image = rgb_image
         
            image = image.convert('RGB')
        
            max_size = (512, 512)
            image.thumbnail(max_size, Image.LANCZOS)
        
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85, optimize=True)
            processed_image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
            return processed_image_data
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    return await asyncio.get_event_loop().run_in_executor(thread_pool, _process_image)



def get_avg_time(time_list):
    return sum(time_list) / len(time_list) if time_list else 0



async def run_ffmpeg_async(command):
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise Exception(f"FFmpeg error: {stderr.decode()}")
    return stdout, stderr






@debug
async def process_speech_to_text(audio_data: str) -> str:
    logger.info("Processing speech to text")
    start_time = time.time()
    try:
        audio_bytes = base64.b64decode(audio_data)
        input_audio_path = f'input_{time.time()}.webm'
        with open(input_audio_path, 'wb') as f:
            f.write(audio_bytes)
        
        # Convert WebM to WAV (Whisper requires WAV format)
        wav_audio_path = input_audio_path.replace('.webm', '.wav')
        os.system(f"ffmpeg -i {input_audio_path} -ar 16000 -ac 1 -c:a pcm_s16le {wav_audio_path} -y")
        
        # Transcribe audio using Whisper
        audio = whisper.load_audio(wav_audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(device)
        
        # Ensure model is on the correct device
        whisper_model.to(device)
        
        _, probs = whisper_model.detect_language(mel)
        options = whisper.DecodingOptions(fp16=torch.cuda.is_available())
        result = whisper.decode(whisper_model, mel, options)
        transcription = result.text
        
        # Clean up temporary files
        os.remove(input_audio_path)
        os.remove(wav_audio_path)
        
        end_time = time.time()
        processing_time = end_time - start_time
        system_stats["speech_in_time"].append(processing_time)
        save_persistent_stats()
        
        logger.info(f"Speech to text processing completed in {processing_time:.2f} seconds")
        logger.info(f"Transcription: {transcription}")
        logger.info(f"Whisper model device: {next(whisper_model.parameters()).device}")
        return transcription
    except Exception as e:
        logger.error(f"Error in speech to text processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in speech to text processing: {str(e)}")


MAX_BARK_WORDS = 20

def pyttsx3_to_audio(text):
    output_file = f'output_{time.time()}.wav'
    tts_engine.save_to_file(text, output_file)
    tts_engine.runAndWait()
    with open(output_file, 'rb') as f:
        audio_data = f.read()
    os.remove(output_file)
    return base64.b64encode(audio_data).decode('utf-8')






async def process_text_to_speech(text: str) -> str:
    word_count = len(text.split())
    logger.info(f"Processing text to speech. Word count: {word_count}")
    
    start_time = time.time()
    try:
        if word_count <= MAX_BARK_WORDS:
            logger.info("Using BARK for text-to-speech")
            audio_array = generate_audio(
                text, text_temp=0.7, waveform_temp=0.7, history_prompt="v2/en_speaker_6"
            )
            trimmed_audio, _ = librosa.effects.trim(audio_array, top_db=20)
            audio_array_int16 = (trimmed_audio * 32767).astype(np.int16)
            output_wav_path = f'output_{time.time()}.wav'
            wavfile.write(output_wav_path, SAMPLE_RATE, audio_array_int16)
            with open(output_wav_path, 'rb') as f:
                output_audio_data = f.read()
            os.remove(output_wav_path)
            output_audio_base64 = base64.b64encode(output_audio_data).decode('utf-8')
        else:
            logger.info("Query return too big for BARK - using pyttsx3 instead")
            prefix = "Query return too big to BARK - speech synth out instead. "
            full_text = prefix + text
            output_audio_base64 = await asyncio.to_thread(pyttsx3_to_audio, full_text)

        end_time = time.time()
        processing_time = end_time - start_time
        system_stats["speech_out_time"].append(processing_time)
        save_persistent_stats()
        
        logger.info(f"Text to speech processing completed in {processing_time:.2f} seconds")
        return output_audio_base64
    except Exception as e:
        logger.error(f"Error in text to speech processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in text to speech processing: {str(e)}")





async def log_gpu_memory_usage():
    while True:
        if torch.cuda.is_available():
            logger.info(f"Current GPU memory allocated: {torch.cuda.memory_allocated(0)}")
            logger.info(f"Current GPU memory reserved: {torch.cuda.memory_reserved(0)}")
        await asyncio.sleep(60)  # Log every 60 seconds





@debug
async def process_query(query: Query) -> Union[str, bytes]:
    logger.info(f"Processing query: {query.query_type} - {query.model_type}")
    try:
        if query.query_type == 'speech':
            transcription = await process_speech_to_text(query.audio)
            query.prompt = transcription
            query.query_type = 'chat'

        result = await process_query_based_on_type(query)

        if query.model_type == 'speech' and query.query_type != 'imagine':
            audio_result = await process_text_to_speech(result)
            return audio_result
        elif query.query_type == 'imagine':
            # For imagine queries, always return the image result without text-to-speech
            return result
        else:
            return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@debug
async def process_query_based_on_type(query: Query) -> str:
    if query.model_type == "huggingface":
        return await process_query_huggingface(query)
    elif query.model_type == "claude":
        return await process_query_claude(query)
    else:
        return await process_query_worker_node(query)

@debug
async def process_query_worker_node(query: Query) -> Union[str, bytes]:
    logger.info(f"Processing query with worker node: {query.model_name}")
    worker = select_worker(query.query_type)
    if not worker:
        logger.error("No available worker nodes")
        raise HTTPException(status_code=503, detail="No available worker nodes")
    
    logger.debug(f"Selected worker: {worker.name}")
    async with aiohttp.ClientSession() as session:
        data = {
            "prompt": query.prompt,
            "type": query.query_type,
            "model_type": query.model_type,
            "model_name": query.model_name
        }

        if query.image:
            data["image"] = query.image
        
        try:
            if worker.type == 'imagine':
                # Stable Diffusion specific endpoint and payload
                worker_url = f"http://{worker.address}/sdapi/v1/txt2img"
                payload = {
                    "prompt": query.prompt,
                    "negative_prompt": "",
                    "steps": 50,
                    "sampler_name": "Euler a",
                    "cfg_scale": 7,
                    "width": 512,
                    "height": 512,
                    "seed": -1,
                }
            else:
                worker_url = f"http://{worker.address}/predict"
                payload = data

            logger.debug(f"Sending request to worker: {worker_url}")
            result = await send_request_to_worker(session, worker_url, payload, QUERY_TIMEOUT)
            logger.info("Query processed successfully by worker node")
            
            if worker.type == 'imagine':
                image_data = base64.b64decode(result["images"][0])
                return image_data
            return result["response"]
        except Exception as e:
            logger.error(f"Error processing query after retries: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing query after retries: {str(e)}")

@debug
async def process_query_huggingface(query: Query) -> str:
    logger.info(f"Processing query with Hugging Face model: {query.model_name}")
    model_name = query.model_name if query.model_name in huggingface_models else list(huggingface_models.keys())[0]
    if model_name not in huggingface_models:
        logger.error(f"Unknown Hugging Face model: {model_name}")
        raise HTTPException(status_code=400, detail=f"Unknown Hugging Face model: {model_name}")
    
    logger.debug(f"Using Hugging Face model: {model_name}")
    try:
        client = InferenceClient(model=model_name, token=HUGGINGFACE_API_KEY)
        response = await asyncio.to_thread(client.text_generation, query.prompt, max_new_tokens=50)
        logger.info("Query processed successfully by Hugging Face model")
        if isinstance(response, str):
            return response
        elif isinstance(response, list) and len(response) > 0:
            return response[0].get('generated_text', str(response[0]))
        else:
            return str(response)
    except Exception as e:
        logger.error(f"Error processing Hugging Face query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing Hugging Face query: {str(e)}")

@debug
async def process_query_claude(query: Query) -> str:
    logger.info("Processing query with Claude")
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01"
            }
            data = {
                "model": CLAUDE_MODEL,
                "messages": [
                    {"role": "user", "content": query.prompt}
                ],
                "max_tokens": 300
            }
            async with session.post(CLAUDE_ENDPOINT, json=data, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Claude API error: Status {response.status}, Response: {error_text}")
                    raise HTTPException(status_code=response.status, detail=f"Error from Claude API: {error_text}")
                result = await response.json()
                if 'content' in result:
                    return result['content'][0]['text']
                else:
                    logger.error(f"Unexpected Claude API response structure: {result}")
                    raise HTTPException(status_code=500, detail="Unexpected response structure from Claude API")
    except aiohttp.ClientError as e:
        logger.error(f"Error communicating with Claude API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error communicating with Claude API: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error processing Claude query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error processing Claude query: {str(e)}")

@debug
def select_worker(query_type: str) -> Optional[AIWorker]:
    logger.debug(f"Selecting worker for query type: {query_type}")
    available_workers = [w for w in ai_workers.values() if w.type == query_type and not w.is_blacklisted and w.name != "claude"]
    if not available_workers:
        logger.warning(f"No available workers for query type: {query_type}")
        return None
    selected_worker = max(available_workers, key=lambda w: w.health_score)
    logger.info(f"Selected worker: {selected_worker.name}")
    return selected_worker

@debug
async def update_worker_health():
    logger.info("Starting worker health update loop")
    while True:
        for worker in ai_workers.values():
            try:
                if worker.name == "claude":
                    logger.debug("Skipping health check for Claude API")
                    worker.health_score = 100
                    worker.is_blacklisted = False
                elif worker.is_blacklisted:
                    # Attempt to recover blacklisted workers
                    async with aiohttp.ClientSession() as session:
                        worker_url = f"http://{worker.address}/health"
                        async with session.get(worker_url, timeout=10 if worker.type == 'imagine' else 5) as response:
                            if response.status == 200:
                                worker.health_score = 50  # Restore to 50% health
                                worker.is_blacklisted = False
                                logger.info(f"Worker {worker.name} recovered from blacklist")
                else:
                    logger.debug(f"Checking health for worker: {worker.name}")
                    async with aiohttp.ClientSession() as session:
                        worker_url = f"http://{worker.address}/health"
                        async with session.get(worker_url, timeout=10 if worker.type == 'imagine' else 5) as response:
                            if response.status == 200:
                                worker.health_score = min(100, worker.health_score + 10)
                                worker.is_blacklisted = False
                                logger.info(f"Worker {worker.name} health check passed. New score: {worker.health_score}")
                            else:
                                worker.health_score = max(0, worker.health_score - 10)  # Reduced penalty
                                if worker.health_score == 0 and not (NO_BLACKLIST_IMAGINE and worker.type == 'imagine'):
                                    worker.is_blacklisted = True
                                    logger.warning(f"Worker {worker.name} blacklisted due to health check failures")
            except Exception as e:
                logger.error(f"Error checking worker health for {worker.name}: {str(e)}")
                worker.health_score = max(0, worker.health_score - 5)  # Further reduced penalty
                if worker.health_score == 0 and not (NO_BLACKLIST_IMAGINE and worker.type == 'imagine'):
                    worker.is_blacklisted = True
                    logger.warning(f"Worker {worker.name} blacklisted due to health check failures")
            
            worker.last_active = datetime.now().isoformat()
            
            db = get_db()
            cursor = db.cursor()
            cursor.execute("""
            UPDATE ai_workers
            SET health_score = ?, is_blacklisted = ?, last_active = ?
            WHERE name = ?
            """, (worker.health_score, worker.is_blacklisted, worker.last_active, worker.name))
            db.commit()
            db.close()
        
        logger.debug(f"Worker health update complete. Sleeping for {HEALTH_CHECK_INTERVAL} seconds")
        await asyncio.sleep(HEALTH_CHECK_INTERVAL)


async def watchdog():
    last_api_check = 0
    while True:
        await asyncio.sleep(300)  # Main watchdog check every 5 minutes
        current_time = time.time()

        # Check queue processor
        if queue_processor_status.is_running and (current_time - queue_processor_status.last_heartbeat) > 30:
            logger.error("Queue processor seems to be frozen. Restarting...")
            queue_processor_status.is_running = False
            asyncio.create_task(start_queue_processor())
            await send_sysop_message("WARNING: Queue processor restarted due to inactivity")

        # Check AI worker health
        await check_ai_worker_health()

        # Periodic API accessibility check (every 5 minutes)
        if current_time - last_api_check >= 300:
            await check_api_accessibility()
            last_api_check = current_time

        



async def check_ai_worker_health():
    chat_workers = [w for w in ai_workers.values() if w.type == 'chat' and not w.is_blacklisted]
    if not chat_workers:
        await send_sysop_message("WARNING: No healthy CHAT workers available")

    imagine_workers = [w for w in ai_workers.values() if w.type == 'imagine' and not w.is_blacklisted]
    if not imagine_workers:
        await send_sysop_message("WARNING: No healthy IMAGINE workers available")

    # Check if any worker has been blacklisted recently
    for worker in ai_workers.values():
        if worker.is_blacklisted:
            await send_sysop_message(f"WARNING: Worker {worker.name} has been blacklisted")



async def check_api_accessibility():
    async def check_api(name, url, timeout=10):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        return True
                    else:
                        logger.warning(f"{name} API returned status {response.status}")
                        return False
        except asyncio.TimeoutError:
            logger.warning(f"{name} API request timed out")
            return False
        except ClientError as e:
            logger.error(f"Error connecting to {name} API: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking {name} API: {str(e)}")
            return False

    # For Claude, we'll check if the API key is valid (don't make an actual query)
    async def check_claude_api():
        headers = {
            "X-API-Key": CLAUDE_API_KEY,
            "Content-Type": "application/json"
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(CLAUDE_ENDPOINT, headers=headers, timeout=10) as response:
                    if response.status == 401:  # Unauthorized, but API is reachable
                        logger.error("Claude API key may be invalid")
                        return False
                    return response.status < 500  # Consider any non-5xx response as OK
            except Exception as e:
                logger.error(f"Error checking Claude API: {str(e)}")
                return False

    # For HuggingFace, we'll use their status endpoint
    huggingface_health = await check_api("HuggingFace", "https://api-inference.huggingface.co/status")
    
    # For Claude, we'll use our custom check
    claude_health = await check_claude_api()

    if not claude_health:
        await send_sysop_message("WARNING: Claude API may be experiencing issues or the API key is invalid")
    if not huggingface_health:
        await send_sysop_message("WARNING: HuggingFace API may be experiencing issues")

    logger.info(f"API Health Check: Claude: {'OK' if claude_health else 'Issues'}, HuggingFace: {'OK' if huggingface_health else 'Issues'}")

    return claude_health and huggingface_health









async def send_sysop_message(message: str):
    logger.warning(message)
    await manager.broadcast({"type": "sysop_message", "message": message})













        
        
async def start_queue_processor():
    global queue_processor_status
    if not queue_processor_status.is_running:
        queue_processor_status.is_running = True
        asyncio.create_task(process_queue())



async def process_queue():
    global queue_processor_status
    queue_processor_status.is_running = True
    logger.info("Starting queue processing loop")
    last_empty_log = 0
    while True:
        try:
            queue_processor_status.last_heartbeat = time.time()
            current_time = time.time()
            queue_size = state.query_queue.qsize()
            
            if queue_size == 0:
                # Log empty queue status once per minute
                if current_time - last_empty_log > 60:
                    logger.info("Queue has been empty for the last minute")
                    last_empty_log = current_time
                await asyncio.sleep(1)  # Sleep for 1 second if queue is empty
                continue
            
            logger.debug(f"Attempting to get query from queue at {current_time:.2f}. Current depth: {queue_size}")
            
            try:
                cancellable_query = await asyncio.wait_for(state.query_queue.get(), timeout=0.1)
                logger.debug(f"Got query from queue. Depth after get: {state.query_queue.qsize()}")
                logger.info(f"Processing query: {cancellable_query.query_data['query']}")
                try:
                    logger.debug("Starting query execution")
                    result = await cancellable_query.run()
                    logger.debug("Query execution completed")
                    if not cancellable_query.cancelled:
                        logger.debug("Processing query result")
                        processing_time = (datetime.now() - datetime.fromisoformat(cancellable_query.query_data['timestamp'])).total_seconds()
                        cost = BASE_COST_PER_QUERY + (processing_time * COST_PER_SECOND)
                        
                        # Update stats
                        query_type = cancellable_query.query_data['query'].query_type
                        if f"{query_type}_time" in system_stats:
                            system_stats[f"{query_type}_time"].append(processing_time)
                        system_stats["total_queries"] += 1
                        save_persistent_stats()
                        
                        result_type = "text"
                        if isinstance(result, bytes):  # Image result
                            base64_image = base64.b64encode(result).decode('utf-8')
                            result = base64_image
                            result_type = "image"
                        elif cancellable_query.query_data['query'].model_type == 'speech':  # Audio result
                            result_type = "audio"
                        
                        await cancellable_query.query_data['websocket'].send_json({
                            "type": "query_result",
                            "result": result,
                            "result_type": result_type,
                            "processing_time": processing_time,
                            "cost": cost
                        })
                        
                        # Insert the query into the database
                        insert_query(cancellable_query.query_data['user'], cancellable_query.query_data['query'], processing_time, cost)
                        
                        update_user_stats(cancellable_query.query_data['user'], processing_time, cost)
                        update_system_stats(get_db(), processing_time, cost)
                        logger.info(f"Query processed successfully. Time: {processing_time:.2f}s, Cost: ${cost:.4f}")
                except asyncio.CancelledError:
                    logger.info(f"Query cancelled: {cancellable_query.query_data['query']}")
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}", exc_info=True)
                    await cancellable_query.query_data['websocket'].send_json({"type": "error", "message": str(e)})
                finally:
                    user_guid = cancellable_query.query_data['user'].guid
                    await state.query_queue.clear_processing(user_guid)
            except asyncio.TimeoutError:
                # This is expected behavior when the queue is empty
                pass
        except Exception as e:
            logger.error(f"Unexpected error in process_queue: {str(e)}", exc_info=True)
            await asyncio.sleep(1)  # Sleep for a bit before retrying
        finally:
            await manager.broadcast({"type": "queue_update", "depth": state.query_queue.qsize(), "total": state.total_workers})


@debug
def update_user_stats(user: User, processing_time: float, cost: float):
    global total_costs_lifetime
    logger.debug(f"Updating stats for user {user.guid}: time +{processing_time}, cost +{cost}")
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
    UPDATE users
    SET total_query_time = total_query_time + ?,
        total_cost = total_cost + ?,
        query_count = query_count + 1
    WHERE guid = ?
    """, (processing_time, cost, user.guid))
    db.commit()
    total_costs_lifetime += cost
    save_persistent_stats()
    db.close()
    logger.info(f"Updated stats for user {user.guid}")



# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_guid: str):
        self.active_connections[user_guid] = websocket
        logger.info(f"New WebSocket connection: {websocket.client}")

    def disconnect(self, user_guid: str):
        if user_guid in self.active_connections:
            del self.active_connections[user_guid]
            logger.info(f"WebSocket disconnected: {user_guid}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            await connection.send_json(message)

    async def send_active_users_to_sysop(self):
        active_users = list(self.active_connections.keys())
        for user_guid, connection in self.active_connections.items():
            user = get_or_create_user(get_db(), user_guid)
            if user and user.is_sysop:
                await connection.send_json({
                    "type": "active_users",
                    "users": active_users
                })

manager = ConnectionManager()

# FastAPI setup
app = FastAPI()

# Serve static files
static_dir = os.path.join(BASE_DIR, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Static files directory mounted: {static_dir}")
else:
    logger.error(f"Static files directory not found: {static_dir}")

# Templates
templates_dir = os.path.join(BASE_DIR, "templates")
if os.path.exists(templates_dir):
    templates = Jinja2Templates(directory=templates_dir)
    logger.info(f"Templates directory set: {templates_dir}")
else:
    logger.error(f"Templates directory not found: {templates_dir}")





@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up the application")
    if not os.path.exists(DATABASE_NAME):
        logger.info("Database not found, initializing...")
        init_db()
    ensure_query_count_column()
    load_persistent_stats()
    reset_stats_if_zero()
    load_ai_workers()
    load_huggingface_models()
    asyncio.create_task(update_worker_health())
    asyncio.create_task(start_queue_processor())
    asyncio.create_task(watchdog())
    await asyncio.sleep(1)  # Give tasks a moment to start
    yield
    # Shutdown
    logger.info("Shutting down the application")







app = FastAPI(lifespan=lifespan)

# API routes
@api_router.post("/chat")
async def chat_api(query: Query):
    return await process_query(query)

@api_router.post("/vision")
async def vision_api(query: Query):
    query.query_type = "vision"
    return await process_query(query)

@api_router.post("/imagine")
async def imagine_api(query: Query):
    query.query_type = "imagine"
    return await process_query(query)

@api_router.post("/whisper")
async def whisper_api(query: Query):
    return await process_speech_to_text(query.audio)

@api_router.post("/bark")
async def bark_api(query: Query):
    return await process_text_to_speech(query.prompt)

# Include the API router
app.include_router(api_router, prefix="/api")

# Routes
@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    logger.info("Serving index page")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    static_file = os.path.join(static_dir, file_path)
    if os.path.exists(static_file):
        logger.info(f"Serving static file: {static_file}")
        return FileResponse(static_file)
    else:
        logger.error(f"Static file not found: {static_file}")
        raise HTTPException(status_code=404, detail="File not found")



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    user_guid = None
    db = get_db()

    try:
        cookies = websocket.cookies
        user_guid = cookies.get("user_guid")
        
        if not user_guid:
            user_guid = str(uuid.uuid4())
            await websocket.send_json({"type": "set_cookie", "name": "user_guid", "value": user_guid})
            logger.info(f"New user connected. Assigned GUID: {user_guid}")

        user = get_or_create_user(db, user_guid)
        await manager.connect(websocket, user_guid)
        await websocket.send_json({"type": "user_info", "data": user.dict()})
        
        if user.is_banned:
            logger.warning(f"Banned user attempted to connect: {user.guid}")
            await websocket.send_json({"type": "error", "message": "You are banned from using this service."})
            return

        cursor = db.cursor()

        cursor.execute("SELECT prompt, processing_time FROM queries WHERE user_guid = ? ORDER BY timestamp DESC LIMIT 5", (user.guid,))
        previous_queries = cursor.fetchall()
        prev_queries_msg = "Your recent queries:\n" + "\n".join([f"Query: {q[0][:30]}... Time: {q[1]:.2f}s" for q in previous_queries])
        await websocket.send_json({"type": "sysop_message", "message": prev_queries_msg})

        await websocket.send_json({"type": "sysop_message", "message": f"Your total lifetime costs: ${user.total_cost:.2f}"})
        await websocket.send_json({"type": "sysop_message", "message": f"System-wide total lifetime costs: ${total_costs_lifetime:.2f}"})

        avg_times = {
            "chat": get_avg_time(system_stats["chat_time"]),
            "vision": get_avg_time(system_stats["vision_time"]),
            "imagine": get_avg_time(system_stats["imagine_time"]),
            "speech_in": get_avg_time(system_stats["speech_in_time"]),
            "speech_out": get_avg_time(system_stats["speech_out_time"])
        }
        avg_times_msg = "Average query service times:\n" + "\n".join([f"{k.capitalize()}: {v:.2f}s" for k, v in avg_times.items()])
        await websocket.send_json({"type": "sysop_message", "message": avg_times_msg})

        connected_users = len(manager.active_connections)
        system_stats["max_connected_users"] = max(system_stats["max_connected_users"], connected_users)
        facts_msg = f"Currently connected users: {connected_users}\nMost users ever connected: {system_stats['max_connected_users']}"
        await websocket.send_json({"type": "sysop_message", "message": facts_msg})

        cursor.execute("""
            SELECT users.nickname, 
                   users.query_count,
                   users.total_cost 
            FROM users 
            WHERE users.guid IN (""" + ",".join(["?" for _ in manager.active_connections]) + ")", 
            tuple(manager.active_connections.keys())
        )
        connected_users_info = cursor.fetchall()
        users_info_msg = "Connected users:\n" + "\n".join([f"Nick: {u[0]}, Queries: {u[1]}, Total cost: ${u[2]:.2f}" for u in connected_users_info])
        await websocket.send_json({"type": "sysop_message", "message": users_info_msg})

        cursor.execute("SELECT * FROM queries WHERE user_guid = ? ORDER BY timestamp DESC LIMIT 10", (user.guid,))
        previous_queries = cursor.fetchall()
        await websocket.send_json({
            "type": "previous_queries",
            "data": [dict(q) for q in previous_queries]
        })
        
        await websocket.send_json({
            "type": "worker_update",
            "workers": [w.dict() for w in ai_workers.values()]
        })
        await websocket.send_json({
            "type": "huggingface_update",
            "models": [m.dict() for m in huggingface_models.values()]
        })
        
        if user.is_sysop:
            await manager.send_active_users_to_sysop()
        
        while True:
            try:
                data = await websocket.receive_json()
                message_type = data.get("type")
                logger.debug(f"Received message from {user.guid}: {message_type}")
                
                if message_type == "set_nickname":
                    await handle_set_nickname(user, data, db, websocket)
                elif message_type == "submit_query":
                    await handle_submit_query(user, data, websocket)
                elif message_type == "speech_to_text":
                    audio_data = data['audio']
                    try:
                        transcription = await process_speech_to_text(audio_data)
                        await websocket.send_json({"type": "transcription_result", "text": transcription})
                        logger.info(f"Speech to text completed for user {user.guid}")
                    except Exception as e:
                        logger.error(f"Error in speech to text processing: {str(e)}")
                        await websocket.send_json({"type": "error", "message": f"Error in speech to text processing: {str(e)}"})
                elif message_type == "text_to_speech":
                    text = data['text']
                    try:
                        audio_result = await process_text_to_speech(text)
                        await websocket.send_json({"type": "speech_result", "audio": audio_result})
                        logger.info(f"Text to speech completed for user {user.guid}")
                    except Exception as e:
                        logger.error(f"Error in text to speech processing: {str(e)}")
                        await websocket.send_json({"type": "error", "message": f"Error in text to speech processing: {str(e)}"})
                elif message_type == "vision_chunk":
                    await handle_vision_chunk(user, data, websocket)
                elif message_type == "get_stats" and user.is_sysop:
                    await handle_get_stats(db, websocket)
                elif message_type == "add_worker" and user.is_sysop:
                    await handle_add_worker(data, db, websocket)
                elif message_type == "remove_worker" and user.is_sysop:
                    await handle_remove_worker(data, db, websocket)
                elif message_type == "add_huggingface_model" and user.is_sysop:
                    await handle_add_huggingface_model(data, db, websocket)
                elif message_type == "remove_huggingface_model" and user.is_sysop:
                    await handle_remove_huggingface_model(data, db, websocket)
                elif message_type == "ban_user" and user.is_sysop:
                    await handle_ban_user(data, db, websocket)
                elif message_type == "unban_user" and user.is_sysop:
                    await handle_unban_user(data, db, websocket)
                elif message_type == "terminate_query" and user.is_sysop:
                    await handle_terminate_query(data, websocket)
                elif message_type == "sysop_message" and user.is_sysop:
                    await handle_sysop_message(data, websocket)
                elif message_type == "get_previous_queries":
                    await handle_get_previous_queries(user, db, websocket)
                elif message_type == "pong":
                    # Client responded to our ping, connection is still alive
                    pass
                else:
                    logger.warning(f"Unknown message type received: {message_type}")
                    await websocket.send_json({"type": "error", "message": "Unknown message type"})

            except WebSocketDisconnect:
                manager.disconnect(user_guid)
                logger.info(f"WebSocket disconnected for user: {user.guid}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket connection: {str(e)}")
                await websocket.send_json({"type": "error", "message": str(e)})

    finally:
        db.close()


# Helper functions for handling different message types
async def handle_set_nickname(user: User, data: dict, db: sqlite3.Connection, websocket: WebSocket):
    new_nickname = data["nickname"]
    cursor = db.cursor()
    try:
        cursor.execute("UPDATE users SET nickname = ? WHERE guid = ?", (new_nickname, user.guid))
        db.commit()
        user = get_or_create_user(db, user.guid)
        await websocket.send_json({"type": "user_info", "data": user.dict()})
        logger.info(f"User {user.guid} updated nickname to {new_nickname}")
    except sqlite3.IntegrityError:
        await websocket.send_json({"type": "error", "message": "Nickname already taken"})
        logger.warning(f"Nickname '{new_nickname}' already taken")

async def handle_submit_query(user: User, data: dict, websocket: WebSocket):
    logger.debug(f"Handling submit query for user {user.guid}")
    if state.query_queue.qsize() >= MAX_QUEUE_SIZE:
        await websocket.send_json({"type": "error", "message": "Queue is full, please try again later"})
        logger.warning("Query rejected: Queue is full")
    else:
        query = Query(**data["query"])
        await state.query_queue.put({
            "query": query,
            "user": user,
            "websocket": websocket,
            "timestamp": datetime.now().isoformat()
        })
        await manager.broadcast({"type": "queue_update", "depth": state.query_queue.qsize(), "total": state.total_workers})
        logger.info(f"Query added to queue for user {user.guid}. Current depth: {state.query_queue.qsize()}")

async def handle_speech_to_text(user: User, data: dict, websocket: WebSocket):
    audio_data = data['audio']
    try:
        transcription = await process_speech_to_text(audio_data)
        await websocket.send_json({"type": "transcription_result", "text": transcription})
        logger.info(f"Speech to text completed for user {user.guid}")
    except Exception as e:
        logger.error(f"Error in speech to text processing: {str(e)}")
        await websocket.send_json({"type": "error", "message": f"Error in speech to text processing: {str(e)}"})

async def handle_text_to_speech(user: User, data: dict, websocket: WebSocket):
    text = data['text']
    try:
        audio_result = await process_text_to_speech(text)
        await websocket.send_json({"type": "speech_result", "audio": audio_result})
        logger.info(f"Text to speech completed for user {user.guid}")
    except Exception as e:
        logger.error(f"Error in text to speech processing: {str(e)}")
        await websocket.send_json({"type": "error", "message": f"Error in text to speech processing: {str(e)}"})

async def handle_vision_chunk(user: User, data: dict, websocket: WebSocket):
    complete_image = await vision_chunker.receive_chunk(data)
    if complete_image:
        processed_image = await process_image(complete_image)
        query = Query(
            prompt=data["prompt"],
            query_type="vision",
            model_type=data["model_type"],
            model_name=data["model_name"],
            image=processed_image
        )
        await state.query_queue.put({
            "query": query,
            "user": user,
            "websocket": websocket,
            "timestamp": datetime.now().isoformat()
        })
        await manager.broadcast({"type": "queue_update", "depth": state.query_queue.qsize(), "total": state.total_workers})
        await websocket.send_json({"type": "vision_upload_complete", "message": "Image upload complete and processed"})
        logger.info(f"Vision query added to queue for user {user.guid}. Current depth: {state.query_queue.qsize()}")
    else:
        await websocket.send_json({"type": "vision_chunk_received", "message": "Chunk received"})

async def handle_get_stats(db: sqlite3.Connection, websocket: WebSocket):
    cursor = db.cursor()
    cursor.execute("SELECT * FROM system_stats ORDER BY last_updated DESC LIMIT 1")
    stats = cursor.fetchone()
    if stats:
        await websocket.send_json({
            "type": "system_stats",
            "data": dict(stats)
        })
    
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    await websocket.send_json({
        "type": "user_stats",
        "data": [dict(u) for u in users]
    })
    
    await websocket.send_json({
        "type": "worker_health",
        "data": [w.dict() for w in ai_workers.values()]
    })
    logger.info(f"Sent system stats to sysop")

async def handle_add_worker(data: dict, db: sqlite3.Connection, websocket: WebSocket):
    new_worker = AIWorker(**data["worker"])
    ai_workers[new_worker.name] = new_worker
    cursor = db.cursor()
    cursor.execute("""
    INSERT INTO ai_workers (name, address, type, health_score, is_blacklisted, last_active)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (new_worker.name, new_worker.address, new_worker.type, new_worker.health_score, new_worker.is_blacklisted, new_worker.last_active))
    db.commit()
    state.total_workers += 1
    await manager.broadcast({"type": "worker_update", "workers": [w.dict() for w in ai_workers.values()]})
    logger.info(f"New worker added: {new_worker.name}")

async def handle_remove_worker(data: dict, db: sqlite3.Connection, websocket: WebSocket):
    worker_name = data["worker_name"]
    if worker_name in ai_workers:
        del ai_workers[worker_name]
        cursor = db.cursor()
        cursor.execute("DELETE FROM ai_workers WHERE name = ?", (worker_name,))
        db.commit()
        state.total_workers -= 1
        await manager.broadcast({"type": "worker_update", "workers": [w.dict() for w in ai_workers.values()]})
        logger.info(f"Worker removed: {worker_name}")

async def handle_add_huggingface_model(data: dict, db: sqlite3.Connection, websocket: WebSocket):
    new_model = HuggingFaceModel(**data["model"])
    huggingface_models[new_model.name] = new_model
    cursor = db.cursor()
    cursor.execute("""
    INSERT INTO huggingface_models (name, type)
    VALUES (?, ?)
    """, (new_model.name, new_model.type))
    db.commit()
    await manager.broadcast({"type": "huggingface_update", "models": [m.dict() for m in huggingface_models.values()]})
    logger.info(f"New Hugging Face model added: {new_model.name}")

async def handle_remove_huggingface_model(data: dict, db: sqlite3.Connection, websocket: WebSocket):
    model_name = data["model_name"]
    if model_name in huggingface_models:
        del huggingface_models[model_name]
        cursor = db.cursor()
        cursor.execute("DELETE FROM huggingface_models WHERE name = ?", (model_name,))
        db.commit()
        await manager.broadcast({"type": "huggingface_update", "models": [m.dict() for m in huggingface_models.values()]})
        logger.info(f"Hugging Face model removed: {model_name}")

async def handle_ban_user(data: dict, db: sqlite3.Connection, websocket: WebSocket):
    banned_guid = data["user_guid"]
    cursor = db.cursor()
    cursor.execute("UPDATE users SET is_banned = 1 WHERE guid = ?", (banned_guid,))
    db.commit()
    await manager.broadcast({"type": "user_banned", "guid": banned_guid})
    logger.warning(f"User banned: {banned_guid}")

async def handle_unban_user(data: dict, db: sqlite3.Connection, websocket: WebSocket):
    unbanned_guid = data["user_guid"]
    cursor = db.cursor()
    cursor.execute("UPDATE users SET is_banned = 0 WHERE guid = ?", (unbanned_guid,))
    db.commit()
    await manager.broadcast({"type": "user_unbanned", "guid": unbanned_guid})
    logger.info(f"User unbanned: {unbanned_guid}")

async def handle_terminate_query(data: dict, websocket: WebSocket):
    terminated_guid = data["user_guid"]
    await state.query_queue.remove_by_guid(terminated_guid)
    await manager.broadcast({"type": "query_terminated", "guid": terminated_guid})
    await manager.broadcast({"type": "queue_update", "depth": state.query_queue.qsize(), "total": state.total_workers})
    logger.warning(f"Query terminated for user: {terminated_guid}")

async def handle_sysop_message(data: dict, websocket: WebSocket):
    await manager.broadcast({"type": "sysop_message", "message": data["message"]})
    logger.info(f"Sysop message broadcast: {data['message']}")

async def handle_get_previous_queries(user: User, db: sqlite3.Connection, websocket: WebSocket):
    cursor = db.cursor()
    cursor.execute("SELECT * FROM queries WHERE user_guid = ? ORDER BY timestamp DESC LIMIT 10", (user.guid,))
    previous_queries = cursor.fetchall()
    await websocket.send_json({
        "type": "previous_queries",
        "data": [dict(q) for q in previous_queries]
    })
    logger.info(f"Sent previous queries to user: {user.guid}")

async def update_active_users_periodically():
    while True:
        await manager.send_active_users_to_sysop()
        await asyncio.sleep(60)

async def update_system_stats_periodically():
    while True:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT * FROM system_stats ORDER BY last_updated DESC LIMIT 1")
        stats = cursor.fetchone()
        if stats:
            await manager.broadcast({
                "type": "system_stats",
                "data": dict(stats)
            })
        db.close()
        await asyncio.sleep(300)

@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(update_active_users_periodically())
    asyncio.create_task(update_system_stats_periodically())
    if torch.cuda.is_available():
        asyncio.create_task(log_gpu_memory_usage())
    
    # Log initial system information
    logger.info(f"Operating System: {os.name}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Device being used: {device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Current GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0)}")
        logger.info(f"Initial GPU memory reserved: {torch.cuda.memory_reserved(0)}")

# Debug routes
@app.get("/debug/")
async def debug_home(request: Request):
    logger.debug("Serving debug home page")
    return templates.TemplateResponse("debug.html", {"request": request})

@app.post("/debug/init_db")
async def init_db_route(confirm: bool = Form(...)):
    if confirm:
        try:
            init_db()
            logger.info("Database initialized successfully via debug route")
            return RedirectResponse(url="/", status_code=303)
        except Exception as e:
            logger.error(f"Error initializing database via debug route: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        logger.info("Database initialization cancelled")
        return {"message": "Operation cancelled"}

@app.get("/debug/check_sysop/{guid}")
async def check_sysop(guid: str):
    logger.debug(f"Checking sysop status for GUID: {guid}")
    db = get_db()
    user = get_or_create_user(db, guid)
    db.close()
    return {"is_sysop": user.is_sysop}

@app.post("/debug/set_sysop/{guid}")
async def set_sysop(guid: str):
    logger.info(f"Setting sysop status for GUID: {guid}")
    db = get_db()
    cursor = db.cursor()
    cursor.execute("UPDATE users SET is_sysop = ? WHERE guid = ?", (True, guid))
    db.commit()
    db.close()
    return {"message": f"User {guid} is now a sysop"}

@app.get("/debug/system_status")
async def system_status():
    logger.debug("Fetching system status")
    return {
        "database_exists": os.path.exists(DATABASE_NAME),
        "total_workers": state.total_workers,
        "queue_depth": state.query_queue.qsize(),
        "huggingface_models": len(huggingface_models),
    }

# Main execution
if __name__ == "__main__":
    import uvicorn
    
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler("uvicorn.log")
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    uvicorn_logger.addHandler(file_handler)
    
    logger.info("Starting the main server...")
    uvicorn.run("webgui:app", host=HOST, port=PORT, log_config=None)
