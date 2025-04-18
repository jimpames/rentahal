import asyncio
import uuid
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
import sqlite3
from datetime import datetime

app = FastAPI()

DATABASE_NAME = "llm_broker.db"

def get_db():
    db = sqlite3.connect(DATABASE_NAME)
    db.row_factory = sqlite3.Row
    return db

def ensure_single_sysop(guid):
    db = get_db()
    cursor = db.cursor()
    
    # Remove sysop status from all users
    cursor.execute("UPDATE users SET is_sysop = 0")
    
    # Set the new user as sysop
    cursor.execute("""
    INSERT OR REPLACE INTO users (guid, nickname, is_sysop, total_query_time, total_cost, is_banned)
    VALUES (?, ?, 1, 0, 0, 0)
    """, (guid, f"sysop_{guid[:8]}"))
    
    # Ensure related entries exist
    cursor.execute("""
    INSERT OR IGNORE INTO queries (user_guid, query_type, model_type, model_name, prompt, processing_time, cost, timestamp)
    VALUES (?, 'chat', 'system', 'system', 'Sysop initialization', 0, 0, ?)
    """, (guid, datetime.now().isoformat()))
    
    db.commit()
    db.close()

@app.get("/", response_class=HTMLResponse)
async def besysop(request: Request, response: Response):
    guid = str(uuid.uuid4())
    ensure_single_sysop(guid)
    
    response.set_cookie(key="user_guid", value=guid, httponly=True, max_age=31536000)  # 1 year expiry
    
    return f"""
    <html>
        <head>
            <title>Sysop GUID Assignment</title>
        </head>
        <body>
            <h1>Sysop GUID Assigned</h1>
            <p>Your new Sysop GUID is: {guid}</p>
            <p>This GUID has been set as a cookie and is now the only Sysop in the database.</p>
            <p>Please use this GUID to access the main AI LLM Broker interface as a Sysop.</p>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5666)
