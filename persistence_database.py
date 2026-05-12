import sqlite3
from datetime import datetime
import os

# Ensure the database folder exists
os.makedirs("database", exist_ok=True)
DB_NAME = "database/chat_history.db"

def init_db():
    """Database aur tables banata hai agar pehle se na hon."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        
        # Threads table (Sidebar mein chat history dikhane ke liye)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Messages table (Chat ke messages save karne ke liye)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (thread_id) REFERENCES threads(id)
            )
        ''')
        conn.commit()

def save_thread(thread_id, title):
    """Nayi chat (thread) ka title save karta hai."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO threads (id, title, updated_at)
            VALUES (?, ?, ?)
        ''', (str(thread_id), title, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()

def get_all_threads():
    """Database se saari purani chats uthata hai taake sidebar mein dikha sake."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, title FROM threads ORDER BY updated_at DESC')
        rows = cursor.fetchall()
        # Streamlit ke format mein list return karna
        return [{"id": row[0], "title": row[1], "messages": []} for row in rows]

def save_message(thread_id, role, content):
    """Naya message save karta hai."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO messages (thread_id, role, content, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (str(thread_id), role, content, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        
        # Thread ka time update karna taake wo sidebar mein sab se upar aa jaye
        cursor.execute('''
            UPDATE threads SET updated_at = ? WHERE id = ?
        ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(thread_id)))
        
        conn.commit()

def get_chat_history(thread_id):
    """Kisi khaas chat ke saare messages nikalta hai."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT role, content FROM messages WHERE thread_id = ? ORDER BY id ASC
        ''', (str(thread_id),))
        rows = cursor.fetchall()
        return [{"role": row[0], "content": row[1]} for row in rows]

def delete_thread(thread_id):
    """Chat aur uske saare messages delete karta hai."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM messages WHERE thread_id = ?', (str(thread_id),))
        cursor.execute('DELETE FROM threads WHERE id = ?', (str(thread_id),))
        conn.commit()