import sqlite3
from datetime import datetime

DB_NAME = "chatbot.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def insert_chat(question, answer):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO chat_history (question, answer, timestamp)
        VALUES (?, ?, ?)
    """, (question, answer, timestamp))
    conn.commit()
    conn.close()

def fetch_all_chats():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat_history")
    rows = cursor.fetchall()
    conn.close()
    return rows

def chat_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer FROM chat_history")
    history = cursor.fetchall()
    conn.close()
    return history