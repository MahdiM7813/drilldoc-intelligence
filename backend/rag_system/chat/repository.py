from __future__ import annotations
import json, sqlite3, uuid
from datetime import datetime, timezone
from pathlib import Path

def now() -> str: return datetime.now(timezone.utc).isoformat()

class ChatRepository:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve(); self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript('''
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY,title TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY,conversation_id TEXT NOT NULL,role TEXT NOT NULL,content TEXT NOT NULL,created_at TEXT NOT NULL,FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE);
            CREATE TABLE IF NOT EXISTS message_sources(id INTEGER PRIMARY KEY AUTOINCREMENT,message_id TEXT NOT NULL,document_id TEXT NOT NULL,filename TEXT NOT NULL,chunk_id TEXT NOT NULL,page INTEGER NOT NULL,section TEXT,score REAL NOT NULL,excerpt TEXT NOT NULL,FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE);
            CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY,filename TEXT NOT NULL,title TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL,number_of_pages INTEGER);
            CREATE TABLE IF NOT EXISTS document_chunks(chunk_id TEXT PRIMARY KEY,document_id TEXT NOT NULL,page INTEGER NOT NULL,section TEXT,chunk_type TEXT NOT NULL,content TEXT NOT NULL,FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE);
            ''')
    def _connect(self):
        db=sqlite3.connect(self.path); db.row_factory=sqlite3.Row; db.execute('PRAGMA foreign_keys=ON'); return db
    def create_conversation(self,title='New chat'):
        ident=str(uuid.uuid4()); stamp=now()
        with self._connect() as db: db.execute('INSERT INTO conversations VALUES (?,?,?,?)',(ident,title,stamp,stamp))
        return ident
    def add_message(self,conversation_id,role,content,sources=()):
        ident=str(uuid.uuid4()); stamp=now()
        with self._connect() as db:
            db.execute('INSERT INTO messages VALUES (?,?,?,?,?)',(ident,conversation_id,role,content,stamp))
            db.executemany('INSERT INTO message_sources(message_id,document_id,filename,chunk_id,page,section,score,excerpt) VALUES (?,?,?,?,?,?,?,?)',[(ident,s['document_id'],s['filename'],s['chunk_id'],s['page'],s.get('section'),s['score'],s['excerpt']) for s in sources])
            db.execute('UPDATE conversations SET updated_at=? WHERE id=?',(stamp,conversation_id))
        return ident
    def messages(self,conversation_id):
        with self._connect() as db:
            rows=db.execute('SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at',(conversation_id,)).fetchall()
            return [dict(row)|{'sources':[dict(x) for x in db.execute('SELECT document_id,filename,page,section,chunk_id,score,excerpt FROM message_sources WHERE message_id=?',(row['id'],)).fetchall()]} for row in rows]
    def conversations(self):
        with self._connect() as db:
            return [dict(row) for row in db.execute('SELECT * FROM conversations ORDER BY updated_at DESC').fetchall()]
    def rename_conversation(self, conversation_id, title):
        with self._connect() as db:
            db.execute('UPDATE conversations SET title=?,updated_at=? WHERE id=?',(str(title)[:120],now(),conversation_id))
    def delete_conversation(self, conversation_id):
        with self._connect() as db: db.execute('DELETE FROM conversations WHERE id=?',(conversation_id,))
