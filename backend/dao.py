import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash


class SqliteDao:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def init_db(self):
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO app_settings (key, value)
                VALUES ('llm_config', ?)
                """,
                (json.dumps(default_llm_config()),),
            )

        if not self.get_user_by_username("admin"):
            self.create_user("admin", "nimda", role="admin")
        if not self.get_user_by_username("guest"):
            self.create_user("guest", "guest", role="user")

    def get_user_by_username(self, username):
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        return dict(row) if row else None

    def get_user_by_id(self, user_id):
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_users(self):
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, username, role, created_at, updated_at
                FROM users
                ORDER BY role DESC, username ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def create_user(self, username, password, role="user"):
        password_hash = generate_password_hash(password)
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (username, password_hash, role, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (username, password_hash, role),
            )
            user_id = cursor.lastrowid
        return self.get_user_by_id(user_id)

    def update_user_password(self, user_id, password):
        password_hash = generate_password_hash(password)
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE users
                SET password_hash = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (password_hash, user_id),
            )
        return self.get_user_by_id(user_id)

    def verify_user(self, username, password):
        user = self.get_user_by_username(username)
        if not user:
            return None
        if not check_password_hash(user["password_hash"], password):
            return None
        return user

    def get_llm_config(self):
        with self.connect() as connection:
            row = connection.execute(
                "SELECT value FROM app_settings WHERE key = 'llm_config'"
            ).fetchone()
        if not row:
            return default_llm_config()
        try:
            stored = json.loads(row["value"])
        except json.JSONDecodeError:
            stored = {}
        config = default_llm_config()
        config.update({key: value for key, value in stored.items() if key in config})
        return config

    def update_llm_config(self, config):
        merged = default_llm_config()
        merged.update({key: value for key, value in config.items() if key in merged})
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES ('llm_config', ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (json.dumps(merged),),
            )
        return self.get_llm_config()


def default_llm_config():
    return {
        "provider": "auto",
        "qwen_api_key": "",
        "qwen_model": "qwen-turbo",
        "openai_api_key": "",
        "openai_base_url": "https://api.openai.com/v1",
        "openai_model": "gpt-4o-mini",
    }
