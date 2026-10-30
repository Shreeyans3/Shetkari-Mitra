"""All database work lives here. Uses only Python's built-in sqlite3 module."""
import sqlite3
from contextlib import closing
from pathlib import Path

from security import hash_password, verify_password

DB_PATH = Path(__file__).parent / "users.db"

# Extra profile columns. Older users.db files are upgraded automatically.
EXTRA_COLUMNS = {
    "mobile":  "TEXT NOT NULL DEFAULT ''",
    "dob":     "TEXT NOT NULL DEFAULT ''",
    "gender":  "TEXT NOT NULL DEFAULT 'undisclosed'",
    "marital": "TEXT NOT NULL DEFAULT 'undisclosed'",
}
PUBLIC_FIELDS = "id, full_name, email, mobile, dob, gender, marital, created_at"


class DatabaseError(Exception):
    """Something went wrong talking to SQLite."""


class DuplicateEmailError(Exception):
    """That email is already registered."""


class InvalidCredentialsError(Exception):
    """Wrong email or password."""


class UserNotFoundError(Exception):
    """No user with that email."""


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the users table, and add any new profile columns to an old database."""
    try:
        with closing(get_connection()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name     TEXT NOT NULL,
                    email         TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            have = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
            for col, ddl in EXTRA_COLUMNS.items():
                if col not in have:
                    conn.execute(f"ALTER TABLE users ADD COLUMN {col} {ddl}")
    except sqlite3.Error as e:
        raise DatabaseError(f"Could not initialise database: {e}")


# ---------------------------- CREATE ----------------------------
def create_user(full_name, email, password, mobile=""):
    try:
        with closing(get_connection()) as conn, conn:
            conn.execute(
                "INSERT INTO users (full_name, email, password_hash, mobile) VALUES (?, ?, ?, ?)",
                (full_name.strip(), email.strip().lower(), hash_password(password), mobile.strip()),
            )
    except sqlite3.IntegrityError:
        raise DuplicateEmailError("That email is already registered.")
    except sqlite3.Error as e:
        raise DatabaseError(f"Database error: {e}")


# ----------------------------- READ -----------------------------
def get_user_by_email(email):
    try:
        with closing(get_connection()) as conn:
            row = conn.execute(
                f"SELECT {PUBLIC_FIELDS} FROM users WHERE email = ?",
                (email.strip().lower(),),
            ).fetchone()
    except sqlite3.Error as e:
        raise DatabaseError(f"Database error: {e}")
    if row is None:
        raise UserNotFoundError("No user found with that email.")
    return row


def get_all_users():
    try:
        with closing(get_connection()) as conn:
            return conn.execute(f"SELECT {PUBLIC_FIELDS} FROM users ORDER BY id").fetchall()
    except sqlite3.Error as e:
        raise DatabaseError(f"Database error: {e}")


def authenticate_user(email, password):
    """Sign-in check. Returns the user row, or raises InvalidCredentialsError."""
    try:
        with closing(get_connection()) as conn:
            row = conn.execute(
                f"SELECT {PUBLIC_FIELDS}, password_hash FROM users WHERE email = ?",
                (email.strip().lower(),),
            ).fetchone()
    except sqlite3.Error as e:
        raise DatabaseError(f"Database error: {e}")
    # Same error for "no such email" and "wrong password".
    if row is None or not verify_password(password, row["password_hash"]):
        raise InvalidCredentialsError("Invalid email or password.")
    return row


# ---------------------------- UPDATE ----------------------------
def update_user(email, current_password, full_name=None, mobile=None, dob=None,
                gender=None, marital=None, new_password=None):
    """Update profile fields and/or password. Requires the current password."""
    authenticate_user(email, current_password)  # raises if wrong
    changes = {
        "full_name": full_name, "mobile": mobile, "dob": dob,
        "gender": gender, "marital": marital,
    }
    changes = {k: v for k, v in changes.items() if v is not None}
    if new_password:
        changes["password_hash"] = hash_password(new_password)
    if not changes:
        return
    sets = ", ".join(f"{col} = ?" for col in changes)  # column names come from the dict above, never from users
    try:
        with closing(get_connection()) as conn, conn:
            conn.execute(
                f"UPDATE users SET {sets} WHERE email = ?",
                (*changes.values(), email.strip().lower()),
            )
    except sqlite3.Error as e:
        raise DatabaseError(f"Database error: {e}")


# ---------------------------- DELETE ----------------------------
def delete_user(email, password):
    """Delete a user. Requires the password."""
    authenticate_user(email, password)  # raises if wrong
    try:
        with closing(get_connection()) as conn, conn:
            cur = conn.execute("DELETE FROM users WHERE email = ?", (email.strip().lower(),))
            if cur.rowcount == 0:
                raise UserNotFoundError("No user found with that email.")
    except sqlite3.Error as e:
        raise DatabaseError(f"Database error: {e}")
