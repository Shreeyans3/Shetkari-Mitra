"""Password hashing using only Python's standard library (hashlib + secrets)."""
import hashlib
import hmac
import secrets

ITERATIONS = 200_000  # more iterations = slower for attackers


def hash_password(password: str) -> str:
    """Return 'iterations$salt$hash' (all text), safe to store in the database."""
    salt = secrets.token_hex(16)  # a new random salt for every password
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), ITERATIONS
    )
    return f"{ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Check a password against a stored 'iterations$salt$hash' string."""
    try:
        iterations, salt, expected = stored.split("$")
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations)
        )
    except (ValueError, TypeError):
        return False
    # compare_digest avoids timing attacks
    return hmac.compare_digest(digest.hex(), expected)
