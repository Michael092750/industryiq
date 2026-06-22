"""Password hashing -- bcrypt, keeping plaintext out of storage.

Isolated behind two functions so the algorithm can change (or be faked) without
the auth service knowing. bcrypt only considers the first 72 bytes of a password;
the API caps input length up front, so no silent truncation surprises a caller.
"""

import bcrypt


def hash_password(password: str) -> str:
    """Return a salted bcrypt hash of ``password`` (safe to store)."""
    hashed: bytes = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Return whether ``password`` matches a previously stored bcrypt hash."""
    matches: bool = bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    return matches
