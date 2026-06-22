"""AuthService: the registration and login policy.

Depends only on the :class:`UserStore` port plus the pure password/token helpers,
so it unit-tests with an in-memory store and zero network. It owns the rules that
are neither storage nor HTTP: normalize the email, reject a duplicate, verify a
password in constant time (no user-enumeration via timing), and mint/read tokens.
"""

from industryiq.core.auth.models import User
from industryiq.core.auth.passwords import hash_password, verify_password
from industryiq.core.auth.ports import UserStore
from industryiq.core.auth.tokens import decode_token, encode_token

# A precomputed hash to verify against when the email is unknown, so a failed
# login costs the same whether or not the account exists.
_DUMMY_HASH = hash_password("constant-time-placeholder")


class EmailAlreadyRegistered(Exception):
    """Raised when registering an email that already has an account."""


class AuthService:
    """Register accounts, authenticate logins, and issue/verify access tokens."""

    def __init__(
        self,
        users: UserStore,
        *,
        secret: str,
        algorithm: str = "HS256",
        expiry_minutes: int = 60 * 24,
    ) -> None:
        self._users = users
        self._secret = secret
        self._algorithm = algorithm
        self._expiry_minutes = expiry_minutes

    @staticmethod
    def normalize_email(email: str) -> str:
        """Canonical form used as the unique key: trimmed and lowercased."""
        return email.strip().lower()

    def register(self, email: str, password: str) -> User:
        """Create a new account, or raise :class:`EmailAlreadyRegistered`."""
        email = self.normalize_email(email)
        if self._users.get_by_email(email) is not None:
            raise EmailAlreadyRegistered(email)
        return self._users.create(email, hash_password(password))

    def authenticate(self, email: str, password: str) -> User | None:
        """Return the user for valid credentials, else ``None``."""
        user = self._users.get_by_email(self.normalize_email(email))
        if user is None:
            verify_password(password, _DUMMY_HASH)  # equalize timing; ignore result
            return None
        if not verify_password(password, user.password_hash):
            return None
        return user

    def create_token(self, user: User) -> str:
        """Mint a bearer access token for ``user``."""
        return encode_token(
            user.id,
            secret=self._secret,
            algorithm=self._algorithm,
            expiry_minutes=self._expiry_minutes,
        )

    def identify(self, token: str) -> User | None:
        """Resolve a bearer token to its user, or ``None`` if it is not valid."""
        user_id = decode_token(token, secret=self._secret, algorithm=self._algorithm)
        if user_id is None:
            return None
        return self._users.get_by_id(user_id)
