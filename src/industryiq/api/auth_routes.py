"""HTTP routes for authentication: register, login, and 'who am I'.

A thin layer over :class:`AuthService`: validate input, call the service, return
a bearer token (register/login) or the current account (me). Clients then send
the token as ``Authorization: Bearer <token>`` on the conversation routes, where
:func:`industryiq.api.deps.get_current_user` resolves it back to a user.
"""

import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from industryiq.api.deps import get_auth_service, get_current_user
from industryiq.core.auth import AuthService, EmailAlreadyRegistered, User

Auth = Annotated[AuthService, Depends(get_auth_service)]
CurrentUser = Annotated[User, Depends(get_current_user)]

router = APIRouter(prefix="/auth", tags=["auth"])

# A pragmatic format check (one '@', a dotted domain), not full RFC 5322 -- we
# accept what users actually type and let real delivery be proven elsewhere.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class Credentials(BaseModel):
    """Registration/login payload: an email and a password."""

    email: str = Field(min_length=3, max_length=320)
    # bcrypt only hashes the first 72 bytes; cap here so the limit is explicit
    # and a login can never silently disagree with the registration it came from.
    password: str = Field(min_length=8, max_length=72)

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, value: str) -> str:
        value = value.strip().lower()
        if not _EMAIL_RE.match(value):
            raise ValueError("invalid email address")
        return value


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(credentials: Credentials, auth: Auth) -> TokenResponse:
    try:
        user = auth.register(credentials.email, credentials.password)
    except EmailAlreadyRegistered:
        raise HTTPException(status_code=409, detail="Email already registered") from None
    return TokenResponse(access_token=auth.create_token(user))


@router.post("/login", response_model=TokenResponse)
def login(credentials: Credentials, auth: Auth) -> TokenResponse:
    user = auth.authenticate(credentials.email, credentials.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return TokenResponse(access_token=auth.create_token(user))


@router.get("/me", response_model=UserResponse)
def me(user: CurrentUser) -> UserResponse:
    return UserResponse(id=user.id, email=user.email)
