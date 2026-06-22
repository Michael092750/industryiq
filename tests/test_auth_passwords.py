from industryiq.core.auth.passwords import hash_password, verify_password


def test_hash_is_not_plaintext() -> None:
    hashed = hash_password("correct horse battery")
    assert hashed != "correct horse battery"
    assert hashed.startswith("$2")  # bcrypt prefix


def test_verify_accepts_correct_password() -> None:
    hashed = hash_password("s3cret-password")
    assert verify_password("s3cret-password", hashed) is True


def test_verify_rejects_wrong_password() -> None:
    hashed = hash_password("s3cret-password")
    assert verify_password("not-the-password", hashed) is False


def test_same_password_hashes_differently_each_time() -> None:
    # A random salt per hash means two hashes of the same password differ,
    # yet both verify.
    a = hash_password("repeated-password")
    b = hash_password("repeated-password")
    assert a != b
    assert verify_password("repeated-password", a)
    assert verify_password("repeated-password", b)
