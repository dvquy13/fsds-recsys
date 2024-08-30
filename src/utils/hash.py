import hashlib


def hash_string_to_int(s: str) -> int:
    # Use hashlib to create a SHA-256 hash of the string
    hash_object = hashlib.sha256(s.encode())
    # Convert the hash to a hexadecimal string
    hex_dig = hash_object.hexdigest()
    # Convert the hexadecimal string to an integer
    hash_int = int(hex_dig, 16)
    return hash_int
