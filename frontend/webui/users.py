"""Tiny CLI for users.toml entries.

    echo -n 'secret' | python -m webui.users hash     -> prints a bcrypt hash
"""

import sys

import bcrypt


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "hash":
        sys.exit(__doc__)
    password = sys.stdin.read().rstrip("\n").encode()
    if len(password) < 10:
        sys.exit("password must be at least 10 characters")
    print(bcrypt.hashpw(password, bcrypt.gensalt(12)).decode())


if __name__ == "__main__":
    main()
