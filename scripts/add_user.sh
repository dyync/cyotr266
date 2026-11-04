#!/usr/bin/env bash
# Add (or replace) a login in config/users.toml.   Usage: scripts/add_user.sh <name> [--admin]
set -euo pipefail
cd "$(dirname "$0")/.."
name="${1:?usage: $0 <name> [--admin]}"
[[ "$name" =~ ^[A-Za-z0-9._-]{1,64}$ ]] || { echo "invalid user name"; exit 1; }
admin=false; [[ "${2:-}" == "--admin" ]] && admin=true

read -rsp "Password for $name (min 10 chars): " pw; echo
read -rsp "Repeat: " pw2; echo
[[ "$pw" == "$pw2" ]] || { echo "passwords differ"; exit 1; }

hash=$(printf '%s' "$pw" | docker compose run --rm --no-deps -T frontend python -m webui.users hash)
touch config/users.toml
# drop an existing block for this user, then append the new one
python3 - "$name" <<'PY'
import re, sys
p = "config/users.toml"; name = sys.argv[1]
s = open(p).read()
s = re.sub(rf'(?ms)^\[users\.{re.escape(name)}\]\n.*?(?=^\[|\Z)', '', s)
open(p, "w").write(s.rstrip() + ("\n\n" if s.strip() else ""))
PY
printf '[users.%s]\npassword_hash = "%s"\nadmin = %s\n' "$name" "$hash" "$admin" >> config/users.toml
chmod 644 config/users.toml  # read by uid 1000 in the container; contains only bcrypt hashes
echo "Added $name (admin=$admin). The frontend picks it up automatically."
