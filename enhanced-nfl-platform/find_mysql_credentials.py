#!/usr/bin/env python3
"""
Utility to validate the MySQL credentials defined in `.env`.
The script no longer brute-forces common combinations and never prints secrets.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from getpass import getpass
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv


def load_credentials(env_file: Path) -> Tuple[Optional[str], Optional[str], str, str]:
    """Load MySQL credentials from an env file."""
    if env_file.exists():
        load_dotenv(env_file)
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    return user, password, host, port


def mask(secret: Optional[str]) -> str:
    """Return a masked version of a secret for safe logging."""
    if secret is None:
        return "<not-set>"
    if secret == "":
        return "<empty>"
    if len(secret) <= 4:
        return "*" * len(secret)
    return f"{secret[:2]}{'*' * (len(secret) - 4)}{secret[-2:]}"


def mysql_select_1(user: str, password: str, host: str, port: str) -> bool:
    """Run a simple `SELECT 1;` test using the mysql CLI."""
    command = [
        "mysql",
        "-u",
        user,
        f"-p{password}",
        "-h",
        host,
        "-P",
        str(port),
        "-e",
        "SELECT 1;",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        if result.stderr:
            print(f"❌ connection failed: {result.stderr.strip()}")
        return False
    return True


def ensure_credentials(
    env_file: Path, user: Optional[str], password: Optional[str], host: str, port: str
) -> Tuple[str, str, str, str]:
    """Prompt for missing credentials and return a complete tuple."""
    if not user:
        user = input("MySQL user: ").strip()
    if password is None:
        password = getpass("MySQL password (leave blank for none): ")
    password = password or ""
    if not host:
        host = "localhost"
    if not port:
        port = "3306"

    print(f"Using credentials from `{env_file}`")
    print(f"  host: {host}")
    print(f"  port: {port}")
    print(f"  user: {user}")
    print(f"  password: {mask(password)}")
    return user, password, host, port


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MySQL credentials safely.")
    parser.add_argument(
        "--env-file",
        default=Path(".env"),
        type=Path,
        help="Path to the environment file that defines MYSQL_* variables (default: .env)",
    )
    args = parser.parse_args()

    user, password, host, port = load_credentials(args.env_file)
    user, password, host, port = ensure_credentials(args.env_file, user, password, host, port)

    print("\n🔍 Testing MySQL connectivity...")
    if mysql_select_1(user, password, host, port):
        print("✅ Connection successful.")
        print("   Tip: update `.env` and your deployment secrets with these values.")
        return 0

    print("❌ Unable to connect with the supplied credentials.")
    print("   Double-check that MySQL is running and the values in `.env` are correct.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
