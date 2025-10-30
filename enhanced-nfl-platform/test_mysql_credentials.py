#!/usr/bin/env python3
"""
Validate the MySQL credentials defined in `.env` without guessing or printing secrets.
"""

from __future__ import annotations

import argparse
import os
from getpass import getpass
from pathlib import Path
from typing import Optional, Tuple

import pymysql
from dotenv import load_dotenv


def load_credentials(env_file: Path) -> Tuple[Optional[str], Optional[str], str, str, str]:
    """Load connection parameters from an env file."""
    if env_file.exists():
        load_dotenv(env_file)
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    database = os.getenv("MYSQL_DATABASE", "nfl_ai")
    return user, password, host, port, database


def mask(secret: Optional[str]) -> str:
    if secret is None:
        return "<not-set>"
    if secret == "":
        return "<empty>"
    if len(secret) <= 4:
        return "*" * len(secret)
    return f"{secret[:2]}{'*' * (len(secret) - 4)}{secret[-2:]}"


def ensure_credentials(
    env_file: Path,
    user: Optional[str],
    password: Optional[str],
    host: str,
    port: str,
    database: str,
) -> Tuple[str, str, str, int, str]:
    """Prompt for missing values, returning a complete credential set."""
    if not user:
        user = input("MySQL user: ").strip()
    if password is None:
        password = getpass("MySQL password (leave blank for none): ")
    if not host:
        host = "localhost"
    if not port:
        port = "3306"
    if not database:
        database = "nfl_ai"

    password = password or ""

    print(f"Using credentials from `{env_file}`")
    print(f"  host: {host}")
    print(f"  port: {port}")
    print(f"  user: {user}")
    print(f"  password: {mask(password)}")
    print(f"  database: {database}")

    return user, password, host, int(port), database


def check_connection(user: str, password: str, host: str, port: int, database: str) -> bool:
    """Attempt to connect and run a simple query."""
    try:
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4",
        )
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1;")
            cursor.fetchone()
        connection.close()
        return True
    except Exception as exc:  # pylint: disable=broad-except
        print(f"❌ Connection failed: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Test configured MySQL credentials safely.")
    parser.add_argument(
        "--env-file",
        default=Path(".env"),
        type=Path,
        help="Path to the environment file (default: .env)",
    )
    args = parser.parse_args()

    user, password, host, port, database = load_credentials(args.env_file)
    user, password, host, port, database = ensure_credentials(
        args.env_file, user, password, host, port, database
    )

    print("\n🔍 Testing MySQL credentials...")
    if check_connection(user, password, host, port, database):
        print("✅ Credentials are valid.")
        return 0

    print("❌ Unable to connect using the provided credentials.")
    print("   Verify that the database service is running and `.env` values are correct.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
