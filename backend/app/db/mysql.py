"""MySQL database connection and audit logging utilities."""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import aiomysql

from app.config import settings

logger = logging.getLogger(__name__)

# Global connection pool
_pool: Optional[aiomysql.Pool] = None


async def get_db_pool() -> aiomysql.Pool:
    """Get or create the MySQL connection pool."""
    global _pool
    if _pool is None:
        _pool = await aiomysql.create_pool(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password.get_secret_value(),
            db=settings.mysql_database,
            minsize=1,
            maxsize=settings.mysql_pool_size,
            autocommit=True,
            charset="utf8mb4",
        )
        logger.info(f"MySQL connection pool created: {settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}")
    return _pool


async def close_db_pool() -> None:
    """Close the MySQL connection pool."""
    global _pool
    if _pool is not None:
        _pool.close()
        await _pool.wait_closed()
        _pool = None
        logger.info("MySQL connection pool closed")


async def init_audit_tables() -> None:
    """Initialize the audit_logs and users tables if they don't exist."""
    pool = await get_db_pool()
    
    create_audit_logs_table = """
    CREATE TABLE IF NOT EXISTS audit_logs (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        timestamp DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
        user_id VARCHAR(255),
        username VARCHAR(255),
        client_id VARCHAR(255),
        action VARCHAR(100) NOT NULL,
        resource VARCHAR(255),
        method VARCHAR(10),
        path VARCHAR(500),
        status_code INT,
        ip_address VARCHAR(45),
        user_agent TEXT,
        duration_ms FLOAT,
        request_summary TEXT,
        INDEX idx_user_id (user_id),
        INDEX idx_client_id (client_id),
        INDEX idx_timestamp (timestamp),
        INDEX idx_action (action)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    
    create_users_table = """
    CREATE TABLE IF NOT EXISTS users (
        id VARCHAR(36) PRIMARY KEY,
        username VARCHAR(100) UNIQUE NOT NULL,
        email VARCHAR(255) UNIQUE,
        hashed_password VARCHAR(255) NOT NULL,
        is_active BOOLEAN DEFAULT TRUE,
        is_superuser BOOLEAN DEFAULT FALSE,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_username (username),
        INDEX idx_email (email)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(create_audit_logs_table)
            await cursor.execute(create_users_table)
            logger.info("Audit tables initialized successfully")


async def log_audit_event(
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    client_id: Optional[str] = None,
    action: str = "request",
    resource: Optional[str] = None,
    method: Optional[str] = None,
    path: Optional[str] = None,
    status_code: Optional[int] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    duration_ms: Optional[float] = None,
    request_summary: Optional[str] = None,
) -> None:
    """Log an audit event to the database."""
    try:
        pool = await get_db_pool()
        
        insert_query = """
        INSERT INTO audit_logs (
            user_id, username, client_id, action, resource, method, path,
            status_code, ip_address, user_agent, duration_ms, request_summary
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    insert_query,
                    (
                        user_id,
                        username,
                        client_id,
                        action,
                        resource,
                        method,
                        path,
                        status_code,
                        ip_address,
                        user_agent,
                        duration_ms,
                        request_summary,
                    ),
                )
    except Exception as e:
        logger.error(f"Failed to log audit event: {e}")


async def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Retrieve a user by username."""
    try:
        pool = await get_db_pool()
        
        query = """
        SELECT id, username, email, hashed_password, is_active, is_superuser, created_at
        FROM users
        WHERE username = %s AND is_active = TRUE
        """
        
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(query, (username,))
                result = await cursor.fetchone()
                return result
    except Exception as e:
        logger.error(f"Failed to get user by username: {e}")
        return None


async def create_user(
    user_id: str,
    username: str,
    hashed_password: str,
    email: Optional[str] = None,
    is_superuser: bool = False,
) -> bool:
    """Create a new user in the database."""
    try:
        pool = await get_db_pool()
        
        insert_query = """
        INSERT INTO users (id, username, email, hashed_password, is_superuser)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    insert_query,
                    (user_id, username, email, hashed_password, is_superuser),
                )
                logger.info(f"User created: {username}")
                return True
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        return False
