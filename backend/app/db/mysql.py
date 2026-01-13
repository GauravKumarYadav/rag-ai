"""MySQL database connection and audit logging utilities."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    """Initialize the audit_logs, users, and clients tables if they don't exist."""
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
    
    create_clients_table = """
    CREATE TABLE IF NOT EXISTS clients (
        id VARCHAR(36) PRIMARY KEY,
        name VARCHAR(255) NOT NULL UNIQUE,
        aliases JSON,
        metadata JSON,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_name (name),
        INDEX idx_created_at (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(create_audit_logs_table)
            await cursor.execute(create_users_table)
            await cursor.execute(create_clients_table)
            logger.info("Database tables initialized successfully (audit_logs, users, clients)")


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


async def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a user by ID."""
    try:
        pool = await get_db_pool()
        
        query = """
        SELECT id, username, email, is_active, is_superuser, created_at
        FROM users
        WHERE id = %s
        """
        
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(query, (user_id,))
                result = await cursor.fetchone()
                return result
    except Exception as e:
        logger.error(f"Failed to get user by ID: {e}")
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


# ============================================================
# Client Management Functions
# ============================================================

async def init_clients_table() -> None:
    """Initialize the clients table if it doesn't exist."""
    pool = await get_db_pool()
    
    create_clients_table = """
    CREATE TABLE IF NOT EXISTS clients (
        id VARCHAR(36) PRIMARY KEY,
        name VARCHAR(255) NOT NULL UNIQUE,
        aliases JSON,
        metadata JSON,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_name (name),
        INDEX idx_created_at (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(create_clients_table)
            logger.info("Clients table initialized successfully")


async def create_client_db(
    client_id: str,
    name: str,
    aliases: List[str] = None,
    metadata: Dict[str, Any] = None,
) -> Optional[Dict[str, Any]]:
    """Create a new client in the database."""
    try:
        pool = await get_db_pool()
        
        insert_query = """
        INSERT INTO clients (id, name, aliases, metadata)
        VALUES (%s, %s, %s, %s)
        """
        
        aliases_json = json.dumps(aliases or [])
        metadata_json = json.dumps(metadata or {})
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    insert_query,
                    (client_id, name, aliases_json, metadata_json),
                )
                logger.info(f"Client created: {name} (ID: {client_id})")
        
        # Return the created client
        return await get_client_db(client_id)
    except Exception as e:
        logger.error(f"Failed to create client: {e}")
        return None


async def get_client_db(client_id: str) -> Optional[Dict[str, Any]]:
    """Get a client by ID."""
    try:
        pool = await get_db_pool()
        
        query = """
        SELECT id, name, aliases, metadata, created_at, updated_at
        FROM clients
        WHERE id = %s
        """
        
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(query, (client_id,))
                result = await cursor.fetchone()
                if result:
                    # Parse JSON fields
                    result['aliases'] = json.loads(result['aliases']) if result['aliases'] else []
                    result['metadata'] = json.loads(result['metadata']) if result['metadata'] else {}
                return result
    except Exception as e:
        logger.error(f"Failed to get client: {e}")
        return None


async def get_client_by_name_db(name: str) -> Optional[Dict[str, Any]]:
    """Get a client by exact name match."""
    try:
        pool = await get_db_pool()
        
        query = """
        SELECT id, name, aliases, metadata, created_at, updated_at
        FROM clients
        WHERE LOWER(name) = LOWER(%s)
        """
        
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(query, (name,))
                result = await cursor.fetchone()
                if result:
                    result['aliases'] = json.loads(result['aliases']) if result['aliases'] else []
                    result['metadata'] = json.loads(result['metadata']) if result['metadata'] else {}
                return result
    except Exception as e:
        logger.error(f"Failed to get client by name: {e}")
        return None


async def list_clients_db() -> List[Dict[str, Any]]:
    """List all clients."""
    try:
        pool = await get_db_pool()
        
        query = """
        SELECT id, name, aliases, metadata, created_at, updated_at
        FROM clients
        ORDER BY name
        """
        
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(query)
                results = await cursor.fetchall()
                for result in results:
                    result['aliases'] = json.loads(result['aliases']) if result['aliases'] else []
                    result['metadata'] = json.loads(result['metadata']) if result['metadata'] else {}
                return results
    except Exception as e:
        logger.error(f"Failed to list clients: {e}")
        return []


async def update_client_db(
    client_id: str,
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Update a client's information."""
    try:
        pool = await get_db_pool()
        
        # Build dynamic update query
        updates = []
        params = []
        
        if name is not None:
            updates.append("name = %s")
            params.append(name)
        if aliases is not None:
            updates.append("aliases = %s")
            params.append(json.dumps(aliases))
        if metadata is not None:
            updates.append("metadata = %s")
            params.append(json.dumps(metadata))
        
        if not updates:
            return await get_client_db(client_id)
        
        params.append(client_id)
        query = f"UPDATE clients SET {', '.join(updates)} WHERE id = %s"
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(query, tuple(params))
                if cursor.rowcount == 0:
                    return None
                logger.info(f"Client updated: {client_id}")
        
        return await get_client_db(client_id)
    except Exception as e:
        logger.error(f"Failed to update client: {e}")
        return None


async def delete_client_db(client_id: str) -> bool:
    """Delete a client by ID."""
    try:
        pool = await get_db_pool()
        
        query = "DELETE FROM clients WHERE id = %s"
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(query, (client_id,))
                if cursor.rowcount > 0:
                    logger.info(f"Client deleted: {client_id}")
                    return True
                return False
    except Exception as e:
        logger.error(f"Failed to delete client: {e}")
        return False


async def search_clients_db(query: str) -> List[Dict[str, Any]]:
    """Search clients by name substring."""
    try:
        pool = await get_db_pool()
        
        search_query = """
        SELECT id, name, aliases, metadata, created_at, updated_at
        FROM clients
        WHERE LOWER(name) LIKE LOWER(%s)
        ORDER BY name
        """
        
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(search_query, (f"%{query}%",))
                results = await cursor.fetchall()
                for result in results:
                    result['aliases'] = json.loads(result['aliases']) if result['aliases'] else []
                    result['metadata'] = json.loads(result['metadata']) if result['metadata'] else {}
                return results
    except Exception as e:
        logger.error(f"Failed to search clients: {e}")
        return []


# ============================================================
# User-Client Authorization Functions
# ============================================================

async def init_user_clients_table() -> None:
    """Initialize the user_clients table if it doesn't exist."""
    pool = await get_db_pool()
    
    create_user_clients_table = """
    CREATE TABLE IF NOT EXISTS user_clients (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id VARCHAR(36) NOT NULL,
        client_id VARCHAR(36) NOT NULL,
        role ENUM('viewer', 'editor', 'admin') DEFAULT 'viewer',
        assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        assigned_by VARCHAR(36),
        UNIQUE KEY unique_user_client (user_id, client_id),
        INDEX idx_user_id (user_id),
        INDEX idx_client_id (client_id),
        INDEX idx_role (role)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(create_user_clients_table)
            logger.info("User-clients table initialized successfully")


async def assign_user_to_client(
    user_id: str,
    client_id: str,
    role: str = "viewer",
    assigned_by: Optional[str] = None,
) -> bool:
    """
    Assign a user to a client with a specific role.
    
    Args:
        user_id: The user's ID
        client_id: The client's ID
        role: One of 'viewer', 'editor', 'admin'
        assigned_by: ID of the user making the assignment
        
    Returns:
        True if assignment was successful
    """
    try:
        pool = await get_db_pool()
        
        # Use INSERT ... ON DUPLICATE KEY UPDATE to handle re-assignments
        insert_query = """
        INSERT INTO user_clients (user_id, client_id, role, assigned_by)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE role = VALUES(role), assigned_by = VALUES(assigned_by), assigned_at = CURRENT_TIMESTAMP
        """
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    insert_query,
                    (user_id, client_id, role, assigned_by),
                )
                logger.info(f"User {user_id} assigned to client {client_id} with role {role}")
                return True
    except Exception as e:
        logger.error(f"Failed to assign user to client: {e}")
        return False


async def remove_user_from_client(user_id: str, client_id: str) -> bool:
    """
    Remove a user's access to a client.
    
    Args:
        user_id: The user's ID
        client_id: The client's ID
        
    Returns:
        True if removal was successful
    """
    try:
        pool = await get_db_pool()
        
        delete_query = """
        DELETE FROM user_clients
        WHERE user_id = %s AND client_id = %s
        """
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(delete_query, (user_id, client_id))
                if cursor.rowcount > 0:
                    logger.info(f"User {user_id} removed from client {client_id}")
                    return True
                return False
    except Exception as e:
        logger.error(f"Failed to remove user from client: {e}")
        return False


async def get_user_clients_db(user_id: str) -> List[Dict[str, Any]]:
    """
    Get all clients a user has access to.
    
    Args:
        user_id: The user's ID
        
    Returns:
        List of client assignments with role information
    """
    try:
        pool = await get_db_pool()
        
        query = """
        SELECT 
            uc.client_id,
            uc.role,
            uc.assigned_at,
            uc.assigned_by,
            c.name as client_name,
            c.aliases as client_aliases
        FROM user_clients uc
        JOIN clients c ON uc.client_id = c.id
        WHERE uc.user_id = %s
        ORDER BY c.name
        """
        
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(query, (user_id,))
                results = await cursor.fetchall()
                for result in results:
                    if result.get('client_aliases'):
                        result['client_aliases'] = json.loads(result['client_aliases'])
                    else:
                        result['client_aliases'] = []
                return results
    except Exception as e:
        logger.error(f"Failed to get user clients: {e}")
        return []


async def get_user_client_ids(user_id: str) -> List[str]:
    """
    Get just the client IDs a user has access to.
    
    Args:
        user_id: The user's ID
        
    Returns:
        List of client IDs
    """
    try:
        pool = await get_db_pool()
        
        query = """
        SELECT client_id FROM user_clients WHERE user_id = %s
        """
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(query, (user_id,))
                results = await cursor.fetchall()
                return [row[0] for row in results]
    except Exception as e:
        logger.error(f"Failed to get user client IDs: {e}")
        return []


async def get_client_users_db(client_id: str) -> List[Dict[str, Any]]:
    """
    Get all users who have access to a client.
    
    Args:
        client_id: The client's ID
        
    Returns:
        List of user assignments with role information
    """
    try:
        pool = await get_db_pool()
        
        query = """
        SELECT 
            u.id,
            uc.user_id,
            uc.role,
            uc.assigned_at,
            uc.assigned_by,
            u.username,
            u.email,
            u.is_superuser
        FROM user_clients uc
        JOIN users u ON uc.user_id = u.id
        WHERE uc.client_id = %s AND u.is_active = TRUE
        ORDER BY u.username
        """
        
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(query, (client_id,))
                return await cursor.fetchall()
    except Exception as e:
        logger.error(f"Failed to get client users: {e}")
        return []


async def check_user_client_access(
    user_id: str, 
    client_id: str,
    required_role: Optional[str] = None,
) -> bool:
    """
    Check if a user has access to a specific client.
    
    Args:
        user_id: The user's ID
        client_id: The client's ID
        required_role: If specified, check that user has at least this role
        
    Returns:
        True if user has access (and required role if specified)
    """
    try:
        pool = await get_db_pool()
        
        query = """
        SELECT role FROM user_clients
        WHERE user_id = %s AND client_id = %s
        """
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(query, (user_id, client_id))
                result = await cursor.fetchone()
                
                if not result:
                    return False
                
                if not required_role:
                    return True
                
                # Role hierarchy: admin > editor > viewer
                role_hierarchy = {'viewer': 1, 'editor': 2, 'admin': 3}
                user_role_level = role_hierarchy.get(result[0], 0)
                required_level = role_hierarchy.get(required_role, 0)
                
                return user_role_level >= required_level
    except Exception as e:
        logger.error(f"Failed to check user client access: {e}")
        return False


async def get_user_client_role(user_id: str, client_id: str) -> Optional[str]:
    """
    Get the user's role for a specific client.
    
    Args:
        user_id: The user's ID
        client_id: The client's ID
        
    Returns:
        The role ('viewer', 'editor', 'admin') or None if no access
    """
    try:
        pool = await get_db_pool()
        
        query = """
        SELECT role FROM user_clients
        WHERE user_id = %s AND client_id = %s
        """
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(query, (user_id, client_id))
                result = await cursor.fetchone()
                return result[0] if result else None
    except Exception as e:
        logger.error(f"Failed to get user client role: {e}")
        return None


async def update_user_client_role(
    user_id: str,
    client_id: str,
    new_role: str,
    updated_by: Optional[str] = None,
) -> bool:
    """
    Update a user's role for a client.
    
    Args:
        user_id: The user's ID
        client_id: The client's ID
        new_role: The new role to assign
        updated_by: ID of the user making the update
        
    Returns:
        True if update was successful
    """
    try:
        pool = await get_db_pool()
        
        update_query = """
        UPDATE user_clients
        SET role = %s, assigned_by = %s, assigned_at = CURRENT_TIMESTAMP
        WHERE user_id = %s AND client_id = %s
        """
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    update_query,
                    (new_role, updated_by, user_id, client_id),
                )
                if cursor.rowcount > 0:
                    logger.info(f"Updated user {user_id} role to {new_role} for client {client_id}")
                    return True
                return False
    except Exception as e:
        logger.error(f"Failed to update user client role: {e}")
        return False


async def ensure_global_client_exists() -> bool:
    """
    Ensure the global client exists in the database.
    This client is accessible by all users.
    
    Returns:
        True if global client exists or was created
    """
    try:
        pool = await get_db_pool()
        
        # Check if global client exists
        check_query = "SELECT id FROM clients WHERE id = 'global'"
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(check_query)
                result = await cursor.fetchone()
                
                if result:
                    return True
                
                # Create global client
                insert_query = """
                INSERT INTO clients (id, name, aliases, metadata)
                VALUES ('global', 'Global', '["global", "shared", "common"]', 
                        '{"description": "Global client for shared documents accessible by all users"}')
                """
                await cursor.execute(insert_query)
                logger.info("Global client created")
                return True
    except Exception as e:
        logger.error(f"Failed to ensure global client exists: {e}")
        return False
