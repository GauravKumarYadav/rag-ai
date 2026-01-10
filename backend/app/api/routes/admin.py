"""
Admin API routes for audit visibility and simple user management.

These endpoints rely on the existing audit log schema and user table.
"""

from datetime import datetime, timedelta
from typing import List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr, Field

from app.auth.dependencies import require_superuser
from app.auth.password import hash_password
from app.config import settings
from app.core.logging import get_logger
from app.db.mysql import get_db_pool


router = APIRouter()
logger = get_logger(__name__)


class AuditLogEntry(BaseModel):
    id: int
    timestamp: datetime
    user_id: Optional[str]
    username: Optional[str]
    client_id: Optional[str]
    action: str
    resource: Optional[str]
    method: Optional[str]
    path: Optional[str]
    status_code: Optional[int]
    ip_address: Optional[str]
    duration_ms: Optional[float]
    request_summary: Optional[str]


class AuditLogResponse(BaseModel):
    items: List[AuditLogEntry]
    total: int
    page: int
    page_size: int


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    password: str = Field(..., min_length=6)
    is_superuser: bool = False


class UpdateUserRequest(BaseModel):
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=6)
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None


@router.get("/audit-logs", response_model=AuditLogResponse)
async def list_audit_logs(
    username: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    method: Optional[str] = Query(None),
    path_prefix: Optional[str] = Query(None),
    status_code: Optional[int] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    current_user: dict = Depends(require_superuser),
):
    """Paginated audit log listing."""

    conditions = []
    params = []

    if username:
        conditions.append("username LIKE %s")
        params.append(f"%{username}%")
    if action:
        conditions.append("action = %s")
        params.append(action)
    if method:
        conditions.append("method = %s")
        params.append(method.upper())
    if path_prefix:
        conditions.append("path LIKE %s")
        params.append(f"{path_prefix}%")
    if status_code:
        conditions.append("status_code = %s")
        params.append(status_code)
    if start_date:
        conditions.append("timestamp >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("timestamp <= %s")
        params.append(end_date)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                f"SELECT COUNT(*) FROM audit_logs {where_clause}",
                params,
            )
            total = (await cursor.fetchone())[0]

            offset = (page - 1) * page_size
            await cursor.execute(
                f"""
                SELECT id, timestamp, user_id, username, client_id, action, resource,
                       method, path, status_code, ip_address, duration_ms, request_summary
                FROM audit_logs
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT %s OFFSET %s
                """,
                (*params, page_size, offset),
            )
            rows = await cursor.fetchall()

    items = [
        AuditLogEntry(
            id=row[0],
            timestamp=row[1],
            user_id=row[2],
            username=row[3],
            client_id=row[4],
            action=row[5],
            resource=row[6],
            method=row[7],
            path=row[8],
            status_code=row[9],
            ip_address=row[10],
            duration_ms=row[11],
            request_summary=row[12],
        )
        for row in rows
    ]

    return AuditLogResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/audit-logs/actions", response_model=List[str])
async def list_actions(current_user: dict = Depends(require_superuser)):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT DISTINCT action FROM audit_logs ORDER BY action")
            rows = await cursor.fetchall()
    return [row[0] for row in rows]


@router.get("/users")
async def list_users(
    active_only: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(require_superuser),
):
    where_clause = "WHERE is_active = TRUE" if active_only else ""

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                f"""
                SELECT id, username, email, is_active, is_superuser, created_at
                FROM users
                {where_clause}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
            rows = await cursor.fetchall()

    return [
        {
            "id": row[0],
            "username": row[1],
            "email": row[2],
            "is_active": bool(row[3]),
            "is_superuser": bool(row[4]),
            "created_at": row[5].isoformat() if row[5] else None,
        }
        for row in rows
    ]


@router.post("/users")
async def create_user(
    request: CreateUserRequest,
    current_user: dict = Depends(require_superuser),
):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT id FROM users WHERE username = %s", (request.username,))
            if await cursor.fetchone():
                raise HTTPException(status_code=400, detail="Username already exists")

            await cursor.execute(
                """
                INSERT INTO users (id, username, email, hashed_password, is_active, is_superuser)
                VALUES (%s, %s, %s, %s, TRUE, %s)
                """,
                (
                    str(uuid.uuid4()),
                    request.username,
                    request.email,
                    hash_password(request.password),
                    request.is_superuser,
                ),
            )

    return {"message": "User created"}


@router.patch("/users/{user_id}")
async def update_user(
    user_id: str,
    request: UpdateUserRequest,
    current_user: dict = Depends(require_superuser),
):
    updates = []
    params = []

    if request.email is not None:
        updates.append("email = %s")
        params.append(request.email)
    if request.password is not None:
        updates.append("hashed_password = %s")
        params.append(hash_password(request.password))
    if request.is_active is not None:
        updates.append("is_active = %s")
        params.append(request.is_active)
    if request.is_superuser is not None:
        updates.append("is_superuser = %s")
        params.append(request.is_superuser)

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    params.append(user_id)

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE id = %s",
                params,
            )
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="User not found")

    return {"message": "User updated"}


@router.get("/stats")
async def get_stats(current_user: dict = Depends(require_superuser)):
    now = datetime.utcnow()
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT COUNT(*) FROM users")
            total_users = (await cursor.fetchone())[0]

            await cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = TRUE")
            active_users = (await cursor.fetchone())[0]

            await cursor.execute("SELECT COUNT(*) FROM users WHERE is_superuser = TRUE")
            superusers = (await cursor.fetchone())[0]

            await cursor.execute("SELECT COUNT(*) FROM audit_logs WHERE timestamp >= %s", (day_ago,))
            requests_24h = (await cursor.fetchone())[0]

            await cursor.execute("SELECT COUNT(*) FROM audit_logs WHERE timestamp >= %s", (week_ago,))
            requests_7d = (await cursor.fetchone())[0]

            await cursor.execute(
                "SELECT COUNT(*) FROM audit_logs WHERE timestamp >= %s AND status_code >= 400",
                (day_ago,),
            )
            errors_24h = (await cursor.fetchone())[0]

            await cursor.execute(
                """
                SELECT AVG(duration_ms) FROM audit_logs 
                WHERE timestamp >= %s AND duration_ms IS NOT NULL
                """,
                (day_ago,),
            )
            avg_response_time = (await cursor.fetchone())[0] or 0

    error_rate = (errors_24h / requests_24h * 100) if requests_24h else 0

    return {
        "total_users": total_users,
        "active_users": active_users,
        "superusers": superusers,
        "total_requests_24h": requests_24h,
        "total_requests_7d": requests_7d,
        "error_rate_24h": round(error_rate, 2),
        "avg_response_time_ms": round(avg_response_time, 2),
    }


@router.get("/config")
async def get_config(current_user: dict = Depends(require_superuser)):
    return {
        "log_level": settings.logging.level,
        "log_dir": settings.logging.log_dir,
        "log_file": settings.logging.log_file,
        "log_rotation_max_bytes": settings.logging.max_bytes,
        "log_rotation_backup_count": settings.logging.backup_count,
        "evaluation_default_sample_size": settings.evaluation.default_sample_size,
        "evaluation_cron_schedule": settings.evaluation.cron_schedule,
        "evaluation_timezone": settings.evaluation.cron_timezone,
    }
