"""Evaluation scheduler - runs daily evaluation using APScheduler.

Supports cron-based scheduling with configurable timezone (default UTC).
"""

import asyncio
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Global scheduler instance
_scheduler: Optional[AsyncIOScheduler] = None


def get_scheduler() -> AsyncIOScheduler:
    """Get or create the scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler()
    return _scheduler


async def scheduled_evaluation_job():
    """Job that runs scheduled evaluation.
    
    Finds active datasets and runs evaluation on each.
    """
    from app.db.mysql import get_db_pool
    from app.evaluation.runner import run_evaluation
    
    logger.info("Starting scheduled evaluation run")
    
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Get all active datasets
                await cursor.execute(
                    "SELECT id, name FROM evaluation_datasets WHERE active = TRUE"
                )
                datasets = await cursor.fetchall()
        
        if not datasets:
            logger.info("No active datasets for scheduled evaluation")
            return
        
        for dataset_id, dataset_name in datasets:
            try:
                logger.info(
                    f"Running scheduled evaluation for dataset",
                    extra={"dataset_id": dataset_id, "dataset_name": dataset_name}
                )
                
                result = await run_evaluation(dataset_id=dataset_id)
                
                logger.info(
                    f"Scheduled evaluation completed",
                    extra={
                        "dataset_id": dataset_id,
                        "run_id": result.get("run_id"),
                        "hit_rate": result.get("metrics", {}).get("hit_rate", 0),
                    }
                )
                
            except Exception as e:
                logger.error(
                    f"Scheduled evaluation failed for dataset {dataset_id}: {e}",
                    extra={"dataset_id": dataset_id}
                )
                continue
        
        logger.info(f"Scheduled evaluation completed for {len(datasets)} datasets")
        
    except Exception as e:
        logger.error(f"Scheduled evaluation job failed: {e}")


def _run_async_job():
    """Wrapper to run async job in sync context."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.create_task(scheduled_evaluation_job())
    else:
        loop.run_until_complete(scheduled_evaluation_job())


def start_evaluation_scheduler():
    """Start the evaluation scheduler based on settings."""
    if not settings.evaluation.scheduler_enabled:
        logger.info("Evaluation scheduler is disabled")
        return
    
    scheduler = get_scheduler()
    
    # Parse cron schedule (e.g., "0 2 * * *" for 2 AM daily)
    cron_parts = settings.evaluation.cron_schedule.split()
    if len(cron_parts) != 5:
        logger.error(f"Invalid cron schedule: {settings.evaluation.cron_schedule}")
        return
    
    minute, hour, day, month, day_of_week = cron_parts
    
    trigger = CronTrigger(
        minute=minute,
        hour=hour,
        day=day,
        month=month,
        day_of_week=day_of_week,
        timezone=settings.evaluation.cron_timezone,
    )
    
    scheduler.add_job(
        _run_async_job,
        trigger=trigger,
        id="evaluation_daily",
        name="Daily RAG Evaluation",
        replace_existing=True,
    )
    
    if not scheduler.running:
        scheduler.start()
    
    logger.info(
        "Evaluation scheduler started",
        extra={
            "cron_schedule": settings.evaluation.cron_schedule,
            "timezone": settings.evaluation.cron_timezone,
        }
    )


def stop_evaluation_scheduler():
    """Stop the evaluation scheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Evaluation scheduler stopped")
    _scheduler = None


def get_schedule_info() -> dict:
    """Get current schedule information."""
    scheduler = get_scheduler()
    job = scheduler.get_job("evaluation_daily")
    
    return {
        "enabled": settings.evaluation.scheduler_enabled,
        "cron_schedule": settings.evaluation.cron_schedule,
        "timezone": settings.evaluation.cron_timezone,
        "next_run": job.next_run_time.isoformat() if job and job.next_run_time else None,
        "running": scheduler.running if scheduler else False,
    }


def update_schedule(cron_schedule: str, timezone: str = "UTC"):
    """Update the evaluation schedule.
    
    Note: This updates the running scheduler but doesn't persist to config.
    """
    scheduler = get_scheduler()
    
    cron_parts = cron_schedule.split()
    if len(cron_parts) != 5:
        raise ValueError(f"Invalid cron schedule: {cron_schedule}")
    
    minute, hour, day, month, day_of_week = cron_parts
    
    trigger = CronTrigger(
        minute=minute,
        hour=hour,
        day=day,
        month=month,
        day_of_week=day_of_week,
        timezone=timezone,
    )
    
    scheduler.reschedule_job("evaluation_daily", trigger=trigger)
    
    logger.info(
        "Evaluation schedule updated",
        extra={"cron_schedule": cron_schedule, "timezone": timezone}
    )
