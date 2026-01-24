"""
DateTime Tool.

Date and time operations for queries involving temporal data.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from app.tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class DateTimeTool(BaseTool):
    """
    Date and time operations tool.
    
    Provides:
    - Current date/time
    - Date parsing and formatting
    - Date arithmetic (add/subtract days, months, etc.)
    - Date comparisons and differences
    """
    
    name = "datetime"
    description = "Get current date/time, parse dates, calculate date differences, or perform date arithmetic."
    parameters = [
        ToolParameter(
            name="operation",
            description="The operation to perform",
            type="string",
            required=True,
            enum=["now", "parse", "format", "add", "diff", "compare"]
        ),
        ToolParameter(
            name="date",
            description="Date string to operate on (for parse, format, add, diff, compare)",
            type="string",
            required=False
        ),
        ToolParameter(
            name="date2",
            description="Second date for diff or compare operations",
            type="string",
            required=False
        ),
        ToolParameter(
            name="days",
            description="Number of days to add (for add operation)",
            type="number",
            required=False,
            default=0
        ),
        ToolParameter(
            name="format",
            description="Output format (e.g., '%Y-%m-%d', '%B %d, %Y')",
            type="string",
            required=False,
            default="%Y-%m-%d"
        ),
        ToolParameter(
            name="timezone",
            description="Timezone (e.g., 'UTC', 'US/Eastern'). Default is local time.",
            type="string",
            required=False
        )
    ]
    requires_client_scope = False
    
    # Common date formats to try when parsing
    DATE_FORMATS = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m-%d-%Y",
        "%m/%d/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ]
    
    async def execute(
        self,
        operation: str,
        date: Optional[str] = None,
        date2: Optional[str] = None,
        days: int = 0,
        format: str = "%Y-%m-%d",
        timezone: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute date/time operation.
        """
        try:
            if operation == "now":
                return self._get_now(format, timezone)
            
            elif operation == "parse":
                if not date:
                    return ToolResult.fail("Date string required for parse operation")
                return self._parse_date(date)
            
            elif operation == "format":
                if not date:
                    return ToolResult.fail("Date string required for format operation")
                return self._format_date(date, format)
            
            elif operation == "add":
                if not date:
                    return ToolResult.fail("Date string required for add operation")
                return self._add_days(date, days, format)
            
            elif operation == "diff":
                if not date or not date2:
                    return ToolResult.fail("Two dates required for diff operation")
                return self._date_diff(date, date2)
            
            elif operation == "compare":
                if not date or not date2:
                    return ToolResult.fail("Two dates required for compare operation")
                return self._compare_dates(date, date2)
            
            else:
                return ToolResult.fail(f"Unknown operation: {operation}")
                
        except Exception as e:
            logger.error(f"DateTime error: {e}")
            return ToolResult.fail(str(e))
    
    def _get_now(self, format: str, timezone: Optional[str]) -> ToolResult:
        """Get current date/time."""
        now = datetime.now()
        
        # Apply timezone if specified
        if timezone:
            try:
                import pytz
                tz = pytz.timezone(timezone)
                now = datetime.now(tz)
            except ImportError:
                logger.warning("pytz not available, using local time")
            except Exception as e:
                return ToolResult.fail(f"Invalid timezone: {timezone}")
        
        formatted = now.strftime(format)
        
        return ToolResult.ok(
            formatted,
            timestamp=now.isoformat(),
            timezone=timezone or "local"
        )
    
    def _parse_date(self, date_str: str) -> ToolResult:
        """Parse a date string into standard format."""
        parsed = self._try_parse(date_str)
        
        if parsed is None:
            return ToolResult.fail(f"Could not parse date: {date_str}")
        
        return ToolResult.ok(
            parsed.strftime("%Y-%m-%d"),
            iso=parsed.isoformat(),
            year=parsed.year,
            month=parsed.month,
            day=parsed.day,
            weekday=parsed.strftime("%A")
        )
    
    def _format_date(self, date_str: str, format: str) -> ToolResult:
        """Format a date string."""
        parsed = self._try_parse(date_str)
        
        if parsed is None:
            return ToolResult.fail(f"Could not parse date: {date_str}")
        
        try:
            formatted = parsed.strftime(format)
            return ToolResult.ok(formatted)
        except ValueError as e:
            return ToolResult.fail(f"Invalid format: {e}")
    
    def _add_days(self, date_str: str, days: int, format: str) -> ToolResult:
        """Add days to a date."""
        parsed = self._try_parse(date_str)
        
        if parsed is None:
            return ToolResult.fail(f"Could not parse date: {date_str}")
        
        result = parsed + timedelta(days=days)
        
        return ToolResult.ok(
            result.strftime(format),
            original=date_str,
            days_added=days,
            iso=result.isoformat()
        )
    
    def _date_diff(self, date1_str: str, date2_str: str) -> ToolResult:
        """Calculate difference between two dates."""
        date1 = self._try_parse(date1_str)
        date2 = self._try_parse(date2_str)
        
        if date1 is None:
            return ToolResult.fail(f"Could not parse date1: {date1_str}")
        if date2 is None:
            return ToolResult.fail(f"Could not parse date2: {date2_str}")
        
        diff = date2 - date1
        days = diff.days
        
        # Calculate years, months approximately
        years = days // 365
        remaining_days = days % 365
        months = remaining_days // 30
        final_days = remaining_days % 30
        
        return ToolResult.ok(
            f"{days} days",
            total_days=days,
            years=years,
            months=months,
            days=final_days,
            date1=date1.strftime("%Y-%m-%d"),
            date2=date2.strftime("%Y-%m-%d")
        )
    
    def _compare_dates(self, date1_str: str, date2_str: str) -> ToolResult:
        """Compare two dates."""
        date1 = self._try_parse(date1_str)
        date2 = self._try_parse(date2_str)
        
        if date1 is None:
            return ToolResult.fail(f"Could not parse date1: {date1_str}")
        if date2 is None:
            return ToolResult.fail(f"Could not parse date2: {date2_str}")
        
        if date1 < date2:
            comparison = "before"
            description = f"{date1_str} is before {date2_str}"
        elif date1 > date2:
            comparison = "after"
            description = f"{date1_str} is after {date2_str}"
        else:
            comparison = "equal"
            description = f"{date1_str} is the same as {date2_str}"
        
        return ToolResult.ok(
            description,
            comparison=comparison,
            date1=date1.strftime("%Y-%m-%d"),
            date2=date2.strftime("%Y-%m-%d")
        )
    
    def _try_parse(self, date_str: str) -> Optional[datetime]:
        """Try to parse a date string using common formats."""
        date_str = date_str.strip()
        
        for fmt in self.DATE_FORMATS:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
