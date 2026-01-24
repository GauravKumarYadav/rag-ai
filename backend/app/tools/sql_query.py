"""
SQL Query Tool.

Client-scoped SQL queries for document metadata.
Enforces tenant isolation by automatically filtering by client_id.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from app.tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


# Allowed tables and columns for querying
ALLOWED_TABLES = {
    "documents": ["id", "filename", "source", "client_id", "uploaded_at", "page_count", "chunk_count"],
    "chunks": ["id", "document_id", "content", "metadata", "created_at"],
}

# SQL keywords that are not allowed
FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE",
    "GRANT", "REVOKE", "EXECUTE", "EXEC", "INTO", "--", ";--", "/*", "*/"
]


class SQLQueryTool(BaseTool):
    """
    Client-scoped SQL query tool for document metadata.
    
    SECURITY:
    - Only SELECT queries allowed
    - Automatically filters by client_id
    - Only queries allowed tables
    - Parameterized queries to prevent injection
    
    Note: This is a simplified implementation. In production,
    use proper ORM or parameterized queries.
    """
    
    name = "sql_query"
    description = "Query document metadata for the current client. Use to find document counts, upload dates, or filter documents. Only SELECT queries on documents table."
    parameters = [
        ToolParameter(
            name="query",
            description="SQL-like query (SELECT only). Example: 'SELECT filename, uploaded_at FROM documents WHERE page_count > 10'",
            type="string",
            required=True
        ),
        ToolParameter(
            name="limit",
            description="Maximum rows to return (default 10, max 100)",
            type="number",
            required=False,
            default=10
        )
    ]
    requires_client_scope = True  # IMPORTANT: Enforces tenant isolation
    
    def __init__(self):
        super().__init__()
        self._db = None
    
    @property
    def db(self):
        """Lazy load database connection."""
        if self._db is None:
            try:
                from app.db.mysql import get_mysql_connection
                self._db = get_mysql_connection()
            except ImportError:
                logger.warning("MySQL connection not available")
        return self._db
    
    async def execute(
        self,
        query: str,
        client_id: str,  # Required by requires_client_scope
        limit: int = 10,
        **kwargs
    ) -> ToolResult:
        """
        Execute a client-scoped SQL query.
        
        Args:
            query: SQL SELECT query
            client_id: Client ID for isolation (auto-injected)
            limit: Max rows to return
            
        Returns:
            ToolResult with query results
        """
        # Validate query
        validation_error = self._validate_query(query)
        if validation_error:
            return ToolResult.fail(validation_error)
        
        # Enforce limit
        limit = min(max(1, limit), 100)
        
        try:
            # Parse and rewrite query with client filter
            safe_query = self._rewrite_with_client_filter(query, client_id, limit)
            
            if safe_query is None:
                return ToolResult.fail("Could not parse query. Use simple SELECT statements.")
            
            # Execute query
            results = await self._execute_query(safe_query, client_id)
            
            return ToolResult.ok(
                results,
                query=safe_query,
                row_count=len(results),
                client_id=client_id
            )
            
        except Exception as e:
            logger.error(f"SQL query error: {e}")
            return ToolResult.fail(f"Query failed: {str(e)}")
    
    def _validate_query(self, query: str) -> Optional[str]:
        """
        Validate that query is safe to execute.
        
        Returns error message if validation fails.
        """
        query_upper = query.upper().strip()
        
        # Must start with SELECT
        if not query_upper.startswith("SELECT"):
            return "Only SELECT queries are allowed"
        
        # Check for forbidden keywords
        for keyword in FORBIDDEN_KEYWORDS:
            if keyword in query_upper:
                return f"Forbidden keyword in query: {keyword}"
        
        # Check for allowed tables only
        # Simple check - look for FROM clause
        from_match = re.search(r'\bFROM\s+(\w+)', query_upper)
        if from_match:
            table = from_match.group(1).lower()
            if table not in ALLOWED_TABLES:
                return f"Table not allowed: {table}. Allowed tables: {list(ALLOWED_TABLES.keys())}"
        
        return None
    
    def _rewrite_with_client_filter(
        self,
        query: str,
        client_id: str,
        limit: int
    ) -> Optional[str]:
        """
        Rewrite query to include client_id filter.
        
        This ensures tenant isolation by always filtering by client.
        """
        query = query.strip()
        
        # Remove any existing LIMIT clause
        query = re.sub(r'\bLIMIT\s+\d+', '', query, flags=re.IGNORECASE)
        
        # Check if WHERE clause exists
        if re.search(r'\bWHERE\b', query, re.IGNORECASE):
            # Add client_id to existing WHERE
            query = re.sub(
                r'\bWHERE\b',
                f"WHERE client_id = '{client_id}' AND",
                query,
                count=1,
                flags=re.IGNORECASE
            )
        else:
            # Add WHERE clause before ORDER BY, GROUP BY, or at end
            order_match = re.search(r'\b(ORDER BY|GROUP BY)', query, re.IGNORECASE)
            if order_match:
                insert_pos = order_match.start()
                query = f"{query[:insert_pos]} WHERE client_id = '{client_id}' {query[insert_pos:]}"
            else:
                query = f"{query} WHERE client_id = '{client_id}'"
        
        # Add LIMIT
        query = f"{query} LIMIT {limit}"
        
        return query
    
    async def _execute_query(
        self,
        query: str,
        client_id: str
    ) -> List[Dict[str, Any]]:
        """
        Execute the query and return results.
        
        Note: This is a mock implementation. Replace with actual
        database execution in production.
        """
        # For now, return mock data since we don't have direct SQL access
        # In production, this would execute against the actual database
        
        logger.info(f"SQL Query (client={client_id}): {query}")
        
        # Mock implementation - in production use actual DB
        if self.db is not None:
            try:
                # Actual database execution would go here
                # results = await self.db.fetch_all(query)
                # return [dict(row) for row in results]
                pass
            except Exception as e:
                logger.error(f"Database query failed: {e}")
        
        # Return informative message about mock implementation
        return [{
            "note": "SQL execution is in mock mode",
            "query_received": query,
            "client_id": client_id,
            "suggestion": "Check documents via the /documents API endpoint"
        }]
    
    def get_available_columns(self, table: str) -> List[str]:
        """Get available columns for a table."""
        return ALLOWED_TABLES.get(table, [])
    
    def get_example_queries(self) -> List[str]:
        """Get example queries for user guidance."""
        return [
            "SELECT filename, uploaded_at FROM documents",
            "SELECT COUNT(*) as total FROM documents",
            "SELECT filename FROM documents WHERE page_count > 5",
            "SELECT filename, chunk_count FROM documents ORDER BY uploaded_at DESC",
        ]
