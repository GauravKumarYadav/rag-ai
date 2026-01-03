"""Processor Registry for document type handling.

This module provides a central registry for document processors,
allowing dynamic registration and lookup by file type.
"""

from typing import Dict, List, Optional, Type
from app.processors.base import DocumentProcessor, ProcessorConfig


class ProcessorRegistry:
    """Central registry for document processors.
    
    Processors can be registered by extension or mimetype, and the registry
    will return the appropriate processor for a given file.
    
    Usage:
        # Register a processor
        ProcessorRegistry.register(PDFProcessor)
        
        # Get processor for a file
        processor = ProcessorRegistry.get_processor("document.pdf")
        text = processor.extract(content, "document.pdf")
    """
    
    _processors: Dict[str, Type[DocumentProcessor]] = {}
    _mimetype_map: Dict[str, Type[DocumentProcessor]] = {}
    _default_processor: Optional[Type[DocumentProcessor]] = None
    
    @classmethod
    def register(cls, processor_class: Type[DocumentProcessor]) -> Type[DocumentProcessor]:
        """Register a processor class for its supported extensions.
        
        Can be used as a decorator:
            @ProcessorRegistry.register
            class MyProcessor(DocumentProcessor):
                ...
        
        Args:
            processor_class: The processor class to register
            
        Returns:
            The same processor class (for decorator usage)
        """
        # Create a temporary instance to get extensions
        instance = processor_class.__new__(processor_class)
        
        # Register by extension
        for ext in getattr(processor_class, 'supported_extensions', []):
            cls._processors[ext.lower()] = processor_class
        
        # Register by mimetype
        for mimetype in getattr(processor_class, 'supported_mimetypes', []):
            cls._mimetype_map[mimetype] = processor_class
        
        return processor_class
    
    @classmethod
    def set_default(cls, processor_class: Type[DocumentProcessor]) -> None:
        """Set the default processor for unknown file types.
        
        Args:
            processor_class: The processor class to use as default
        """
        cls._default_processor = processor_class
    
    @classmethod
    def get_processor(
        cls,
        filename: str,
        mimetype: Optional[str] = None,
        config: Optional[ProcessorConfig] = None,
    ) -> DocumentProcessor:
        """Get the appropriate processor for a file.
        
        Args:
            filename: Name of the file to process
            mimetype: Optional MIME type of the file
            config: Optional processor configuration
            
        Returns:
            An instance of the appropriate processor
            
        Raises:
            ValueError: If no processor is found and no default is set
        """
        config = config or ProcessorConfig()
        
        # Try by extension first
        ext = cls._get_extension(filename)
        if ext in cls._processors:
            return cls._processors[ext](config)
        
        # Try by mimetype
        if mimetype and mimetype in cls._mimetype_map:
            return cls._mimetype_map[mimetype](config)
        
        # Use default if available
        if cls._default_processor:
            return cls._default_processor(config)
        
        raise ValueError(
            f"No processor found for file '{filename}' "
            f"(extension: {ext}, mimetype: {mimetype}). "
            f"Registered extensions: {list(cls._processors.keys())}"
        )
    
    @classmethod
    def can_process(cls, filename: str, mimetype: Optional[str] = None) -> bool:
        """Check if the registry has a processor for the given file.
        
        Args:
            filename: Name of the file to check
            mimetype: Optional MIME type of the file
            
        Returns:
            True if a processor is available
        """
        ext = cls._get_extension(filename)
        
        if ext in cls._processors:
            return True
        
        if mimetype and mimetype in cls._mimetype_map:
            return True
        
        return cls._default_processor is not None
    
    @classmethod
    def list_processors(cls) -> List[str]:
        """List all registered processor extensions.
        
        Returns:
            List of registered extensions
        """
        return list(cls._processors.keys())
    
    @classmethod
    def list_mimetypes(cls) -> List[str]:
        """List all registered MIME types.
        
        Returns:
            List of registered MIME types
        """
        return list(cls._mimetype_map.keys())
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered processors (mainly for testing)."""
        cls._processors.clear()
        cls._mimetype_map.clear()
        cls._default_processor = None
    
    @staticmethod
    def _get_extension(filename: str) -> str:
        """Get the lowercase file extension including the dot."""
        if "." in filename:
            return "." + filename.rsplit(".", 1)[-1].lower()
        return ""
