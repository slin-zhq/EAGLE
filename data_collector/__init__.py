"""
Data Collector Package for EAGLE-3

Provides different collection modes for instrumentation during inference.
"""

from .simple_dump_collector import SimpleDumpCollector

__all__ = ['SimpleDumpCollector']
