"""Leaf constants shared across abstract plugin components.

This module imports nothing from mloda so that both the feature chain parser and
the (lighter) divergence module can import from it without pulling in the parser's
dependency graph.
"""

ANY_OUTPUT_PART = "*"  # In OUTPUT_PARTS: parts are named at run time
