#!/usr/bin/env python3
"""
Debug script to trace the exact source of .lower() errors in NWTN system.
This will help identify where the errors are happening at runtime.
"""

import sys
import traceback
import re
from typing import Any

# Store the original lower method
original_lower = str.lower

def traced_lower(self):
    """Traced version of str.lower() that catches errors"""
    try:
        return original_lower(self)
    except AttributeError as e:
        print(f"\n🚨 FOUND .lower() ERROR!")
        print(f"Object type: {type(self)}")
        print(f"Object value: {repr(self)}")
        print(f"Stack trace:")
        traceback.print_stack()
        print("=" * 80)
        # Return a safe fallback
        return str(self).lower()

# Monkey patch str.lower to trace calls
str.lower = traced_lower

# Also patch any potential AttributeError for list objects
original_getattribute = list.__getattribute__

def traced_list_getattribute(self, name):
    """Traced version of list.__getattribute__ to catch .lower() calls"""
    if name == 'lower':
        print(f"\n🚨 FOUND list.lower() CALL!")
        print(f"List contents: {repr(self)}")
        print(f"Stack trace:")
        traceback.print_stack()
        print("=" * 80)
        # Return a safe function that converts to string first
        return lambda: str(self).lower()
    return original_getattribute(self, name)

# Monkey patch list.__getattribute__ to catch .lower() calls
list.__getattribute__ = traced_list_getattribute

print("🔍 Debug tracing enabled for .lower() calls")
print("Now run your NWTN test to see exactly where the errors occur...")