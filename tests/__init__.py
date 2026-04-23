"""PRSM test suite root package.

Empty package marker. Exists so subpackages like ``tests.chaos`` can be
imported under their fully-qualified name (``from tests.chaos.harness
import ...``) from test modules that need to cross-reference shared
fixtures or harness code.
"""
