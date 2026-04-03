"""Shared fixtures and import helpers for PRSM CLI tests.

The prsm/__init__.py imports prsm.core.config which requires pydantic.
We stub the prsm top-level module to prevent that cascading import failure,
then import submodules directly.
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Pydantic stub — config_schema.py does `from pydantic import BaseModel, Field`
# We provide a minimal shim so the module can load without real pydantic.
# ---------------------------------------------------------------------------
_pydantic_available = False
try:
    import pydantic as _real_pydantic  # noqa: F401
    _pydantic_available = True
except ImportError:
    pass

if not _pydantic_available:
    # Build a fake pydantic module with BaseModel and Field
    _pydantic = types.ModuleType("pydantic")

    def _field_shim(default=..., *, ge=None, le=None, default_factory=None, **kw):
        """Minimal Field() that just stores metadata."""
        class _FieldDescriptor:
            def __init__(self):
                self.default = default
                self.default_factory = default_factory
                self.ge = ge
                self.le = le
            def __set_name__(self, owner, name):
                self._name = name
            def __get__(self, obj, objtype=None):
                if obj is None:
                    return self
                return obj.__dict__.get(self._name, self.default if self.default is not ... else (self.default_factory() if self.default_factory else None))
            def __set__(self, obj, value):
                obj.__dict__[self._name] = value
        return _FieldDescriptor()

    class _BaseModelMeta(type):
        """Metaclass that processes Field descriptors on class creation."""
        def __new__(mcs, name, bases, namespace):
            cls = super().__new__(mcs, name, bases, namespace)
            # Collect annotations for __init__
            annotations = {}
            for base in reversed(cls.__mro__):
                annotations.update(getattr(base, '__annotations__', {}))
            cls.__all_annotations__ = annotations
            return cls

    class _BaseModel(metaclass=_BaseModelMeta):
        """Minimal BaseModel shim — supports __init__ with kwargs, model_dump, validation-ish."""

        def __init__(self, **kwargs):
            # Set defaults from class-level Field descriptors and plain defaults
            for attr_name in self.__all_annotations__:
                if attr_name in kwargs:
                    setattr(self, attr_name, kwargs[attr_name])
                else:
                    class_val = getattr(type(self), attr_name, ...)
                    if hasattr(class_val, 'default'):
                        if class_val.default is not ...:
                            setattr(self, attr_name, class_val.default)
                        elif class_val.default_factory:
                            setattr(self, attr_name, class_val.default_factory())
                    elif class_val is not ...:
                        setattr(self, attr_name, class_val)

        def model_dump(self, **kwargs):
            import enum as _enum
            json_mode = kwargs.get('mode') == 'json'
            result = {}
            for attr_name in self.__all_annotations__:
                val = getattr(self, attr_name, None)
                if json_mode and isinstance(val, _enum.Enum):
                    val = val.value
                result[attr_name] = val
            return result

        @classmethod
        def model_validate(cls, data):
            return cls(**data)

    _pydantic.BaseModel = _BaseModel
    _pydantic.Field = _field_shim
    sys.modules["pydantic"] = _pydantic


# ---------------------------------------------------------------------------
# Stub the prsm top-level package so submodule imports work without
# triggering prsm/__init__.py (which does `from prsm.core.config import ...`)
# ---------------------------------------------------------------------------
def _ensure_prsm_stub():
    """Ensure prsm package is importable without its heavy __init__.py."""
    if "prsm" not in sys.modules or hasattr(sys.modules["prsm"], "_cli_test_stub"):
        prsm_root = Path(__file__).resolve().parent.parent.parent / "prsm"
        # Create stub for prsm
        mod = types.ModuleType("prsm")
        mod.__path__ = [str(prsm_root)]
        mod.__file__ = str(prsm_root / "__init__.py")
        mod._cli_test_stub = True
        mod.__version__ = "0.22.0"
        sys.modules["prsm"] = mod

        # Also stub prsm.core and prsm.core.config to avoid transitive imports
        for sub in ["prsm.core", "prsm.core.config", "prsm.core.models",
                     "prsm.compute", "prsm.compute.nwtn"]:
            if sub not in sys.modules:
                stub = types.ModuleType(sub)
                if "." in sub:
                    stub.__path__ = [str(prsm_root / sub.replace("prsm.", "").replace(".", "/"))]
                sys.modules[sub] = stub


_ensure_prsm_stub()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_console(monkeypatch):
    """Replace the ui module's console.print with a capturing list."""
    from prsm.cli_modules import ui
    captured = []
    monkeypatch.setattr(ui.console, "print", lambda *a, **kw: captured.append((a, kw)))
    return captured


@pytest.fixture
def registry():
    """A fresh SkillRegistry with builtins loaded."""
    from prsm.skills.registry import SkillRegistry
    return SkillRegistry()
