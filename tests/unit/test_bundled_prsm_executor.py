"""Sprint 177 — bundled real WASM executor end-to-end tests.

Sprint 174 shipped a 36-byte stub (exports `run` → i32(42)) so QO
wiring would construct. Sprint 177 ships the real instruction
interpreter at ~188 KB — written in Rust, compiled to
wasm32-wasip1, interprets all 11 AgentOps mirroring the Python
``DataProcessor`` reference.

Wire contract:
  stdin:  {"manifest": {...}, "data": "<csv|json|jsonl>"}
  stdout: {"status": ..., "records": [...], "count": N, "metadata": ...}

These tests run the actual bundled binary through wasmtime — they
prove the binary in the wheel actually does what its docstring
claims, not just that it loads.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest


pytest.importorskip("wasmtime")


def _run(manifest_instructions, data_str, max_output_records=1000):
    """Execute the bundled WASM executor with the given manifest +
    data. Returns the parsed stdout JSON."""
    import wasmtime
    from prsm.compute.wasm.binaries import PRSM_EXECUTOR_PATH

    engine = wasmtime.Engine()
    module = wasmtime.Module.from_file(engine, str(PRSM_EXECUTOR_PATH))

    inp = {
        "manifest": {
            "instructions": manifest_instructions,
            "max_output_records": max_output_records,
        },
        "data": data_str,
    }
    tmpdir = tempfile.mkdtemp()
    sin_path = os.path.join(tmpdir, "in.json")
    sout_path = os.path.join(tmpdir, "out.json")
    with open(sin_path, "w") as f:
        json.dump(inp, f)

    store = wasmtime.Store(engine)
    config = wasmtime.WasiConfig()
    config.stdin_file = sin_path
    config.stdout_file = sout_path
    store.set_wasi(config)
    linker = wasmtime.Linker(engine)
    linker.define_wasi()
    inst = linker.instantiate(store, module)
    inst.exports(store)["_start"](store)

    with open(sout_path) as f:
        return json.load(f)


# Sample JSON-array data used by most tests.
SAMPLE = json.dumps([
    {"name": "alice", "age": 30, "city": "NY"},
    {"name": "bob", "age": 25, "city": "LA"},
    {"name": "alice", "age": 35, "city": "NY"},
])


class TestBundledExecutorOps:
    def test_count(self):
        out = _run([{"op": "count"}], SAMPLE)
        assert out["status"] == "success"
        assert out["records"] == [{"count": 3}]

    def test_sum(self):
        out = _run([{"op": "sum", "field": "age"}], SAMPLE)
        assert out["status"] == "success"
        assert out["records"][0]["sum"] == 90.0

    def test_average(self):
        out = _run([{"op": "average", "field": "age"}], SAMPLE)
        assert out["records"][0]["average"] == 30.0
        assert out["records"][0]["count"] == 3

    def test_filter_eq(self):
        out = _run(
            [{"op": "filter", "field": "name", "value": "alice"}],
            SAMPLE,
        )
        assert out["count"] == 2

    def test_filter_then_count(self):
        """Pipeline of two ops — the executor must thread state
        between instructions."""
        out = _run(
            [
                {"op": "filter", "field": "name", "value": "alice"},
                {"op": "count"},
            ],
            SAMPLE,
        )
        assert out["records"] == [{"count": 2}]

    def test_group_by(self):
        out = _run([{"op": "group_by", "field": "city"}], SAMPLE)
        groups = {r["group"]: r["count"] for r in out["records"]}
        assert groups == {"ny": 2, "la": 1}

    def test_sort_desc(self):
        out = _run(
            [{"op": "sort", "field": "age",
              "params": {"ascending": False}}],
            SAMPLE,
        )
        ages = [r["age"] for r in out["records"]]
        assert ages == [35, 30, 25]

    def test_limit(self):
        out = _run([{"op": "limit", "value": 2}], SAMPLE)
        assert out["count"] == 2

    def test_select(self):
        out = _run(
            [{"op": "select", "params": {"fields": ["name", "age"]}}],
            SAMPLE,
        )
        assert "city" not in out["records"][0]
        assert set(out["records"][0].keys()) == {"name", "age"}


class TestBundledExecutorDataFormats:
    def test_csv_input(self):
        out = _run([{"op": "count"}], "name,age\nalice,30\nbob,25")
        assert out["records"] == [{"count": 2}]

    def test_jsonl_input(self):
        data = '{"id":1}\n{"id":2}\n{"id":3}'
        out = _run([{"op": "count"}], data)
        assert out["records"] == [{"count": 3}]

    def test_empty_data_returns_error(self):
        """No records to interpret → status=error with a useful
        message. Aggregator-side digest check then rejects."""
        out = _run([{"op": "count"}], "")
        assert out["status"] == "error"
        assert "no records" in out["error"].lower()


class TestBundledExecutorRobustness:
    def test_malformed_input_returns_error(self):
        """If the input is not valid JSON, the executor reports an
        error rather than panicking — exit code stays 0 so the
        WASM runtime doesn't fault."""
        import wasmtime
        from prsm.compute.wasm.binaries import PRSM_EXECUTOR_PATH

        engine = wasmtime.Engine()
        module = wasmtime.Module.from_file(engine, str(PRSM_EXECUTOR_PATH))

        tmpdir = tempfile.mkdtemp()
        sin = os.path.join(tmpdir, "in.json")
        sout = os.path.join(tmpdir, "out.json")
        with open(sin, "w") as f:
            f.write("not valid json at all")
        store = wasmtime.Store(engine)
        config = wasmtime.WasiConfig()
        config.stdin_file = sin
        config.stdout_file = sout
        store.set_wasi(config)
        linker = wasmtime.Linker(engine)
        linker.define_wasi()
        inst = linker.instantiate(store, module)
        inst.exports(store)["_start"](store)
        with open(sout) as f:
            out = json.load(f)
        assert out["status"] == "error"
        assert "parse" in out["error"].lower()

    def test_max_output_records_enforced(self):
        """Output truncated to manifest.max_output_records."""
        many = json.dumps([{"id": i} for i in range(50)])
        out = _run([{"op": "limit", "value": 100}], many, max_output_records=10)
        assert out["count"] == 10
        # total_matched preserves pre-truncation count
        assert out["total_matched"] == 50


def test_executor_size_reasonable():
    """Sprint 177 invariant — the real executor is reasonably small
    so the wheel doesn't bloat (`opt-level=z` + lto + strip).
    Expected ~188 KB; alarm if it grows past 1 MB which would
    suggest accidental debug-build commit."""
    from prsm.compute.wasm.binaries import PRSM_EXECUTOR_PATH
    size = PRSM_EXECUTOR_PATH.stat().st_size
    assert 50_000 < size < 1_000_000, (
        f"prsm_executor.wasm size {size} bytes outside expected "
        f"range [50KB, 1MB]. Verify the release build configuration."
    )
