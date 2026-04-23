"""
Data Processing Agent
=====================

Executes instruction manifests against real data (CSV, JSON, Parquet).
This is the pragmatic MVP: Python-based execution within the node process,
replacing the template WASM binary for real data analysis.

When the WASM SDK is built, this logic will be compiled into WASM modules.
For now, it enables real queries against real datasets.
"""

import csv
import io
import json
import logging
from typing import Any, Dict, List

from prsm.compute.agents.instruction_set import AgentOp, InstructionManifest

logger = logging.getLogger(__name__)


class DataProcessor:
    """Executes instruction manifests against tabular data."""

    def execute(self, manifest: InstructionManifest, data: bytes) -> Dict[str, Any]:
        """Execute an instruction manifest against raw data.

        Args:
            manifest: The instruction pipeline to execute.
            data: Raw bytes (CSV, JSON, or JSON Lines).

        Returns:
            Dict with status, output records, and metadata.
        """
        try:
            records = self._parse_data(data)
            if not records:
                return {"status": "error", "error": "No records parsed from data", "count": 0}

            # Execute instruction pipeline
            result_records = records
            metadata = {"initial_count": len(records), "operations": []}

            for instruction in manifest.instructions:
                result_records, op_meta = self._execute_op(instruction, result_records)
                metadata["operations"].append(op_meta)

            # Limit output
            output = result_records[:manifest.max_output_records]

            return {
                "status": "success",
                "records": output,
                "count": len(output),
                "total_matched": len(result_records),
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            return {"status": "error", "error": str(e), "count": 0}

    def _parse_data(self, data: bytes) -> List[Dict[str, Any]]:
        """Parse raw bytes into a list of record dicts."""
        text = data.decode("utf-8", errors="replace").strip()
        if not text:
            return []

        # Try JSON array
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

        # Try JSON Lines
        lines = text.split("\n")
        if lines and lines[0].strip().startswith("{"):
            records = []
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            if records:
                return records

        # Try CSV
        try:
            reader = csv.DictReader(io.StringIO(text))
            return list(reader)
        except Exception:
            pass

        return []

    def _execute_op(
        self,
        instruction,
        records: List[Dict[str, Any]],
    ) -> tuple:
        """Execute a single instruction against records."""
        op = instruction.op
        field = instruction.field
        value = instruction.value
        params = instruction.params

        if op == AgentOp.FILTER:
            filtered = [r for r in records if self._match(r, field, value, params)]
            return filtered, {"op": "filter", "field": field, "matched": len(filtered)}

        elif op == AgentOp.COUNT:
            return [{"count": len(records)}], {"op": "count", "result": len(records)}

        elif op == AgentOp.SUM:
            total = sum(self._to_num(r.get(field, 0)) for r in records)
            return [{"sum": total, "field": field}], {"op": "sum", "field": field, "result": total}

        elif op == AgentOp.AVERAGE:
            vals = [self._to_num(r.get(field, 0)) for r in records]
            avg = sum(vals) / len(vals) if vals else 0
            return [{"average": avg, "field": field, "count": len(vals)}], {"op": "average", "result": avg}

        elif op == AgentOp.GROUP_BY:
            groups = {}
            for r in records:
                key = str(r.get(field, "unknown"))
                groups.setdefault(key, []).append(r)
            result = [{"group": k, "count": len(v)} for k, v in sorted(groups.items())]
            return result, {"op": "group_by", "field": field, "groups": len(groups)}

        elif op == AgentOp.SORT:
            ascending = params.get("ascending", True)
            try:
                sorted_records = sorted(records, key=lambda r: r.get(field, ""), reverse=not ascending)
            except TypeError:
                sorted_records = records
            return sorted_records, {"op": "sort", "field": field}

        elif op == AgentOp.LIMIT:
            n = int(value or params.get("n", 10))
            return records[:n], {"op": "limit", "n": n}

        elif op == AgentOp.SELECT:
            fields = params.get("fields", [field]) if field else params.get("fields", [])
            if fields:
                selected = [{f: r.get(f) for f in fields} for r in records]
            else:
                selected = records
            return selected, {"op": "select", "fields": fields}

        elif op == AgentOp.AGGREGATE:
            method = params.get("method", "count")
            if method == "count":
                return [{"count": len(records)}], {"op": "aggregate", "method": "count"}
            elif method == "sum" and field:
                total = sum(self._to_num(r.get(field, 0)) for r in records)
                return [{"sum": total}], {"op": "aggregate", "method": "sum"}
            return [{"count": len(records)}], {"op": "aggregate", "method": method}

        elif op == AgentOp.COMPARE:
            # Compare groups by a metric
            return records, {"op": "compare", "note": "passthrough"}

        elif op == AgentOp.TIME_SERIES:
            date_field = params.get("date_field", field)
            metric_field = params.get("metric_field", "count")
            groups = {}
            for r in records:
                period = str(r.get(date_field, "unknown"))
                groups.setdefault(period, []).append(r)
            result = [
                {"period": k, "count": len(v), "metric": sum(self._to_num(r.get(metric_field, 0)) for r in v)}
                for k, v in sorted(groups.items())
            ]
            return result, {"op": "time_series", "periods": len(groups)}

        else:
            return records, {"op": str(op), "note": "unhandled, passthrough"}

    @staticmethod
    def _match(record: Dict, field: str, value: Any, params: Dict) -> bool:
        """Check if a record matches a filter condition."""
        if not field:
            return True
        record_val = record.get(field)
        if record_val is None:
            return value is None
        operator = params.get("operator", "eq")
        if operator == "eq" or not operator:
            return str(record_val).lower() == str(value).lower()
        elif operator == "ne":
            return str(record_val).lower() != str(value).lower()
        elif operator == "gt":
            return DataProcessor._to_num(record_val) > DataProcessor._to_num(value)
        elif operator == "lt":
            return DataProcessor._to_num(record_val) < DataProcessor._to_num(value)
        elif operator == "contains":
            return str(value).lower() in str(record_val).lower()
        return str(record_val).lower() == str(value).lower()

    @staticmethod
    def _to_num(val: Any) -> float:
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0
