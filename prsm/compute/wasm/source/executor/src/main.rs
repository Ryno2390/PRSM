//! PRSM WASM Executor
//! ===================
//!
//! Interprets an `InstructionManifest` against tabular data inside
//! a Wasmtime sandbox. Mirrors the Python `DataProcessor` reference
//! at `prsm/compute/agents/data_processor.py`.
//!
//! Wire contract (WASI command module):
//!   - stdin:  JSON blob `{"manifest": <InstructionManifest>,
//!                          "data": "<csv|json|jsonl text>"}`
//!   - stdout: JSON blob `{"status": "success"|"error",
//!                          "records": [...],
//!                          "count": N,
//!                          "metadata": {...},
//!                          "error": "..."}`
//!
//! Supported ops (mirror DataProcessor):
//!   COUNT, SUM, AVERAGE, FILTER, LIMIT, SELECT, SORT, GROUP_BY,
//!   AGGREGATE, COMPARE, TIME_SERIES
//!
//! Input data formats (auto-detected, in order):
//!   - JSON array
//!   - JSON object (wrapped to single-element array)
//!   - JSON Lines (one object per line)
//!   - CSV with header row
//!
//! Failures: any parse / IO error writes `{"status": "error",
//! "error": "...", "count": 0}` to stdout and exits 0. The
//! aggregator-side verification rejects this against the digest;
//! the orchestrator's retry-loop handles the rejection.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{self, Read, Write};

// ────────────────────────────────────────────────────────────────────
// Wire types — match prsm/compute/agents/instruction_set.py
// ────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct AgentInstruction {
    op: String,
    #[serde(default)]
    field: String,
    #[serde(default)]
    value: Value,
    #[serde(default)]
    params: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
struct InstructionManifest {
    #[serde(default)]
    query: String,
    #[serde(default)]
    instructions: Vec<AgentInstruction>,
    #[serde(default = "default_output_format")]
    output_format: String,
    #[serde(default = "default_max_output")]
    max_output_records: usize,
}

fn default_output_format() -> String { "json".to_string() }
fn default_max_output() -> usize { 1000 }

#[derive(Debug, Deserialize)]
struct ExecutorInput {
    manifest: InstructionManifest,
    #[serde(default)]
    data: String,
}

#[derive(Debug, Serialize)]
struct ExecutorOutput {
    status: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    records: Vec<Value>,
    count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_matched: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

// ────────────────────────────────────────────────────────────────────
// Data parsing — JSON array / object / JSONL / CSV
// ────────────────────────────────────────────────────────────────────

fn parse_data(text: &str) -> Vec<Value> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    // JSON array.
    if let Ok(parsed) = serde_json::from_str::<Value>(text) {
        if let Some(arr) = parsed.as_array() {
            return arr.clone();
        }
        if parsed.is_object() {
            return vec![parsed];
        }
    }
    // JSON Lines.
    if text.starts_with('{') {
        let mut records = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(rec) = serde_json::from_str::<Value>(line) {
                records.push(rec);
            }
        }
        if !records.is_empty() {
            return records;
        }
    }
    // CSV.
    parse_csv(text)
}

fn parse_csv(text: &str) -> Vec<Value> {
    let mut lines = text.lines();
    let header_line = match lines.next() {
        Some(l) => l,
        None => return Vec::new(),
    };
    let headers: Vec<&str> = header_line.split(',').map(str::trim).collect();
    let mut records = Vec::new();
    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let mut obj = serde_json::Map::new();
        for (i, h) in headers.iter().enumerate() {
            let v = fields.get(i).copied().unwrap_or("");
            obj.insert(h.to_string(), Value::String(v.to_string()));
        }
        records.push(Value::Object(obj));
    }
    records
}

// ────────────────────────────────────────────────────────────────────
// Op interpreters — mirror DataProcessor._execute_op
// ────────────────────────────────────────────────────────────────────

fn to_num(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        Value::String(s) => s.parse::<f64>().unwrap_or(0.0),
        Value::Bool(b) => if *b { 1.0 } else { 0.0 },
        _ => 0.0,
    }
}

fn match_field(record: &Value, field: &str, value: &Value, params: &BTreeMap<String, Value>) -> bool {
    if field.is_empty() {
        return true;
    }
    let record_val = record.get(field);
    let record_val = match record_val {
        Some(v) => v,
        None => return value.is_null(),
    };
    let operator = params.get("operator").and_then(Value::as_str).unwrap_or("eq");
    match operator {
        "eq" | "" => stringify_lower(record_val) == stringify_lower(value),
        "ne" => stringify_lower(record_val) != stringify_lower(value),
        "gt" => to_num(record_val) > to_num(value),
        "lt" => to_num(record_val) < to_num(value),
        "contains" => stringify_lower(record_val).contains(&stringify_lower(value)),
        _ => stringify_lower(record_val) == stringify_lower(value),
    }
}

fn stringify_lower(v: &Value) -> String {
    match v {
        Value::String(s) => s.to_lowercase(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "".to_string(),
        _ => v.to_string().to_lowercase(),
    }
}

fn execute_op(instr: &AgentInstruction, records: Vec<Value>) -> (Vec<Value>, Value) {
    let field = instr.field.as_str();
    let value = &instr.value;
    let params = &instr.params;
    match instr.op.as_str() {
        "filter" => {
            let filtered: Vec<Value> = records.iter()
                .filter(|r| match_field(r, field, value, params))
                .cloned()
                .collect();
            let matched = filtered.len();
            (filtered, json!({"op": "filter", "field": field, "matched": matched}))
        }
        "count" => {
            let n = records.len();
            (vec![json!({"count": n})], json!({"op": "count", "result": n}))
        }
        "sum" => {
            let total: f64 = records.iter().map(|r| to_num(r.get(field).unwrap_or(&Value::Null))).sum();
            (vec![json!({"sum": total, "field": field})],
             json!({"op": "sum", "field": field, "result": total}))
        }
        "average" => {
            let vals: Vec<f64> = records.iter().map(|r| to_num(r.get(field).unwrap_or(&Value::Null))).collect();
            let avg = if vals.is_empty() { 0.0 } else { vals.iter().sum::<f64>() / vals.len() as f64 };
            let count = vals.len();
            (vec![json!({"average": avg, "field": field, "count": count})],
             json!({"op": "average", "result": avg}))
        }
        "group_by" => {
            let mut groups: BTreeMap<String, Vec<Value>> = BTreeMap::new();
            for r in &records {
                let key = r.get(field).map(stringify_lower).unwrap_or_else(|| "unknown".to_string());
                groups.entry(key).or_default().push(r.clone());
            }
            let result: Vec<Value> = groups.iter()
                .map(|(k, v)| json!({"group": k, "count": v.len()}))
                .collect();
            let gcount = groups.len();
            (result, json!({"op": "group_by", "field": field, "groups": gcount}))
        }
        "sort" => {
            let ascending = params.get("ascending").and_then(Value::as_bool).unwrap_or(true);
            let mut sorted = records.clone();
            sorted.sort_by(|a, b| {
                let av = a.get(field).map(stringify_lower).unwrap_or_default();
                let bv = b.get(field).map(stringify_lower).unwrap_or_default();
                if ascending { av.cmp(&bv) } else { bv.cmp(&av) }
            });
            (sorted, json!({"op": "sort", "field": field}))
        }
        "limit" => {
            let n = value.as_i64().or_else(|| params.get("n").and_then(Value::as_i64)).unwrap_or(10);
            let n = n.max(0) as usize;
            let taken: Vec<Value> = records.into_iter().take(n).collect();
            (taken, json!({"op": "limit", "n": n}))
        }
        "select" => {
            let fields: Vec<String> = if let Some(arr) = params.get("fields").and_then(Value::as_array) {
                arr.iter().filter_map(Value::as_str).map(str::to_string).collect()
            } else if !field.is_empty() {
                vec![field.to_string()]
            } else {
                Vec::new()
            };
            let selected: Vec<Value> = if fields.is_empty() {
                records
            } else {
                records.iter().map(|r| {
                    let mut obj = serde_json::Map::new();
                    for f in &fields {
                        obj.insert(f.clone(), r.get(f).cloned().unwrap_or(Value::Null));
                    }
                    Value::Object(obj)
                }).collect()
            };
            (selected, json!({"op": "select", "fields": fields}))
        }
        "aggregate" => {
            let method = params.get("method").and_then(Value::as_str).unwrap_or("count");
            match method {
                "sum" if !field.is_empty() => {
                    let total: f64 = records.iter().map(|r| to_num(r.get(field).unwrap_or(&Value::Null))).sum();
                    (vec![json!({"sum": total})], json!({"op": "aggregate", "method": "sum"}))
                }
                _ => {
                    let n = records.len();
                    (vec![json!({"count": n})], json!({"op": "aggregate", "method": method}))
                }
            }
        }
        "compare" => (records, json!({"op": "compare", "note": "passthrough"})),
        "time_series" => {
            let date_field = params.get("date_field").and_then(Value::as_str).unwrap_or(field).to_string();
            let metric_field = params.get("metric_field").and_then(Value::as_str).unwrap_or("count").to_string();
            let mut groups: BTreeMap<String, Vec<Value>> = BTreeMap::new();
            for r in &records {
                let period = r.get(&date_field).map(stringify_lower).unwrap_or_else(|| "unknown".to_string());
                groups.entry(period).or_default().push(r.clone());
            }
            let result: Vec<Value> = groups.iter().map(|(k, v)| {
                let metric: f64 = v.iter().map(|r| to_num(r.get(&metric_field).unwrap_or(&Value::Null))).sum();
                json!({"period": k, "count": v.len(), "metric": metric})
            }).collect();
            let p = groups.len();
            (result, json!({"op": "time_series", "periods": p}))
        }
        _ => (records, json!({"op": instr.op.as_str(), "note": "unhandled, passthrough"})),
    }
}

// ────────────────────────────────────────────────────────────────────
// Pipeline
// ────────────────────────────────────────────────────────────────────

fn run(input: ExecutorInput) -> ExecutorOutput {
    let records = parse_data(&input.data);
    if records.is_empty() {
        return ExecutorOutput {
            status: "error".into(),
            records: Vec::new(),
            count: 0,
            total_matched: None,
            metadata: None,
            error: Some("No records parsed from data".into()),
        };
    }
    let initial_count = records.len();
    let mut current = records;
    let mut op_meta: Vec<Value> = Vec::new();
    for instr in &input.manifest.instructions {
        let (next, meta) = execute_op(instr, current);
        current = next;
        op_meta.push(meta);
    }
    let total_matched = current.len();
    let limit = input.manifest.max_output_records;
    current.truncate(limit);
    let output_len = current.len();
    ExecutorOutput {
        status: "success".into(),
        records: current,
        count: output_len,
        total_matched: Some(total_matched),
        metadata: Some(json!({
            "initial_count": initial_count,
            "operations": op_meta,
        })),
        error: None,
    }
}

fn main() {
    let mut buf = String::new();
    let read_result = io::stdin().read_to_string(&mut buf);
    let output = match read_result {
        Err(e) => ExecutorOutput {
            status: "error".into(),
            records: Vec::new(),
            count: 0,
            total_matched: None,
            metadata: None,
            error: Some(format!("stdin read failed: {}", e)),
        },
        Ok(_) => {
            match serde_json::from_str::<ExecutorInput>(&buf) {
                Ok(input) => run(input),
                Err(e) => ExecutorOutput {
                    status: "error".into(),
                    records: Vec::new(),
                    count: 0,
                    total_matched: None,
                    metadata: None,
                    error: Some(format!("input parse failed: {}", e)),
                },
            }
        }
    };
    let out_text = serde_json::to_string(&output).unwrap_or_else(|_|
        r#"{"status":"error","count":0,"error":"serialization failed"}"#.to_string()
    );
    let _ = io::stdout().write_all(out_text.as_bytes());
}
