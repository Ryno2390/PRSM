#!/usr/bin/env python3
"""
Advanced Data Transformation Engine
===================================

Sophisticated data transformation system supporting complex rules, schema mapping,
data quality validation, and real-time transformations.
"""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, Pattern
import uuid
import math

from prsm.compute.plugins import require_optional, has_optional_dependency

logger = logging.getLogger(__name__)


class TransformationType(Enum):
    """Types of data transformations"""
    FIELD_MAPPING = "field_mapping"
    DATA_TYPE_CONVERSION = "data_type_conversion"
    VALUE_TRANSFORMATION = "value_transformation"
    CONDITIONAL_LOGIC = "conditional_logic"
    AGGREGATION = "aggregation"
    LOOKUP = "lookup"
    VALIDATION = "validation"
    ENRICHMENT = "enrichment"
    NORMALIZATION = "normalization"
    CUSTOM_FUNCTION = "custom_function"


class ValidationLevel(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class TransformationRule:
    """Configuration for individual transformation rules"""
    rule_id: str
    name: str
    transformation_type: TransformationType
    description: str = ""
    
    # Rule configuration
    source_field: Optional[str] = None
    target_field: Optional[str] = None
    
    # Transformation parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Conditional execution
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Error handling
    on_error: str = "skip"  # skip, fail, default_value
    default_value: Any = None
    
    # Execution settings
    enabled: bool = True
    priority: int = 100
    
    # Custom function reference
    custom_function: Optional[Callable] = None
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "transformation_type": self.transformation_type.value,
            "description": self.description,
            "source_field": self.source_field,
            "target_field": self.target_field,
            "parameters": self.parameters,
            "conditions": self.conditions,
            "on_error": self.on_error,
            "default_value": self.default_value,
            "enabled": self.enabled,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags
        }


@dataclass
class ValidationRule:
    """Configuration for data validation rules"""
    rule_id: str
    name: str
    description: str = ""
    
    # Validation configuration
    field_name: str
    validation_type: str  # required, type, range, pattern, custom
    validation_level: ValidationLevel = ValidationLevel.ERROR
    
    # Validation parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Error message
    error_message: Optional[str] = None
    
    # Custom validation function
    custom_validator: Optional[Callable] = None
    
    # Execution settings
    enabled: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "field_name": self.field_name,
            "validation_type": self.validation_type,
            "validation_level": self.validation_level.value,
            "parameters": self.parameters,
            "error_message": self.error_message,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags
        }


@dataclass
class TransformationResult:
    """Result of data transformation"""
    success: bool
    transformed_data: Dict[str, Any]
    original_data: Dict[str, Any]
    applied_rules: List[str] = field(default_factory=list)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "transformed_data": self.transformed_data,
            "original_data": self.original_data,
            "applied_rules": self.applied_rules,
            "validation_results": self.validation_results,
            "errors": self.errors,
            "warnings": self.warnings,
            "execution_time_ms": self.execution_time_ms
        }


class DataTransformer(ABC):
    """Abstract base class for data transformers"""
    
    def __init__(self, transformer_id: str, name: str):
        self.transformer_id = transformer_id
        self.name = name
        
        # Statistics
        self.stats = {
            "transformations_applied": 0,
            "successful_transformations": 0,
            "failed_transformations": 0,
            "avg_execution_time_ms": 0.0
        }
    
    @abstractmethod
    async def transform(self, data: Dict[str, Any], rule: TransformationRule) -> Any:
        """Apply transformation to data"""
        pass
    
    def update_stats(self, execution_time_ms: float, success: bool):
        """Update transformer statistics"""
        self.stats["transformations_applied"] += 1
        
        if success:
            self.stats["successful_transformations"] += 1
        else:
            self.stats["failed_transformations"] += 1
        
        # Update average execution time
        total_transformations = self.stats["transformations_applied"]
        current_avg = self.stats["avg_execution_time_ms"]
        self.stats["avg_execution_time_ms"] = \
            (current_avg * (total_transformations - 1) + execution_time_ms) / total_transformations
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transformer statistics"""
        return {
            "transformer_id": self.transformer_id,
            "name": self.name,
            "statistics": self.stats
        }


class FieldMappingTransformer(DataTransformer):
    """Transformer for field mapping operations"""
    
    def __init__(self):
        super().__init__("field_mapping", "Field Mapping Transformer")
    
    async def transform(self, data: Dict[str, Any], rule: TransformationRule) -> Any:
        """Apply field mapping transformation"""
        source_field = rule.source_field
        target_field = rule.target_field or source_field
        
        if not source_field:
            raise ValueError("Source field required for field mapping")
        
        # Get value from source field (supports nested fields)
        value = self._get_nested_value(data, source_field)
        
        # Apply any value transformations
        if "value_transformation" in rule.parameters:
            value = await self._apply_value_transformation(value, rule.parameters["value_transformation"])
        
        return {target_field: value}
    
    def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get value from nested field path (e.g., 'user.profile.name')"""
        fields = field_path.split('.')
        current_value = data
        
        for field in fields:
            if isinstance(current_value, dict) and field in current_value:
                current_value = current_value[field]
            else:
                return None
        
        return current_value
    
    async def _apply_value_transformation(self, value: Any, transformation: str) -> Any:
        """Apply value transformation"""
        if transformation == "uppercase":
            return str(value).upper() if value is not None else None
        elif transformation == "lowercase":
            return str(value).lower() if value is not None else None
        elif transformation == "trim":
            return str(value).strip() if value is not None else None
        elif transformation == "title_case":
            return str(value).title() if value is not None else None
        else:
            return value


class DataTypeConversionTransformer(DataTransformer):
    """Transformer for data type conversions"""
    
    def __init__(self):
        super().__init__("data_type_conversion", "Data Type Conversion Transformer")
    
    async def transform(self, data: Dict[str, Any], rule: TransformationRule) -> Any:
        """Apply data type conversion"""
        source_field = rule.source_field
        target_type = rule.parameters.get("target_type")
        
        if not source_field or not target_type:
            raise ValueError("Source field and target type required for conversion")
        
        value = data.get(source_field)
        if value is None:
            return None
        
        try:
            if target_type == "string":
                converted_value = str(value)
            elif target_type == "integer":
                converted_value = int(float(value))  # Handle string numbers
            elif target_type == "float":
                converted_value = float(value)
            elif target_type == "boolean":
                if isinstance(value, str):
                    converted_value = value.lower() in ['true', '1', 'yes', 'on']
                else:
                    converted_value = bool(value)
            elif target_type == "datetime":
                converted_value = self._parse_datetime(value, rule.parameters.get("format"))
            elif target_type == "date":
                dt = self._parse_datetime(value, rule.parameters.get("format"))
                converted_value = dt.date() if dt else None
            elif target_type == "json":
                if isinstance(value, str):
                    converted_value = json.loads(value)
                else:
                    converted_value = value
            else:
                raise ValueError(f"Unsupported target type: {target_type}")
            
            return converted_value
            
        except Exception as e:
            if rule.on_error == "default_value":
                return rule.default_value
            elif rule.on_error == "skip":
                return value  # Return original value
            else:
                raise ValueError(f"Type conversion failed: {e}")
    
    def _parse_datetime(self, value: Any, date_format: Optional[str] = None) -> Optional[datetime]:
        """Parse datetime from various formats"""
        if isinstance(value, datetime):
            return value
        
        if not isinstance(value, str):
            value = str(value)
        
        try:
            if date_format:
                return datetime.strptime(value, date_format)
            else:
                # Try common formats
                from dateutil import parser
                return parser.parse(value)
        except Exception:
            return None


class ValueTransformationTransformer(DataTransformer):
    """Transformer for complex value transformations"""
    
    def __init__(self):
        super().__init__("value_transformation", "Value Transformation Transformer")
    
    async def transform(self, data: Dict[str, Any], rule: TransformationRule) -> Any:
        """Apply value transformation"""
        source_field = rule.source_field
        transformation_type = rule.parameters.get("transformation_type")
        
        if not source_field:
            raise ValueError("Source field required for value transformation")
        
        value = data.get(source_field)
        
        if transformation_type == "regex_replace":
            return self._regex_replace(value, rule.parameters)
        elif transformation_type == "substring":
            return self._substring(value, rule.parameters)
        elif transformation_type == "concatenate":
            return self._concatenate(data, rule.parameters)
        elif transformation_type == "split":
            return self._split(value, rule.parameters)
        elif transformation_type == "calculate":
            return self._calculate(data, rule.parameters)
        elif transformation_type == "lookup":
            return await self._lookup(value, rule.parameters)
        elif transformation_type == "format_template":
            return self._format_template(data, rule.parameters)
        else:
            raise ValueError(f"Unsupported transformation type: {transformation_type}")
    
    def _regex_replace(self, value: Any, params: Dict[str, Any]) -> str:
        """Apply regex replacement"""
        if value is None:
            return None
        
        pattern = params.get("pattern")
        replacement = params.get("replacement", "")
        flags = params.get("flags", 0)
        
        return re.sub(pattern, replacement, str(value), flags=flags)
    
    def _substring(self, value: Any, params: Dict[str, Any]) -> str:
        """Extract substring"""
        if value is None:
            return None
        
        str_value = str(value)
        start = params.get("start", 0)
        end = params.get("end")
        
        if end is not None:
            return str_value[start:end]
        else:
            return str_value[start:]
    
    def _concatenate(self, data: Dict[str, Any], params: Dict[str, Any]) -> str:
        """Concatenate multiple fields"""
        fields = params.get("fields", [])
        separator = params.get("separator", "")
        
        values = []
        for field in fields:
            value = data.get(field)
            if value is not None:
                values.append(str(value))
        
        return separator.join(values)
    
    def _split(self, value: Any, params: Dict[str, Any]) -> List[str]:
        """Split value into list"""
        if value is None:
            return []
        
        delimiter = params.get("delimiter", ",")
        max_splits = params.get("max_splits", -1)
        
        return str(value).split(delimiter, max_splits)
    
    def _calculate(self, data: Dict[str, Any], params: Dict[str, Any]) -> float:
        """Perform calculations on field values"""
        expression = params.get("expression")
        if not expression:
            raise ValueError("Expression required for calculation")
        
        # Simple expression evaluator (could be extended with safe_eval)
        # For now, support basic arithmetic with field references
        
        # Replace field references with values
        for field_name, field_value in data.items():
            if isinstance(field_value, (int, float)):
                expression = expression.replace(f"{{{field_name}}}", str(field_value))
        
        try:
            # Use eval with restricted globals for safety
            allowed_names = {
                "__builtins__": {},
                "abs": abs,
                "min": min,
                "max": max,
                "round": round,
                "math": math
            }
            return eval(expression, allowed_names)
        except Exception as e:
            raise ValueError(f"Calculation failed: {e}")
    
    async def _lookup(self, value: Any, params: Dict[str, Any]) -> Any:
        """Perform lookup transformation"""
        lookup_table = params.get("lookup_table", {})
        default_value = params.get("default_value")
        
        return lookup_table.get(str(value), default_value)
    
    def _format_template(self, data: Dict[str, Any], params: Dict[str, Any]) -> str:
        """Format template with data values"""
        template = params.get("template", "")
        
        # Simple template substitution
        try:
            return template.format(**data)
        except KeyError as e:
            raise ValueError(f"Template formatting failed - missing field: {e}")


class ConditionalLogicTransformer(DataTransformer):
    """Transformer for conditional logic"""
    
    def __init__(self):
        super().__init__("conditional_logic", "Conditional Logic Transformer")
    
    async def transform(self, data: Dict[str, Any], rule: TransformationRule) -> Any:
        """Apply conditional logic transformation"""
        conditions = rule.parameters.get("conditions", [])
        default_value = rule.parameters.get("default_value")
        
        for condition in conditions:
            if self._evaluate_condition(data, condition):
                return condition.get("result")
        
        return default_value
    
    def _evaluate_condition(self, data: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """Evaluate a condition"""
        field = condition.get("field")
        operator = condition.get("operator")
        value = condition.get("value")
        
        if not field or not operator:
            return False
        
        field_value = data.get(field)
        
        if operator == "equals":
            return field_value == value
        elif operator == "not_equals":
            return field_value != value
        elif operator == "greater_than":
            return field_value > value
        elif operator == "less_than":
            return field_value < value
        elif operator == "greater_equal":
            return field_value >= value
        elif operator == "less_equal":
            return field_value <= value
        elif operator == "contains":
            return str(value) in str(field_value)
        elif operator == "starts_with":
            return str(field_value).startswith(str(value))
        elif operator == "ends_with":
            return str(field_value).endswith(str(value))
        elif operator == "in":
            return field_value in value
        elif operator == "not_in":
            return field_value not in value
        elif operator == "is_null":
            return field_value is None
        elif operator == "is_not_null":
            return field_value is not None
        
        return False


class DataValidator:
    """Data validation engine"""
    
    def __init__(self):
        self.validation_rules: Dict[str, ValidationRule] = {}
        
        # Statistics
        self.stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "validation_errors": 0,
            "validation_warnings": 0
        }
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add validation rule"""
        self.validation_rules[rule.rule_id] = rule
        logger.info(f"Added validation rule: {rule.name}")
    
    async def validate_record(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate a single record against all rules"""
        validation_results = []
        
        for rule in self.validation_rules.values():
            if not rule.enabled:
                continue
            
            try:
                result = await self._apply_validation_rule(data, rule)
                validation_results.append(result)
                
                if not result["valid"]:
                    if rule.validation_level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]:
                        self.stats["validation_errors"] += 1
                    else:
                        self.stats["validation_warnings"] += 1
                
            except Exception as e:
                logger.error(f"Validation rule error: {rule.name} - {e}")
                validation_results.append({
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "valid": False,
                    "level": "error",
                    "message": f"Validation rule failed: {e}"
                })
        
        self.stats["total_validations"] += 1
        if all(r["valid"] for r in validation_results):
            self.stats["successful_validations"] += 1
        else:
            self.stats["failed_validations"] += 1
        
        return validation_results
    
    async def _apply_validation_rule(self, data: Dict[str, Any], rule: ValidationRule) -> Dict[str, Any]:
        """Apply a single validation rule"""
        field_value = data.get(rule.field_name)
        
        result = {
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "field_name": rule.field_name,
            "level": rule.validation_level.value,
            "valid": True,
            "message": ""
        }
        
        try:
            if rule.validation_type == "required":
                if field_value is None or field_value == "":
                    result["valid"] = False
                    result["message"] = rule.error_message or f"Field {rule.field_name} is required"
            
            elif rule.validation_type == "type":
                expected_type = rule.parameters.get("expected_type")
                if field_value is not None and not self._check_type(field_value, expected_type):
                    result["valid"] = False
                    result["message"] = rule.error_message or f"Field {rule.field_name} should be of type {expected_type}"
            
            elif rule.validation_type == "range":
                if field_value is not None:
                    min_value = rule.parameters.get("min_value")
                    max_value = rule.parameters.get("max_value")
                    
                    if min_value is not None and field_value < min_value:
                        result["valid"] = False
                        result["message"] = rule.error_message or f"Field {rule.field_name} below minimum: {min_value}"
                    elif max_value is not None and field_value > max_value:
                        result["valid"] = False
                        result["message"] = rule.error_message or f"Field {rule.field_name} above maximum: {max_value}"
            
            elif rule.validation_type == "length":
                if field_value is not None:
                    min_length = rule.parameters.get("min_length")
                    max_length = rule.parameters.get("max_length")
                    field_length = len(str(field_value))
                    
                    if min_length is not None and field_length < min_length:
                        result["valid"] = False
                        result["message"] = rule.error_message or f"Field {rule.field_name} too short (min: {min_length})"
                    elif max_length is not None and field_length > max_length:
                        result["valid"] = False
                        result["message"] = rule.error_message or f"Field {rule.field_name} too long (max: {max_length})"
            
            elif rule.validation_type == "pattern":
                if field_value is not None:
                    pattern = rule.parameters.get("pattern")
                    if pattern and not re.match(pattern, str(field_value)):
                        result["valid"] = False
                        result["message"] = rule.error_message or f"Field {rule.field_name} doesn't match pattern"
            
            elif rule.validation_type == "enum":
                if field_value is not None:
                    allowed_values = rule.parameters.get("allowed_values", [])
                    if field_value not in allowed_values:
                        result["valid"] = False
                        result["message"] = rule.error_message or f"Field {rule.field_name} not in allowed values"
            
            elif rule.validation_type == "custom":
                if rule.custom_validator:
                    custom_result = await rule.custom_validator(field_value, data, rule.parameters)
                    if not custom_result:
                        result["valid"] = False
                        result["message"] = rule.error_message or f"Custom validation failed for {rule.field_name}"
            
        except Exception as e:
            result["valid"] = False
            result["message"] = f"Validation error: {e}"
        
        return result
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "integer":
            return isinstance(value, int)
        elif expected_type == "float":
            return isinstance(value, (int, float))
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "list":
            return isinstance(value, list)
        elif expected_type == "dict":
            return isinstance(value, dict)
        elif expected_type == "datetime":
            return isinstance(value, datetime)
        else:
            return True  # Unknown type, assume valid
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            "validation_rules_count": len(self.validation_rules),
            "statistics": self.stats
        }


class TransformationEngine:
    """Main transformation engine orchestrator"""
    
    def __init__(self):
        # Transformer registry
        self.transformers: Dict[TransformationType, DataTransformer] = {
            TransformationType.FIELD_MAPPING: FieldMappingTransformer(),
            TransformationType.DATA_TYPE_CONVERSION: DataTypeConversionTransformer(),
            TransformationType.VALUE_TRANSFORMATION: ValueTransformationTransformer(),
            TransformationType.CONDITIONAL_LOGIC: ConditionalLogicTransformer(),
        }
        
        # Rules and validators
        self.transformation_rules: Dict[str, TransformationRule] = {}
        self.validator = DataValidator()
        
        # Schema mappings
        self.schema_mappings: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.stats = {
            "total_transformations": 0,
            "successful_transformations": 0,
            "failed_transformations": 0,
            "avg_execution_time_ms": 0.0,
            "rules_applied": 0
        }
        
        logger.info("Transformation Engine initialized")
    
    def add_transformation_rule(self, rule: TransformationRule):
        """Add transformation rule"""
        self.transformation_rules[rule.rule_id] = rule
        logger.info(f"Added transformation rule: {rule.name}")
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add validation rule"""
        self.validator.add_validation_rule(rule)
    
    def register_custom_transformer(self, transformation_type: TransformationType, 
                                   transformer: DataTransformer):
        """Register custom transformer"""
        self.transformers[transformation_type] = transformer
        logger.info(f"Registered custom transformer: {transformer.name}")
    
    def add_schema_mapping(self, mapping_id: str, source_schema: Dict[str, Any], 
                          target_schema: Dict[str, Any]):
        """Add schema mapping configuration"""
        self.schema_mappings[mapping_id] = {
            "source_schema": source_schema,
            "target_schema": target_schema,
            "field_mappings": self._generate_field_mappings(source_schema, target_schema)
        }
    
    async def transform_record(self, data: Dict[str, Any], 
                              rule_ids: Optional[List[str]] = None) -> TransformationResult:
        """Transform a single record"""
        start_time = datetime.now()
        
        result = TransformationResult(
            success=True,
            transformed_data=data.copy(),
            original_data=data.copy()
        )
        
        try:
            # Get rules to apply
            rules_to_apply = []
            if rule_ids:
                rules_to_apply = [self.transformation_rules[rid] for rid in rule_ids 
                                if rid in self.transformation_rules]
            else:
                rules_to_apply = list(self.transformation_rules.values())
            
            # Sort rules by priority
            rules_to_apply.sort(key=lambda r: r.priority)
            
            # Apply transformation rules
            for rule in rules_to_apply:
                if not rule.enabled:
                    continue
                
                try:
                    # Check conditions
                    if rule.conditions and not self._evaluate_conditions(result.transformed_data, rule.conditions):
                        continue
                    
                    # Apply transformation
                    if rule.transformation_type in self.transformers:
                        transformer = self.transformers[rule.transformation_type]
                        
                        if rule.custom_function:
                            # Use custom function
                            transformed_value = await rule.custom_function(result.transformed_data, rule)
                        else:
                            # Use registered transformer
                            transformed_value = await transformer.transform(result.transformed_data, rule)
                        
                        # Apply transformed value
                        if isinstance(transformed_value, dict):
                            result.transformed_data.update(transformed_value)
                        elif rule.target_field:
                            result.transformed_data[rule.target_field] = transformed_value
                        
                        result.applied_rules.append(rule.rule_id)
                        self.stats["rules_applied"] += 1
                        
                        # Update transformer stats
                        transformer.update_stats(0, True)  # Execution time tracked separately
                    
                except Exception as e:
                    error_msg = f"Rule {rule.name} failed: {e}"
                    
                    if rule.on_error == "fail":
                        result.success = False
                        result.errors.append(error_msg)
                        break
                    elif rule.on_error == "default_value" and rule.default_value is not None:
                        if rule.target_field:
                            result.transformed_data[rule.target_field] = rule.default_value
                        result.warnings.append(f"{error_msg} - using default value")
                    else:
                        result.warnings.append(f"{error_msg} - skipping rule")
            
            # Validate transformed data
            if result.success:
                result.validation_results = await self.validator.validate_record(result.transformed_data)
                
                # Check for critical validation errors
                critical_errors = [v for v in result.validation_results 
                                 if not v["valid"] and v["level"] == "critical"]
                if critical_errors:
                    result.success = False
                    result.errors.extend([v["message"] for v in critical_errors])
            
            # Update statistics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
            
            self.stats["total_transformations"] += 1
            if result.success:
                self.stats["successful_transformations"] += 1
            else:
                self.stats["failed_transformations"] += 1
            
            # Update average execution time
            total_transformations = self.stats["total_transformations"]
            current_avg = self.stats["avg_execution_time_ms"]
            self.stats["avg_execution_time_ms"] = \
                (current_avg * (total_transformations - 1) + execution_time) / total_transformations
            
        except Exception as e:
            result.success = False
            result.errors.append(f"Transformation engine error: {e}")
            logger.error(f"Transformation failed: {e}")
        
        return result
    
    async def transform_batch(self, data_batch: List[Dict[str, Any]], 
                             rule_ids: Optional[List[str]] = None) -> List[TransformationResult]:
        """Transform batch of records"""
        results = []
        
        for record in data_batch:
            result = await self.transform_record(record, rule_ids)
            results.append(result)
        
        return results
    
    async def apply_schema_mapping(self, data: Dict[str, Any], 
                                  mapping_id: str) -> TransformationResult:
        """Apply schema mapping to transform data structure"""
        if mapping_id not in self.schema_mappings:
            raise ValueError(f"Schema mapping not found: {mapping_id}")
        
        mapping = self.schema_mappings[mapping_id]
        field_mappings = mapping["field_mappings"]
        
        # Create transformation rules from schema mapping
        rules = []
        for source_field, target_field in field_mappings.items():
            rule = TransformationRule(
                rule_id=f"schema_mapping_{source_field}_{target_field}",
                name=f"Map {source_field} to {target_field}",
                transformation_type=TransformationType.FIELD_MAPPING,
                source_field=source_field,
                target_field=target_field
            )
            rules.append(rule)
        
        # Apply rules
        result = TransformationResult(
            success=True,
            transformed_data={},
            original_data=data.copy()
        )
        
        for rule in rules:
            try:
                transformer = self.transformers[TransformationType.FIELD_MAPPING]
                transformed_value = await transformer.transform(data, rule)
                
                if isinstance(transformed_value, dict):
                    result.transformed_data.update(transformed_value)
                
                result.applied_rules.append(rule.rule_id)
                
            except Exception as e:
                result.warnings.append(f"Schema mapping failed for {rule.source_field}: {e}")
        
        return result
    
    def _evaluate_conditions(self, data: Dict[str, Any], conditions: List[Dict[str, Any]]) -> bool:
        """Evaluate conditional logic"""
        if not conditions:
            return True
        
        # Simple AND logic for now (could be extended)
        for condition in conditions:
            field = condition.get("field")
            operator = condition.get("operator")
            value = condition.get("value")
            
            field_value = data.get(field)
            
            if operator == "equals" and field_value != value:
                return False
            elif operator == "not_equals" and field_value == value:
                return False
            elif operator == "greater_than" and field_value <= value:
                return False
            elif operator == "less_than" and field_value >= value:
                return False
            # Add more operators as needed
        
        return True
    
    def _generate_field_mappings(self, source_schema: Dict[str, Any], 
                                target_schema: Dict[str, Any]) -> Dict[str, str]:
        """Generate field mappings from schemas"""
        mappings = {}
        
        # Simple field name matching (could be enhanced with fuzzy matching)
        source_fields = set(source_schema.get("fields", {}).keys())
        target_fields = set(target_schema.get("fields", {}).keys())
        
        # Exact matches
        for field in source_fields.intersection(target_fields):
            mappings[field] = field
        
        # Could add fuzzy matching, semantic matching, etc.
        
        return mappings
    
    def get_transformation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive transformation statistics"""
        transformer_stats = {}
        for transform_type, transformer in self.transformers.items():
            transformer_stats[transform_type.value] = transformer.get_stats()
        
        return {
            "engine_statistics": self.stats,
            "transformer_statistics": transformer_stats,
            "validation_statistics": self.validator.get_stats(),
            "transformation_rules_count": len(self.transformation_rules),
            "schema_mappings_count": len(self.schema_mappings)
        }
    
    def list_transformation_rules(self) -> List[Dict[str, Any]]:
        """List all transformation rules"""
        return [rule.to_dict() for rule in self.transformation_rules.values()]
    
    def list_validation_rules(self) -> List[Dict[str, Any]]:
        """List all validation rules"""
        return [rule.to_dict() for rule in self.validator.validation_rules.values()]


# Export main classes
__all__ = [
    'TransformationType',
    'ValidationLevel',
    'TransformationRule',
    'ValidationRule',
    'TransformationResult',
    'DataTransformer',
    'FieldMappingTransformer',
    'DataTypeConversionTransformer',
    'ValueTransformationTransformer',
    'ConditionalLogicTransformer',
    'DataValidator',
    'TransformationEngine'
]