"""
RLT Explanation Formatter

Implements input/output formatting for Sakana's RLT methodology.
Handles the transformation between question+solution inputs and 
structured explanations for student distillation.

Key Features:
- Question+solution input formatting for teachers
- Think token extraction for student training
- Distillation prompt creation
- Multi-format support (text, structured, JSON)
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import UUID, uuid4
import structlog

from prsm.core.models import PRSMBaseModel

logger = structlog.get_logger()


@dataclass
class RLTFormatConfig:
    """Configuration for RLT formatting"""
    # Input formatting
    question_prefix: str = "Question:"
    solution_prefix: str = "Solution:"
    explanation_prefix: str = "Explain:"
    
    # Output formatting
    think_start_tag: str = "<think>"
    think_end_tag: str = "</think>"
    solution_start_tag: str = "<solution>"
    solution_end_tag: str = "</solution>"
    
    # Limits
    max_question_length: int = 1000
    max_solution_length: int = 500
    max_explanation_length: int = 2048
    
    # Validation
    require_structured_output: bool = True
    validate_explanation_quality: bool = True


class FormattedRLTInput(PRSMBaseModel):
    """Structured RLT input containing question and solution"""
    question: str
    solution: str
    formatted_input: str
    metadata: Dict[str, Any] = {}
    
    def __str__(self) -> str:
        return self.formatted_input


class FormattedRLTOutput(PRSMBaseModel):
    """Structured RLT output with extracted components"""
    raw_output: str
    think_content: str
    solution_content: str
    explanation_quality_score: float = 0.0
    is_valid: bool = True
    validation_errors: List[str] = []
    metadata: Dict[str, Any] = {}


class StudentDistillationPrompt(PRSMBaseModel):
    """Formatted prompt for student distillation training"""
    question: str
    think_tokens: str
    solution: str
    formatted_prompt: str
    distillation_type: str = "standard"  # "standard", "enhanced", "adaptive"
    metadata: Dict[str, Any] = {}


class RLTFormatter:
    """
    Advanced formatter for RLT input/output processing.
    
    Handles the conversion between Sakana's question+solution methodology
    and structured teacher-student training formats.
    """
    
    def __init__(self, config: Optional[RLTFormatConfig] = None):
        self.config = config or RLTFormatConfig()
        self.logger = logger.bind(component="RLTFormatter")
        
        # Formatting statistics
        self.format_count = 0
        self.parse_count = 0
        self.validation_errors = 0
        
    def format_question_solution_input(
        self,
        question: str,
        solution: str,
        include_metadata: bool = True
    ) -> FormattedRLTInput:
        """
        Format question and solution into RLT teacher input format.
        
        Creates the input prompt that provides both question and solution
        to the teacher model for explanation generation.
        
        Args:
            question: The question to be explained
            solution: The known solution to the question
            include_metadata: Whether to include formatting metadata
            
        Returns:
            FormattedRLTInput object with formatted prompt
        """
        try:
            # Validate inputs
            if not question or not solution:
                raise ValueError("Question and solution cannot be empty")
            
            if len(question) > self.config.max_question_length:
                self.logger.warning("Question exceeds max length, truncating")
                question = question[:self.config.max_question_length] + "..."
            
            if len(solution) > self.config.max_solution_length:
                self.logger.warning("Solution exceeds max length, truncating")
                solution = solution[:self.config.max_solution_length] + "..."
            
            # Create formatted input
            formatted_input = (
                f"{self.config.question_prefix} {question}\n"
                f"{self.config.solution_prefix} {solution}\n"
                f"{self.config.explanation_prefix}"
            )
            
            # Prepare metadata
            metadata = {}
            if include_metadata:
                metadata = {
                    "formatted_at": datetime.now(timezone.utc).isoformat(),
                    "question_length": len(question),
                    "solution_length": len(solution),
                    "total_input_length": len(formatted_input),
                    "format_version": "1.0"
                }
            
            # Create structured input
            rlt_input = FormattedRLTInput(
                question=question,
                solution=solution,
                formatted_input=formatted_input,
                metadata=metadata
            )
            
            self.format_count += 1
            
            self.logger.debug(
                "Formatted RLT input",
                question_length=len(question),
                solution_length=len(solution),
                total_length=len(formatted_input)
            )
            
            return rlt_input
            
        except Exception as e:
            self.logger.error("Error formatting RLT input", error=str(e))
            raise
    
    def parse_rlt_output(
        self,
        raw_output: str,
        validate: bool = True
    ) -> FormattedRLTOutput:
        """
        Parse raw RLT teacher output into structured components.
        
        Extracts think tokens and solution content from the teacher's
        formatted response for use in student training.
        
        Args:
            raw_output: Raw output from RLT teacher model
            validate: Whether to validate the parsed output
            
        Returns:
            FormattedRLTOutput with extracted components
        """
        try:
            if not raw_output:
                raise ValueError("Raw output cannot be empty")
            
            # Initialize parsed components
            think_content = ""
            solution_content = ""
            validation_errors = []
            is_valid = True
            
            # Extract think content
            think_pattern = f"{re.escape(self.config.think_start_tag)}(.*?){re.escape(self.config.think_end_tag)}"
            think_match = re.search(think_pattern, raw_output, re.DOTALL)
            
            if think_match:
                think_content = think_match.group(1).strip()
            else:
                # Try to extract without tags (fallback)
                think_content = raw_output.strip()
                if validate and self.config.require_structured_output:
                    validation_errors.append("No think tags found in output")
                    is_valid = False
            
            # Extract solution content
            solution_pattern = f"{re.escape(self.config.solution_start_tag)}(.*?){re.escape(self.config.solution_end_tag)}"
            solution_match = re.search(solution_pattern, raw_output, re.DOTALL)
            
            if solution_match:
                solution_content = solution_match.group(1).strip()
            elif validate and self.config.require_structured_output:
                validation_errors.append("No solution tags found in output")
                is_valid = False
            
            # Validate content quality if requested
            explanation_quality_score = 0.0
            if validate and self.config.validate_explanation_quality:
                explanation_quality_score = self._assess_explanation_quality(think_content)
                
                if explanation_quality_score < 0.3:
                    validation_errors.append("Low explanation quality score")
                    is_valid = False
            
            # Create structured output
            rlt_output = FormattedRLTOutput(
                raw_output=raw_output,
                think_content=think_content,
                solution_content=solution_content,
                explanation_quality_score=explanation_quality_score,
                is_valid=is_valid,
                validation_errors=validation_errors,
                metadata={
                    "parsed_at": datetime.now(timezone.utc).isoformat(),
                    "raw_length": len(raw_output),
                    "think_length": len(think_content),
                    "solution_length": len(solution_content),
                    "has_think_tags": think_match is not None,
                    "has_solution_tags": solution_match is not None
                }
            )
            
            self.parse_count += 1
            if not is_valid:
                self.validation_errors += 1
            
            self.logger.debug(
                "Parsed RLT output",
                is_valid=is_valid,
                think_length=len(think_content),
                quality_score=explanation_quality_score,
                errors=len(validation_errors)
            )
            
            return rlt_output
            
        except Exception as e:
            self.logger.error("Error parsing RLT output", error=str(e))
            raise
    
    def extract_think_tokens(self, rlt_output: str) -> str:
        """
        Extract think tokens from RLT output for student training.
        
        Simplified version of parse_rlt_output that only extracts
        the explanation content without full validation.
        
        Args:
            rlt_output: Raw RLT teacher output
            
        Returns:
            Extracted think/explanation content
        """
        try:
            # Try structured extraction first
            think_pattern = f"{re.escape(self.config.think_start_tag)}(.*?){re.escape(self.config.think_end_tag)}"
            think_match = re.search(think_pattern, rlt_output, re.DOTALL)
            
            if think_match:
                return think_match.group(1).strip()
            
            # Fallback: return entire output
            return rlt_output.strip()
            
        except Exception as e:
            self.logger.warning("Error extracting think tokens", error=str(e))
            return rlt_output.strip()
    
    def create_student_distillation_prompt(
        self,
        question: str,
        think_tokens: str,
        solution: str,
        distillation_type: str = "standard"
    ) -> StudentDistillationPrompt:
        """
        Create formatted prompt for student distillation training.
        
        Formats the question, extracted think tokens, and solution
        into a prompt suitable for training student models.
        
        Args:
            question: Original question
            think_tokens: Extracted explanation from teacher
            solution: Ground truth solution
            distillation_type: Type of distillation ("standard", "enhanced", "adaptive")
            
        Returns:
            StudentDistillationPrompt ready for training
        """
        try:
            if distillation_type == "standard":
                # Standard format: Question -> Think -> Solution
                formatted_prompt = (
                    f"{self.config.question_prefix} {question}\n"
                    f"{self.config.think_start_tag}\n{think_tokens}\n{self.config.think_end_tag}\n"
                    f"{self.config.solution_start_tag}\n{solution}\n{self.config.solution_end_tag}"
                )
                
            elif distillation_type == "enhanced":
                # Enhanced format with explicit reasoning steps
                formatted_prompt = (
                    f"{self.config.question_prefix} {question}\n"
                    f"Let me think through this step by step:\n"
                    f"{self.config.think_start_tag}\n{think_tokens}\n{self.config.think_end_tag}\n"
                    f"Based on this reasoning, {self.config.solution_prefix.lower()} {solution}"
                )
                
            elif distillation_type == "adaptive":
                # Adaptive format that adjusts to content complexity
                complexity = self._assess_content_complexity(question, solution)
                
                if complexity > 0.7:
                    # High complexity: detailed format
                    formatted_prompt = (
                        f"Complex Problem: {question}\n"
                        f"Detailed Analysis:\n"
                        f"{self.config.think_start_tag}\n{think_tokens}\n{self.config.think_end_tag}\n"
                        f"Final Answer: {solution}"
                    )
                else:
                    # Lower complexity: simpler format
                    formatted_prompt = (
                        f"Problem: {question}\n"
                        f"Reasoning: {think_tokens}\n"
                        f"Answer: {solution}"
                    )
            else:
                raise ValueError(f"Unknown distillation type: {distillation_type}")
            
            # Create structured prompt
            distillation_prompt = StudentDistillationPrompt(
                question=question,
                think_tokens=think_tokens,
                solution=solution,
                formatted_prompt=formatted_prompt,
                distillation_type=distillation_type,
                metadata={
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "prompt_length": len(formatted_prompt),
                    "think_tokens_length": len(think_tokens),
                    "complexity_assessed": distillation_type == "adaptive"
                }
            )
            
            self.logger.debug(
                "Created student distillation prompt",
                type=distillation_type,
                length=len(formatted_prompt)
            )
            
            return distillation_prompt
            
        except Exception as e:
            self.logger.error("Error creating distillation prompt", error=str(e))
            raise
    
    def batch_format_inputs(
        self,
        questions: List[str],
        solutions: List[str]
    ) -> List[FormattedRLTInput]:
        """
        Format multiple question-solution pairs in batch.
        
        Args:
            questions: List of questions
            solutions: List of corresponding solutions
            
        Returns:
            List of formatted RLT inputs
        """
        if len(questions) != len(solutions):
            raise ValueError("Questions and solutions must have the same length")
        
        formatted_inputs = []
        for i, (question, solution) in enumerate(zip(questions, solutions)):
            try:
                formatted_input = self.format_question_solution_input(question, solution)
                formatted_inputs.append(formatted_input)
            except Exception as e:
                self.logger.error(f"Error formatting input {i}", error=str(e))
                # Continue with other inputs
                continue
        
        self.logger.info(
            "Batch formatting completed",
            total_inputs=len(questions),
            successful=len(formatted_inputs),
            failed=len(questions) - len(formatted_inputs)
        )
        
        return formatted_inputs
    
    def batch_create_distillation_prompts(
        self,
        questions: List[str],
        think_tokens_list: List[str],
        solutions: List[str],
        distillation_type: str = "standard"
    ) -> List[StudentDistillationPrompt]:
        """
        Create multiple student distillation prompts in batch.
        """
        if not (len(questions) == len(think_tokens_list) == len(solutions)):
            raise ValueError("All input lists must have the same length")
        
        prompts = []
        for i, (question, think_tokens, solution) in enumerate(zip(questions, think_tokens_list, solutions)):
            try:
                prompt = self.create_student_distillation_prompt(
                    question, think_tokens, solution, distillation_type
                )
                prompts.append(prompt)
            except Exception as e:
                self.logger.error(f"Error creating distillation prompt {i}", error=str(e))
                continue
        
        return prompts
    
    def _assess_explanation_quality(self, explanation: str) -> float:
        """
        Simple heuristic assessment of explanation quality.
        """
        if not explanation:
            return 0.0
        
        # Basic quality indicators
        quality_score = 0.0
        
        # Length indicator
        if 50 <= len(explanation) <= 1000:
            quality_score += 0.3
        
        # Sentence structure
        sentences = explanation.split('.')
        if 2 <= len(sentences) <= 10:
            quality_score += 0.2
        
        # Logical connectors
        connectors = ['because', 'therefore', 'since', 'thus', 'so', 'then', 'first', 'next']
        connector_count = sum(1 for conn in connectors if conn.lower() in explanation.lower())
        quality_score += min(0.3, connector_count * 0.1)
        
        # Mathematical/technical terms (domain-specific enhancement)
        tech_terms = ['equation', 'formula', 'theorem', 'proof', 'calculate', 'derive']
        tech_count = sum(1 for term in tech_terms if term.lower() in explanation.lower())
        quality_score += min(0.2, tech_count * 0.05)
        
        return min(1.0, quality_score)
    
    def _assess_content_complexity(self, question: str, solution: str) -> float:
        """
        Assess the complexity of question-solution pair.
        """
        complexity_score = 0.0
        
        # Length-based complexity
        total_length = len(question) + len(solution)
        if total_length > 200:
            complexity_score += 0.3
        
        # Mathematical complexity indicators
        math_indicators = ['integral', 'derivative', 'equation', 'theorem', 'proof', 'formula']
        math_count = sum(1 for term in math_indicators 
                        if term.lower() in question.lower() or term.lower() in solution.lower())
        complexity_score += min(0.4, math_count * 0.1)
        
        # Number of steps/components
        step_indicators = ['step', 'first', 'second', 'then', 'next', 'finally']
        step_count = sum(1 for term in step_indicators 
                        if term.lower() in question.lower() or term.lower() in solution.lower())
        complexity_score += min(0.3, step_count * 0.1)
        
        return min(1.0, complexity_score)
    
    def get_formatting_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about formatting operations.
        """
        total_operations = self.format_count + self.parse_count
        error_rate = self.validation_errors / max(self.parse_count, 1)
        
        return {
            "total_format_operations": self.format_count,
            "total_parse_operations": self.parse_count,
            "total_operations": total_operations,
            "validation_errors": self.validation_errors,
            "error_rate": error_rate,
            "config": self.config.__dict__
        }
    
    def export_formatted_data(
        self,
        formatted_inputs: List[FormattedRLTInput],
        output_format: str = "json"
    ) -> str:
        """
        Export formatted data to various formats.
        
        Args:
            formatted_inputs: List of formatted inputs to export
            output_format: Export format ("json", "csv", "txt")
            
        Returns:
            Formatted data as string
        """
        if output_format == "json":
            data = [input_obj.dict() for input_obj in formatted_inputs]
            return json.dumps(data, indent=2)
        
        elif output_format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow(["question", "solution", "formatted_input", "metadata"])
            
            # Data rows
            for input_obj in formatted_inputs:
                writer.writerow([
                    input_obj.question,
                    input_obj.solution,
                    input_obj.formatted_input,
                    json.dumps(input_obj.metadata)
                ])
            
            return output.getvalue()
        
        elif output_format == "txt":
            lines = []
            for i, input_obj in enumerate(formatted_inputs):
                lines.append(f"=== Example {i+1} ===")
                lines.append(input_obj.formatted_input)
                lines.append("")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")


# Example usage and testing
def test_rlt_formatter():
    """Test function for RLT formatter"""
    # Initialize formatter
    formatter = RLTFormatter()
    
    # Test data
    question = "What is the derivative of x^2?"
    solution = "The derivative of x^2 is 2x"
    
    # Test input formatting
    formatted_input = formatter.format_question_solution_input(question, solution)
    print("Formatted Input:")
    print(formatted_input.formatted_input)
    print()
    
    # Simulate teacher output
    teacher_output = f"""
    <think>
    To find the derivative of x^2, I need to apply the power rule. 
    The power rule states that for x^n, the derivative is n*x^(n-1).
    For x^2, n=2, so the derivative is 2*x^(2-1) = 2*x^1 = 2x.
    </think>
    <solution>
    The derivative of x^2 is 2x
    </solution>
    """
    
    # Test output parsing
    parsed_output = formatter.parse_rlt_output(teacher_output)
    print("Parsed Output:")
    print(f"Think Content: {parsed_output.think_content}")
    print(f"Solution Content: {parsed_output.solution_content}")
    print(f"Is Valid: {parsed_output.is_valid}")
    print(f"Quality Score: {parsed_output.explanation_quality_score:.3f}")
    print()
    
    # Test distillation prompt creation
    think_tokens = formatter.extract_think_tokens(teacher_output)
    distillation_prompt = formatter.create_student_distillation_prompt(
        question, think_tokens, solution, "enhanced"
    )
    
    print("Student Distillation Prompt:")
    print(distillation_prompt.formatted_prompt)
    print()
    
    # Statistics
    stats = formatter.get_formatting_statistics()
    print("Formatting Statistics:")
    print(json.dumps(stats, indent=2))
    
    return formatter


if __name__ == "__main__":
    test_rlt_formatter()