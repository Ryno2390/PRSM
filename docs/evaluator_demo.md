# Checkpoint Evaluator – Adversarial Assessment

The CheckpointEvaluator adds adversarial evaluation to NWTN's checkpoint synthesis lifecycle.

## Overview

**Problem:** The Live Scribe coordinates but doesn't challenge agent work. Per Anthropic's harness article: "out of the box, Claude is a poor QA agent." Evaluators need to be tuned to be skeptical.

**Solution:** The Scribe now serves as the **evaluator** during checkpoint synthesis — not just assembling the narrative, but critically assessing each agent's work against MetaPlan criteria.

## Key Components

### 1. EvaluationResult Dataclass
```python
@dataclass
class EvaluationResult:
    agent_id: str
    milestone_index: int
    criteria_met: Dict[str, bool]  # acceptance criteria → met/not met
    quality_score: float           # 0.0-1.0
    issues_found: List[str]        # specific problems identified
    confidence: float              # evaluator's confidence
    divergence_notes: str          # where evaluator disagrees with agent self-assessment
    llm_assisted: bool = False
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def passed(self) -> bool:
        return self.quality_score >= 0.6
```

### 2. CheckpointEvaluator Class
- **Skeptical by default** — assumes work is incomplete until proven otherwise
- **Heuristic fallback** — works without LLM backend (score based on surprise, entry count, keyword matching)
- **LLM-assisted** — when backend available, uses structured prompts for nuanced assessment
- **Divergence logging** — tracks where evaluator's judgment differs from agent claims (P3 tuning)
- **Criteria prompt tuning** — allows operators to refine evaluator prompts for specific agent/criterion pairs

### 3. EvaluationBatch
- Aggregates multiple agent evaluations
- Team-level statistics (`team_quality_score()`, `all_passed`, `passed_agents`, `failed_agents`)
- Narrative block for checkpoint synthesis output

## Integration with LiveScribe

The evaluator is wired into `CheckpointLifecycleManager.initiate_checkpoint()`:

1. **Before synthesis:** Evaluate each agent's work against milestone criteria
2. **Embed results:** Inject evaluation summary into synthesis narrative  
3. **Notify failures:** Send IMPORTANT inbox updates to agents whose work didn't meet criteria
4. **Log divergence:** Store evaluation history for P3 tuning loop

## P3 Tuning Loop Interface

```python
# Human operator can review systematic errors
history = scribe.review_evaluation_history()
# history = [
#   {
#     "agent_id": "agent/coder",
#     "milestone_index": 0,
#     "quality_score": 0.3,
#     "divergence_notes": "Agent claimed completion but no evidence of edge case testing",
#     ...
#   }
# ]

# Tune evaluator prompts for specific agents or criteria
scribe.update_evaluation_criterion_prompt(
    agent_id="agent/coder",
    criterion="All unit tests pass with coverage above 80%",
    new_prompt="Look for explicit coverage percentage mentioned (e.g., 85%) and test count",
)
```

## Skepticism Heuristics

### Default Assumptions
1. **No entries → score = 0.0** — no evidence of work
2. **Few entries (<3) → score reduced 20%** — completeness unclear
3. **No matching keywords → criteria not met** — agent may claim "done" but lacks specific evidence
4. **Contradiction signals → quality penalty** — entries that contradict each other

### LLM Evaluation (when available)
Structured prompt template:
```
You are an adversarial evaluator for scientific AI agents.

AGENT: {agent_id}
MILESTONE: {milestone.title}

CRITERIA:
{criterion1}
{criterion2}

WHITEBOARD ENTRIES:
1. [coder] (surprise=0.65) Wrote unit tests with 85% coverage
2. [coder] (surprise=0.45) Edge cases handled for input validation
...

ASSESS each criterion:
1. [criterion1]: true/false
2. [criterion2]: true/false
...

OVERALL quality score (0.0-1.0): 
ISSUES found:
DIVERGENCE notes:
CONFIDENCE (0.0-1.0):
```

## Usage Example

```python
# Basic usage
evaluator = CheckpointEvaluator(meta_plan=plan, backend_registry=backend)

# Evaluate single agent
result = await evaluator.evaluate_agent_work(
    agent_id="agent/coder",
    checkpoint_entries=whiteboard_entries,
    milestone=current_milestone,
)

# Evaluate entire team
batch = await evaluator.evaluate_team(
    agent_entries={
        "agent/coder": coder_entries,
        "agent/researcher": researcher_entries,
    },
    milestone=current_milestone,
)

# Check results
print(f"Team passed: {batch.all_passed}")
print(f"Team score: {batch.team_quality_score():.2f}")
print(f"Failed agents: {batch.failed_agents}")

# Add to checkpoint narrative
narrative = synthesizer.synthesize(...) + batch.to_narrative_block()

# Tune evaluator based on observed divergence
if result.divergence_notes:
    print(f"Divergence: {result.divergence_notes}")
    # Human operator can update prompts if evaluator systematically misjudges
    evaluator.update_criteria_prompt("agent/coder", "criterion", "improved prompt")
```

## Benefits

1. **Early feedback** — Agents learn about gaps before next milestone
2. **Adversarial rigor** — Prevents premature convergence on suboptimal solutions
3. **Tunable skepticism** — Operators can adjust evaluator strictness per agent/criterion
4. **Divergence logging** — Enables P3-style human-in-the-loop tuning
5. **Backwards compatible** — Works with existing LiveScribe checkpoint workflow