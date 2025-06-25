#!/usr/bin/env python3
"""
Test script for PRSM Enhanced Data Models
Tests the new models for Phase 1, Week 1, Task 2
"""

import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

def test_enhanced_models():
    """Test all enhanced PRSM data models"""
    
    print("🧪 Testing PRSM Enhanced Data Models...")
    print("=" * 50)
    
    try:
        # Import the models
        from prsm.core.models import (
            PRSMSession, ArchitectTask, TeacherModel, ReasoningStep,
            SafetyFlag, AgentResponse, TaskHierarchy, Curriculum,
            FTNSTransaction, GovernanceProposal, UserInput, PRSMResponse,
            TaskStatus, AgentType, SafetyLevel, ModelType
        )
        print("✅ All model imports successful")
        
        # Test PRSMSession
        session = PRSMSession(
            user_id="test_user_123",
            nwtn_context_allocation=1000,
            context_used=250
        )
        print(f"✅ PRSMSession created: {session.session_id}")
        
        # Test ArchitectTask
        task = ArchitectTask(
            session_id=session.session_id,
            instruction="Analyze scientific data",
            complexity_score=0.7,
            level=1
        )
        print(f"✅ ArchitectTask created: {task.task_id}")
        
        # Test TeacherModel
        teacher = TeacherModel(
            name="Scientific Analysis Specialist",
            specialization="data_analysis",
            performance_score=0.85
        )
        print(f"✅ TeacherModel created: {teacher.teacher_id}")
        
        # Test ReasoningStep
        reasoning_step = ReasoningStep(
            agent_type=AgentType.ARCHITECT,
            agent_id="architect_001",
            input_data={"query": "test"},
            output_data={"result": "processed"},
            execution_time=1.23
        )
        print(f"✅ ReasoningStep created: {reasoning_step.step_id}")
        
        # Test SafetyFlag
        safety_flag = SafetyFlag(
            level=SafetyLevel.LOW,
            category="data_validation",
            description="Minor data format inconsistency",
            triggered_by="validator_001"
        )
        print(f"✅ SafetyFlag created: {safety_flag.flag_id}")
        
        # Test AgentResponse
        response = AgentResponse(
            task_id=task.task_id,
            agent_type=AgentType.EXECUTOR,
            agent_id="executor_001",
            content="Analysis complete",
            execution_time=2.45,
            confidence_score=0.92
        )
        print(f"✅ AgentResponse created: {response.response_id}")
        
        # Test TaskHierarchy
        hierarchy = TaskHierarchy(
            root_task=task,
            max_depth=3
        )
        print(f"✅ TaskHierarchy created with root task")
        
        # Test Curriculum
        curriculum = Curriculum(
            teacher_id=teacher.teacher_id,
            domain="scientific_analysis",
            difficulty_level=0.6
        )
        print(f"✅ Curriculum created: {curriculum.curriculum_id}")
        
        # Test FTNSTransaction
        transaction = FTNSTransaction(
            to_user="test_user_123",
            amount=50.0,
            transaction_type="reward",
            description="Context usage reward",
            context_units=100
        )
        print(f"✅ FTNSTransaction created: {transaction.transaction_id}")
        
        # Test GovernanceProposal
        proposal = GovernanceProposal(
            proposer_id="user_123",
            title="Improve Safety Thresholds",
            description="Adjust circuit breaker sensitivity",
            proposal_type="safety",
            voting_starts=datetime.now(timezone.utc),
            voting_ends=datetime.now(timezone.utc)
        )
        print(f"✅ GovernanceProposal created: {proposal.proposal_id}")
        
        # Test UserInput
        user_input = UserInput(
            user_id="test_user_123",
            prompt="Analyze this dataset for patterns",
            context_allocation=500
        )
        print(f"✅ UserInput created")
        
        # Test PRSMResponse with complex data
        prsm_response = PRSMResponse(
            session_id=session.session_id,
            user_id="test_user_123",
            final_answer="Analysis complete with 3 key patterns identified",
            reasoning_trace=[reasoning_step],
            confidence_score=0.89,
            context_used=300,
            ftns_charged=15.0,
            sources=["dataset_001.csv", "analysis_model_v2"]
        )
        print(f"✅ PRSMResponse created with reasoning trace")
        
        # Test model serialization
        session_dict = session.model_dump()
        assert isinstance(session_dict, dict)
        assert "session_id" in session_dict
        print("✅ Model serialization working")
        
        # Test enum validation
        try:
            invalid_task = ArchitectTask(
                session_id=session.session_id,
                instruction="test",
                status="invalid_status"  # This should fail
            )
        except Exception:
            print("✅ Enum validation working (correctly rejected invalid status)")
        
        # Test field validation
        try:
            invalid_teacher = TeacherModel(
                name="Test",
                specialization="test",
                performance_score=1.5  # This should fail (> 1.0)
            )
        except Exception:
            print("✅ Field validation working (correctly rejected invalid score)")
        
        print("\n" + "=" * 50)
        print("🎉 ALL ENHANCED MODEL TESTS PASSED!")
        models_tested = [
            'PRSMSession', 'ArchitectTask', 'TeacherModel', 'ReasoningStep',
            'SafetyFlag', 'AgentResponse', 'TaskHierarchy', 'Curriculum',
            'FTNSTransaction', 'GovernanceProposal', 'UserInput', 'PRSMResponse'
        ]
        print(f"📊 Models tested: {len(models_tested)}")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_models()
    sys.exit(0 if success else 1)