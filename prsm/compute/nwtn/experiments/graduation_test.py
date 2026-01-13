"""
NWTN Graduation Test: Molecular Assembler Optimization Challenge (MAOC)
======================================================================

Forces the Cross-Core Gating Network (CCGN) to specialize:
1. SSM: Long-range context (Assembly Blueprints)
2. SANM: High-precision logic (Atomic Sequence Planning)
3. FSMN: Low-latency signal processing (Thermal Vibration Streams)
"""

import torch
import random
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GraduationSample:
    input_data: torch.Tensor
    target_labels: torch.Tensor
    expected_core: str # 'ssm', 'sanm', or 'fsmn'
    description: str

class MAOCCurriculum:
    """
    Generates training data that forces core specialization.
    """
    def __init__(self, d_model: int = 512):
        self.d_model = d_model

    def generate_blueprint_context(self, seq_len: int = 1024) -> GraduationSample:
        """Forces SSM: Low-entropy, repetitive structural data"""
        # Simulated 'blueprint' embedding (structured patterns)
        data = torch.zeros(1, seq_len, self.d_model)
        for i in range(0, seq_len, 16):
            data[0, i:i+4, :] = 1.0 # Repetitive 'header' tokens
            
        return GraduationSample(
            input_data=data,
            target_labels=torch.tensor([0]), # Class 0: SSM
            expected_core="ssm",
            description="Long-range Assembler Blueprint"
        )

    def generate_logic_gate(self, seq_len: int = 128) -> GraduationSample:
        """Forces SANM: High-entropy, complex branching logic"""
        # Simulated 'logic' embedding (random but correlated pairs)
        data = torch.randn(1, seq_len, self.d_model) * 2.0
        
        return GraduationSample(
            input_data=data,
            target_labels=torch.tensor([1]), # Class 1: SANM
            expected_core="sanm",
            description="Atomic Synthesis Logic Gate"
        )

    def generate_sensor_stream(self, seq_len: int = 512) -> GraduationSample:
        """Forces FSMN: High-frequency oscillatory data (physics)"""
        # Simulated 'vibration' embedding (sine waves + noise)
        t = torch.linspace(0, 10, seq_len)
        signal = torch.sin(t).view(1, seq_len, 1).repeat(1, 1, self.d_model)
        noise = torch.randn(1, seq_len, self.d_model) * 0.1
        data = signal + noise
        
        return GraduationSample(
            input_data=data,
            target_labels=torch.tensor([2]), # Class 2: FSMN
            expected_core="fsmn",
            description="Thermal Vibration Sensor Stream"
        )

def run_graduation_bench():
    """
    Executes the graduation test and trains the CCGN.
    """
    print("🎓 Starting NWTN Graduation Test [MAOC]...")
    from prsm.compute.nwtn.architectures.hybrid_architecture import create_hybrid_nwtn_engine
    
    engine = create_hybrid_nwtn_engine(agent_id="grad_candidate_01")
    curriculum = MAOCCurriculum()
    
    # 1. SETUP TRAINING
    # We only train the GATER, not the cores themselves
    optimizer = torch.optim.Adam(engine.multi_core_stack.gater.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("\n🏋️ Training CCGN for core specialization...")
    for epoch in range(50):
        total_loss = 0
        samples = [
            curriculum.generate_blueprint_context(),
            curriculum.generate_logic_gate(),
            curriculum.generate_sensor_stream()
        ]
        random.shuffle(samples)
        
        for sample in samples:
            optimizer.zero_grad()
            core_weights = engine.multi_core_stack.gater(sample.input_data)
            loss = criterion(core_weights, sample.target_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50 - Loss: {total_loss:.4f}")

    # 2. FINAL GRADUATION TEST
    print("\n🎯 Running Final Graduation Benchmark...")
    final_tests = [
        curriculum.generate_blueprint_context(),
        curriculum.generate_logic_gate(),
        curriculum.generate_sensor_stream()
    ]
    
    pass_count = 0
    for test in final_tests:
        output, core_weights = engine.multi_core_stack(test.input_data)
        predicted_core_idx = torch.argmax(core_weights).item()
        core_names = engine.multi_core_stack.core_names
        selected_name = core_names[predicted_core_idx]
        
        print(f"\nTask: {test.description}")
        print(f"Target Core: {test.expected_core}")
        print(f"CCGN Selected: {selected_name}")
        
        if selected_name == test.expected_core:
            print("✅ PASS: Specialized!")
            pass_count += 1
        else:
            print("❌ FAIL: Still confused.")

    if pass_count == len(final_tests):
        print("\n🎓 GRADUATION SUCCESS! NWTN is battle-ready for APM.")
    else:
        print("\n⚠️ GRADUATION FAILED: Suboptimal specialization.")

if __name__ == "__main__":
    run_graduation_bench()
