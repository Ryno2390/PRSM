#!/usr/bin/env python3
"""
Identify Key Physics Topics for WorldModelCore
==============================================

This script identifies the most essential physics topics based on:
1. Fundamental importance (laws, principles, constants)
2. Educational curriculum standards
3. Cross-disciplinary relevance
4. Certainty level (well-established vs. theoretical)
5. Mathematical precision
"""

import json
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass

@dataclass
class PhysicsTopic:
    """Represents a physics topic with importance metrics"""
    name: str
    wikipedia_variations: List[str]
    importance_score: float
    certainty_level: float
    category: str
    mathematical_precision: bool
    educational_level: str
    cross_disciplinary: bool
    reasoning_value: str
    
def get_fundamental_laws_and_principles() -> List[PhysicsTopic]:
    """Core physics laws and principles - highest priority"""
    return [
        PhysicsTopic(
            name="Newton's laws of motion",
            wikipedia_variations=["Newton's laws of motion", "Newtons laws of motion", "Newton's_laws_of_motion"],
            importance_score=10.0,
            certainty_level=0.9999,
            category="classical_mechanics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Fundamental for all mechanical reasoning"
        ),
        PhysicsTopic(
            name="Conservation of energy",
            wikipedia_variations=["Conservation of energy", "Energy conservation", "Conservation_of_energy"],
            importance_score=10.0,
            certainty_level=0.9999,
            category="thermodynamics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Universal principle across all physics"
        ),
        PhysicsTopic(
            name="Conservation of momentum",
            wikipedia_variations=["Conservation of momentum", "Momentum conservation", "Conservation_of_momentum"],
            importance_score=9.8,
            certainty_level=0.9999,
            category="classical_mechanics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Essential for collision and motion analysis"
        ),
        PhysicsTopic(
            name="First law of thermodynamics",
            wikipedia_variations=["First law of thermodynamics", "First_law_of_thermodynamics"],
            importance_score=9.5,
            certainty_level=0.9999,
            category="thermodynamics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Energy conservation in thermal systems"
        ),
        PhysicsTopic(
            name="Second law of thermodynamics",
            wikipedia_variations=["Second law of thermodynamics", "Second_law_of_thermodynamics"],
            importance_score=9.5,
            certainty_level=0.9999,
            category="thermodynamics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Entropy and irreversibility"
        ),
        PhysicsTopic(
            name="Coulomb's law",
            wikipedia_variations=["Coulomb's law", "Coulombs law", "Coulomb's_law"],
            importance_score=9.0,
            certainty_level=0.9999,
            category="electromagnetism",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Fundamental electrostatic force law"
        ),
        PhysicsTopic(
            name="Ohm's law",
            wikipedia_variations=["Ohm's law", "Ohms law", "Ohm's_law"],
            importance_score=8.5,
            certainty_level=0.9999,
            category="electromagnetism",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Basic electrical circuit analysis"
        ),
        PhysicsTopic(
            name="Ideal gas law",
            wikipedia_variations=["Ideal gas law", "Ideal_gas_law", "Perfect gas law"],
            importance_score=8.5,
            certainty_level=0.999,
            category="thermodynamics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Gas behavior under standard conditions"
        ),
        PhysicsTopic(
            name="Hooke's law",
            wikipedia_variations=["Hooke's law", "Hookes law", "Hooke's_law"],
            importance_score=8.0,
            certainty_level=0.9999,
            category="classical_mechanics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Elastic deformation and springs"
        ),
        PhysicsTopic(
            name="Wave equation",
            wikipedia_variations=["Wave equation", "Wave_equation"],
            importance_score=8.0,
            certainty_level=0.9999,
            category="wave_mechanics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Wave propagation in all media"
        )
    ]

def get_fundamental_constants() -> List[PhysicsTopic]:
    """Fundamental physical constants - very high certainty"""
    return [
        PhysicsTopic(
            name="Speed of light",
            wikipedia_variations=["Speed of light", "Speed_of_light", "Light speed"],
            importance_score=9.8,
            certainty_level=0.9999,
            category="fundamental_constants",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Universal constant in relativity"
        ),
        PhysicsTopic(
            name="Planck constant",
            wikipedia_variations=["Planck constant", "Planck's constant", "Planck_constant"],
            importance_score=9.5,
            certainty_level=0.9999,
            category="fundamental_constants",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Quantum mechanics fundamental"
        ),
        PhysicsTopic(
            name="Gravitational constant",
            wikipedia_variations=["Gravitational constant", "Gravitational_constant", "Newton's gravitational constant"],
            importance_score=9.0,
            certainty_level=0.999,
            category="fundamental_constants",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Universal gravitation strength"
        ),
        PhysicsTopic(
            name="Elementary charge",
            wikipedia_variations=["Elementary charge", "Elementary_charge", "Electron charge"],
            importance_score=8.5,
            certainty_level=0.9999,
            category="fundamental_constants",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Basic unit of electric charge"
        ),
        PhysicsTopic(
            name="Electron mass",
            wikipedia_variations=["Electron mass", "Electron_mass", "Mass of electron"],
            importance_score=8.0,
            certainty_level=0.9999,
            category="fundamental_constants",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Fundamental particle mass"
        )
    ]

def get_key_concepts() -> List[PhysicsTopic]:
    """Key physics concepts - high importance for reasoning"""
    return [
        PhysicsTopic(
            name="Force",
            wikipedia_variations=["Force", "Force (physics)"],
            importance_score=9.5,
            certainty_level=0.9999,
            category="classical_mechanics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Fundamental concept in mechanics"
        ),
        PhysicsTopic(
            name="Energy",
            wikipedia_variations=["Energy", "Energy (physics)"],
            importance_score=9.5,
            certainty_level=0.9999,
            category="general_physics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Universal concept across all physics"
        ),
        PhysicsTopic(
            name="Momentum",
            wikipedia_variations=["Momentum", "Linear momentum"],
            importance_score=9.0,
            certainty_level=0.9999,
            category="classical_mechanics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Fundamental quantity in mechanics"
        ),
        PhysicsTopic(
            name="Mass",
            wikipedia_variations=["Mass", "Mass (physics)"],
            importance_score=9.0,
            certainty_level=0.9999,
            category="classical_mechanics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Basic property of matter"
        ),
        PhysicsTopic(
            name="Acceleration",
            wikipedia_variations=["Acceleration", "Acceleration (physics)"],
            importance_score=8.5,
            certainty_level=0.9999,
            category="classical_mechanics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Rate of change of velocity"
        ),
        PhysicsTopic(
            name="Velocity",
            wikipedia_variations=["Velocity", "Velocity (physics)"],
            importance_score=8.0,
            certainty_level=0.9999,
            category="classical_mechanics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Rate of change of position"
        ),
        PhysicsTopic(
            name="Gravity",
            wikipedia_variations=["Gravity", "Gravitation", "Gravitational force"],
            importance_score=9.0,
            certainty_level=0.9999,
            category="classical_mechanics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Fundamental force in nature"
        ),
        PhysicsTopic(
            name="Electric field",
            wikipedia_variations=["Electric field", "Electric_field"],
            importance_score=8.5,
            certainty_level=0.9999,
            category="electromagnetism",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Field concept in electromagnetism"
        ),
        PhysicsTopic(
            name="Magnetic field",
            wikipedia_variations=["Magnetic field", "Magnetic_field"],
            importance_score=8.5,
            certainty_level=0.9999,
            category="electromagnetism",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Field concept in magnetism"
        ),
        PhysicsTopic(
            name="Temperature",
            wikipedia_variations=["Temperature", "Temperature (physics)"],
            importance_score=8.0,
            certainty_level=0.9999,
            category="thermodynamics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Measure of thermal energy"
        ),
        PhysicsTopic(
            name="Pressure",
            wikipedia_variations=["Pressure", "Pressure (physics)"],
            importance_score=8.0,
            certainty_level=0.9999,
            category="thermodynamics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Force per unit area"
        ),
        PhysicsTopic(
            name="Work (physics)",
            wikipedia_variations=["Work (physics)", "Work", "Mechanical work"],
            importance_score=8.0,
            certainty_level=0.9999,
            category="classical_mechanics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Energy transfer concept"
        ),
        PhysicsTopic(
            name="Power (physics)",
            wikipedia_variations=["Power (physics)", "Power", "Mechanical power"],
            importance_score=7.5,
            certainty_level=0.9999,
            category="classical_mechanics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Rate of energy transfer"
        )
    ]

def get_advanced_topics() -> List[PhysicsTopic]:
    """Advanced but essential topics"""
    return [
        PhysicsTopic(
            name="Special relativity",
            wikipedia_variations=["Special relativity", "Special_relativity"],
            importance_score=9.0,
            certainty_level=0.9999,
            category="relativity",
            mathematical_precision=True,
            educational_level="graduate",
            cross_disciplinary=True,
            reasoning_value="Space-time relationship"
        ),
        PhysicsTopic(
            name="Quantum mechanics",
            wikipedia_variations=["Quantum mechanics", "Quantum_mechanics"],
            importance_score=9.0,
            certainty_level=0.999,
            category="quantum_physics",
            mathematical_precision=True,
            educational_level="graduate",
            cross_disciplinary=True,
            reasoning_value="Microscopic world behavior"
        ),
        PhysicsTopic(
            name="Maxwell's equations",
            wikipedia_variations=["Maxwell's equations", "Maxwells equations", "Maxwell's_equations"],
            importance_score=9.0,
            certainty_level=0.9999,
            category="electromagnetism",
            mathematical_precision=True,
            educational_level="graduate",
            cross_disciplinary=True,
            reasoning_value="Unified electromagnetic theory"
        ),
        PhysicsTopic(
            name="Schrödinger equation",
            wikipedia_variations=["Schrödinger equation", "Schrodinger equation", "Schrödinger_equation"],
            importance_score=8.5,
            certainty_level=0.999,
            category="quantum_physics",
            mathematical_precision=True,
            educational_level="graduate",
            cross_disciplinary=True,
            reasoning_value="Quantum state evolution"
        ),
        PhysicsTopic(
            name="Uncertainty principle",
            wikipedia_variations=["Uncertainty principle", "Heisenberg uncertainty principle", "Uncertainty_principle"],
            importance_score=8.5,
            certainty_level=0.999,
            category="quantum_physics",
            mathematical_precision=True,
            educational_level="graduate",
            cross_disciplinary=True,
            reasoning_value="Fundamental quantum limitation"
        )
    ]

def get_specialized_laws() -> List[PhysicsTopic]:
    """Specialized but important laws"""
    return [
        PhysicsTopic(
            name="Faraday's law",
            wikipedia_variations=["Faraday's law", "Faradays law", "Faraday's_law"],
            importance_score=7.5,
            certainty_level=0.9999,
            category="electromagnetism",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Electromagnetic induction"
        ),
        PhysicsTopic(
            name="Lenz's law",
            wikipedia_variations=["Lenz's law", "Lenzs law", "Lenz's_law"],
            importance_score=7.0,
            certainty_level=0.9999,
            category="electromagnetism",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Direction of induced current"
        ),
        PhysicsTopic(
            name="Boyle's law",
            wikipedia_variations=["Boyle's law", "Boyles law", "Boyle's_law"],
            importance_score=7.0,
            certainty_level=0.9999,
            category="thermodynamics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Pressure-volume relationship"
        ),
        PhysicsTopic(
            name="Charles's law",
            wikipedia_variations=["Charles's law", "Charles law", "Charles's_law"],
            importance_score=7.0,
            certainty_level=0.9999,
            category="thermodynamics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Volume-temperature relationship"
        ),
        PhysicsTopic(
            name="Stefan-Boltzmann law",
            wikipedia_variations=["Stefan-Boltzmann law", "Stefan Boltzmann law", "Stefan-Boltzmann_law"],
            importance_score=7.0,
            certainty_level=0.9999,
            category="thermodynamics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Blackbody radiation"
        ),
        PhysicsTopic(
            name="Doppler effect",
            wikipedia_variations=["Doppler effect", "Doppler_effect"],
            importance_score=7.0,
            certainty_level=0.9999,
            category="wave_mechanics",
            mathematical_precision=True,
            educational_level="undergraduate",
            cross_disciplinary=True,
            reasoning_value="Wave frequency changes"
        )
    ]

def analyze_and_rank_topics() -> List[PhysicsTopic]:
    """Combine all topics and rank by importance"""
    
    all_topics = (
        get_fundamental_laws_and_principles() +
        get_fundamental_constants() +
        get_key_concepts() +
        get_advanced_topics() +
        get_specialized_laws()
    )
    
    # Sort by importance score (descending)
    ranked_topics = sorted(all_topics, key=lambda x: x.importance_score, reverse=True)
    
    return ranked_topics

def main():
    """Main analysis function"""
    
    print("🔬 Essential Physics Topics Analysis")
    print("=" * 60)
    
    # Get all topics
    ranked_topics = analyze_and_rank_topics()
    
    print(f"📊 Total topics identified: {len(ranked_topics)}")
    
    # Analyze by category
    categories = {}
    for topic in ranked_topics:
        if topic.category not in categories:
            categories[topic.category] = []
        categories[topic.category].append(topic)
    
    print(f"\n📁 Topics by category:")
    for category, topics in categories.items():
        print(f"   {category}: {len(topics)} topics")
    
    # Show top priorities
    print(f"\n⭐ Top 20 Essential Physics Topics:")
    print(f"{'Rank':<4} {'Topic':<35} {'Score':<6} {'Certainty':<10} {'Category':<20}")
    print("-" * 80)
    
    for i, topic in enumerate(ranked_topics[:20]):
        print(f"{i+1:<4} {topic.name:<35} {topic.importance_score:<6} {topic.certainty_level:<10} {topic.category:<20}")
    
    # Create extraction lists
    top_tier = [t for t in ranked_topics if t.importance_score >= 9.0]
    high_priority = [t for t in ranked_topics if t.importance_score >= 8.0]
    all_essential = ranked_topics
    
    print(f"\n📋 Extraction Recommendations:")
    print(f"   Top Tier (score ≥ 9.0): {len(top_tier)} topics (~10-15 minutes)")
    print(f"   High Priority (score ≥ 8.0): {len(high_priority)} topics (~20-30 minutes)")
    print(f"   All Essential: {len(all_essential)} topics (~45-60 minutes)")
    
    # Save extraction lists
    output_dir = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/physics/"
    
    # Top tier list
    top_tier_list = []
    for topic in top_tier:
        top_tier_list.extend(topic.wikipedia_variations)
    
    with open(output_dir + "top_tier_topics.json", 'w') as f:
        json.dump(top_tier_list, f, indent=2)
    
    # High priority list
    high_priority_list = []
    for topic in high_priority:
        high_priority_list.extend(topic.wikipedia_variations)
    
    with open(output_dir + "high_priority_topics.json", 'w') as f:
        json.dump(high_priority_list, f, indent=2)
    
    # All essential list
    all_essential_list = []
    for topic in all_essential:
        all_essential_list.extend(topic.wikipedia_variations)
    
    with open(output_dir + "all_essential_topics.json", 'w') as f:
        json.dump(all_essential_list, f, indent=2)
    
    # Detailed analysis
    analysis = {
        'total_topics': len(ranked_topics),
        'categories': {cat: len(topics) for cat, topics in categories.items()},
        'top_tier': [
            {
                'name': topic.name,
                'variations': topic.wikipedia_variations,
                'importance_score': topic.importance_score,
                'certainty_level': topic.certainty_level,
                'category': topic.category,
                'reasoning_value': topic.reasoning_value
            }
            for topic in top_tier
        ],
        'high_priority': [
            {
                'name': topic.name,
                'variations': topic.wikipedia_variations,
                'importance_score': topic.importance_score,
                'certainty_level': topic.certainty_level,
                'category': topic.category,
                'reasoning_value': topic.reasoning_value
            }
            for topic in high_priority
        ]
    }
    
    with open(output_dir + "physics_topics_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n💾 Files saved to: {output_dir}")
    print(f"   - top_tier_topics.json ({len(top_tier_list)} variations)")
    print(f"   - high_priority_topics.json ({len(high_priority_list)} variations)")
    print(f"   - all_essential_topics.json ({len(all_essential_list)} variations)")
    print(f"   - physics_topics_analysis.json (detailed analysis)")
    
    print(f"\n✅ Ready for targeted extraction!")

if __name__ == "__main__":
    main()