#!/usr/bin/env python3
"""
Empirical Constants Curation for NWTN WorldModelCore
====================================================

Creates essential empirical constants knowledge items manually from authoritative sources
to expand the WorldModelCore with fundamental physical and mathematical constants.
"""

import json
import os
from typing import Dict, List, Any

def create_essential_empirical_constants() -> List[Dict[str, Any]]:
    """
    Create essential empirical constants knowledge items from authoritative sources
    
    Returns:
        List of essential empirical constants knowledge items
    """
    
    essential_knowledge = [
        # Fundamental Physical Constants
        {
            'content': 'The speed of light in vacuum is exactly 299,792,458 meters per second',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'fundamental_physics',
            'mathematical_form': 'c = 299,792,458 m/s',
            'dependencies': [],
            'references': ['NIST', 'SI Definition', 'Physics textbooks'],
            'applicable_conditions': ['vacuum', 'relativity'],
            'principle_name': 'Speed of Light'
        },
        {
            'content': 'Planck constant relates energy and frequency of electromagnetic radiation',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'fundamental_physics',
            'mathematical_form': 'h = 6.62607015×10⁻³⁴ J⋅s',
            'dependencies': [],
            'references': ['NIST', 'Quantum Mechanics', 'Physics textbooks'],
            'applicable_conditions': ['quantum mechanics', 'electromagnetic radiation'],
            'principle_name': 'Planck Constant'
        },
        {
            'content': 'Elementary charge is the electric charge of a single proton',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'fundamental_physics',
            'mathematical_form': 'e = 1.602176634×10⁻¹⁹ C',
            'dependencies': [],
            'references': ['NIST', 'Electromagnetism', 'Physics textbooks'],
            'applicable_conditions': ['electromagnetic interactions', 'atomic physics'],
            'principle_name': 'Elementary Charge'
        },
        {
            'content': 'Gravitational constant determines the strength of gravitational attraction',
            'certainty': 0.999,
            'domain': 'empirical_constants',
            'category': 'fundamental_physics',
            'mathematical_form': 'G = 6.67430×10⁻¹¹ m³⋅kg⁻¹⋅s⁻²',
            'dependencies': [],
            'references': ['NIST', 'Gravitational Physics', 'Physics textbooks'],
            'applicable_conditions': ['gravity', 'mass interactions'],
            'principle_name': 'Gravitational Constant'
        },
        {
            'content': 'Boltzmann constant relates temperature to kinetic energy',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'fundamental_physics',
            'mathematical_form': 'kB = 1.380649×10⁻²³ J/K',
            'dependencies': [],
            'references': ['NIST', 'Statistical Mechanics', 'Thermodynamics textbooks'],
            'applicable_conditions': ['thermodynamics', 'statistical mechanics'],
            'principle_name': 'Boltzmann Constant'
        },
        {
            'content': 'Fine structure constant characterizes the strength of electromagnetic interaction',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'fundamental_physics',
            'mathematical_form': 'α = 7.2973525693×10⁻³ ≈ 1/137',
            'dependencies': [],
            'references': ['NIST', 'Quantum Electrodynamics', 'Physics textbooks'],
            'applicable_conditions': ['quantum electrodynamics', 'atomic physics'],
            'principle_name': 'Fine Structure Constant'
        },
        {
            'content': 'Avogadro constant defines the number of particles in one mole',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'fundamental_physics',
            'mathematical_form': 'NA = 6.02214076×10²³ mol⁻¹',
            'dependencies': [],
            'references': ['NIST', 'Chemistry textbooks', 'Statistical Mechanics'],
            'applicable_conditions': ['chemistry', 'molecular physics'],
            'principle_name': 'Avogadro Constant'
        },
        
        # Mathematical Constants
        {
            'content': 'Pi is the ratio of a circle\'s circumference to its diameter',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'mathematical',
            'mathematical_form': 'π = 3.1415926535897932384626433832795...',
            'dependencies': [],
            'references': ['Mathematical constants', 'Geometry textbooks'],
            'applicable_conditions': ['geometry', 'trigonometry', 'calculus'],
            'principle_name': 'Pi'
        },
        {
            'content': 'Euler\'s number is the base of natural logarithm',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'mathematical',
            'mathematical_form': 'e = 2.7182818284590452353602874713527...',
            'dependencies': [],
            'references': ['Mathematical constants', 'Calculus textbooks'],
            'applicable_conditions': ['calculus', 'exponential functions', 'complex analysis'],
            'principle_name': 'Euler\'s Number'
        },
        {
            'content': 'Golden ratio appears in nature and art as aesthetically pleasing proportion',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'mathematical',
            'mathematical_form': 'φ = (1 + √5)/2 = 1.6180339887...',
            'dependencies': [],
            'references': ['Mathematical constants', 'Fibonacci sequence', 'Geometry'],
            'applicable_conditions': ['geometry', 'number theory', 'art'],
            'principle_name': 'Golden Ratio'
        },
        {
            'content': 'Square root of 2 is the ratio of diagonal to side in a square',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'mathematical',
            'mathematical_form': '√2 = 1.4142135623730950488016887242097...',
            'dependencies': [],
            'references': ['Mathematical constants', 'Geometry textbooks'],
            'applicable_conditions': ['geometry', 'Pythagorean theorem'],
            'principle_name': 'Square Root of 2'
        },
        
        # Atomic and Molecular Constants
        {
            'content': 'Electron rest mass is the invariant mass of an electron',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'atomic_physics',
            'mathematical_form': 'me = 9.1093837015×10⁻³¹ kg',
            'dependencies': [],
            'references': ['NIST', 'Atomic Physics', 'Particle Physics textbooks'],
            'applicable_conditions': ['atomic physics', 'particle physics'],
            'principle_name': 'Electron Mass'
        },
        {
            'content': 'Proton rest mass is the invariant mass of a proton',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'atomic_physics',
            'mathematical_form': 'mp = 1.67262192369×10⁻²⁷ kg',
            'dependencies': [],
            'references': ['NIST', 'Atomic Physics', 'Particle Physics textbooks'],
            'applicable_conditions': ['atomic physics', 'nuclear physics'],
            'principle_name': 'Proton Mass'
        },
        {
            'content': 'Neutron rest mass is the invariant mass of a neutron',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'atomic_physics',
            'mathematical_form': 'mn = 1.67492749804×10⁻²⁷ kg',
            'dependencies': [],
            'references': ['NIST', 'Atomic Physics', 'Nuclear Physics textbooks'],
            'applicable_conditions': ['atomic physics', 'nuclear physics'],
            'principle_name': 'Neutron Mass'
        },
        {
            'content': 'Atomic mass unit is one twelfth of the mass of carbon-12 atom',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'atomic_physics',
            'mathematical_form': 'u = 1.66053906660×10⁻²⁷ kg',
            'dependencies': [],
            'references': ['NIST', 'Atomic Physics', 'Chemistry textbooks'],
            'applicable_conditions': ['atomic physics', 'chemistry'],
            'principle_name': 'Atomic Mass Unit'
        },
        {
            'content': 'Rydberg constant relates to the wavelengths of spectral lines',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'atomic_physics',
            'mathematical_form': 'R∞ = 1.0973731568160×10⁷ m⁻¹',
            'dependencies': [],
            'references': ['NIST', 'Atomic Spectroscopy', 'Quantum Mechanics textbooks'],
            'applicable_conditions': ['atomic spectroscopy', 'quantum mechanics'],
            'principle_name': 'Rydberg Constant'
        },
        
        # Thermodynamic Constants
        {
            'content': 'Universal gas constant relates energy and temperature for ideal gases',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'thermodynamics',
            'mathematical_form': 'R = 8.314462618 J⋅mol⁻¹⋅K⁻¹',
            'dependencies': [],
            'references': ['NIST', 'Thermodynamics textbooks', 'Physical Chemistry'],
            'applicable_conditions': ['thermodynamics', 'ideal gas law'],
            'principle_name': 'Universal Gas Constant'
        },
        {
            'content': 'Stefan-Boltzmann constant relates energy radiated by black body to temperature',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'thermodynamics',
            'mathematical_form': 'σ = 5.670374419×10⁻⁸ W⋅m⁻²⋅K⁻⁴',
            'dependencies': [],
            'references': ['NIST', 'Thermodynamics textbooks', 'Statistical Mechanics'],
            'applicable_conditions': ['black body radiation', 'thermodynamics'],
            'principle_name': 'Stefan-Boltzmann Constant'
        },
        
        # Electromagnetic Constants
        {
            'content': 'Permittivity of free space characterizes electric field in vacuum',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'electromagnetism',
            'mathematical_form': 'ε₀ = 8.8541878128×10⁻¹² F/m',
            'dependencies': [],
            'references': ['NIST', 'Electromagnetism textbooks', 'Physics textbooks'],
            'applicable_conditions': ['electromagnetism', 'vacuum'],
            'principle_name': 'Permittivity of Free Space'
        },
        {
            'content': 'Permeability of free space characterizes magnetic field in vacuum',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'electromagnetism',
            'mathematical_form': 'μ₀ = 4π×10⁻⁷ H/m',
            'dependencies': [],
            'references': ['NIST', 'Electromagnetism textbooks', 'Physics textbooks'],
            'applicable_conditions': ['electromagnetism', 'vacuum'],
            'principle_name': 'Permeability of Free Space'
        },
        {
            'content': 'Impedance of free space characterizes electromagnetic wave propagation in vacuum',
            'certainty': 0.9999,
            'domain': 'empirical_constants',
            'category': 'electromagnetism',
            'mathematical_form': 'Z₀ = 376.730313668 Ω',
            'dependencies': [],
            'references': ['NIST', 'Electromagnetism textbooks', 'Antenna theory'],
            'applicable_conditions': ['electromagnetic waves', 'vacuum'],
            'principle_name': 'Impedance of Free Space'
        },
        
        # Cosmological Constants
        {
            'content': 'Hubble constant describes the rate of expansion of the universe',
            'certainty': 0.95,
            'domain': 'empirical_constants',
            'category': 'cosmology',
            'mathematical_form': 'H₀ ≈ 70 km⋅s⁻¹⋅Mpc⁻¹',
            'dependencies': [],
            'references': ['Cosmology textbooks', 'Astronomical observations', 'WMAP/Planck'],
            'applicable_conditions': ['cosmology', 'universe expansion'],
            'principle_name': 'Hubble Constant'
        },
        {
            'content': 'Cosmological constant represents the energy density of empty space',
            'certainty': 0.9,
            'domain': 'empirical_constants',
            'category': 'cosmology',
            'mathematical_form': 'Λ ≈ 1.1×10⁻⁵² m⁻²',
            'dependencies': [],
            'references': ['Cosmology textbooks', 'General Relativity', 'Dark energy observations'],
            'applicable_conditions': ['cosmology', 'dark energy', 'general relativity'],
            'principle_name': 'Cosmological Constant'
        }
    ]
    
    return essential_knowledge

def expand_world_model_empirical_constants():
    """Expand the WorldModelCore with essential empirical constants"""
    
    print("🔢 Creating Essential Empirical Constants Knowledge Base")
    print("=" * 60)
    
    # Create essential empirical constants knowledge
    essential_knowledge = create_essential_empirical_constants()
    
    print(f"📚 Created {len(essential_knowledge)} essential empirical constants")
    
    # Organize by category
    categories = {}
    for item in essential_knowledge:
        category = item['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(item)
    
    print(f"\n📊 Knowledge by category:")
    for category, items in categories.items():
        print(f"   {category}: {len(items)} items")
    
    # Calculate average certainty
    avg_certainty = sum(item['certainty'] for item in essential_knowledge) / len(essential_knowledge)
    print(f"\n📈 Average certainty: {avg_certainty:.3f}")
    
    # Save to processed knowledge
    output_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/processed_knowledge/empirical_constants_essential_manual_v1.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(essential_knowledge, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Essential empirical constants saved to:")
    print(f"   {output_file}")
    
    # Show sample items
    print(f"\n📄 Sample Knowledge Items:")
    for i, item in enumerate(essential_knowledge[:5]):
        print(f"   {i+1}. {item['content']}")
        print(f"      Mathematical form: {item['mathematical_form']}")
        print(f"      Certainty: {item['certainty']}")
        print(f"      Principle: {item['principle_name']}")
        print()
    
    # Create integration summary
    summary = {
        'creation_date': '2025-07-16',
        'method': 'manual_curation_from_authoritative_sources',
        'total_items': len(essential_knowledge),
        'categories': {cat: len(items) for cat, items in categories.items()},
        'average_certainty': avg_certainty,
        'certainty_distribution': {
            '0.999+': len([item for item in essential_knowledge if item['certainty'] >= 0.999]),
            '0.99-0.999': len([item for item in essential_knowledge if 0.99 <= item['certainty'] < 0.999]),
            '0.95-0.99': len([item for item in essential_knowledge if 0.95 <= item['certainty'] < 0.99]),
            '0.9-0.95': len([item for item in essential_knowledge if 0.9 <= item['certainty'] < 0.95])
        },
        'expansion_impact': f"WorldModelCore will expand from ~102 to {102 + len(essential_knowledge)} items"
    }
    
    summary_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/empirical_constants/essential_empirical_constants_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Summary saved to: {summary_file}")
    
    print(f"\n🎉 Essential Empirical Constants Knowledge Base Complete!")
    print(f"💡 WorldModelCore will expand from ~102 to {102 + len(essential_knowledge)} items!")
    print(f"🧠 High-quality knowledge with {avg_certainty:.1%} average certainty")
    
    # Show knowledge distribution
    print(f"\n📋 Knowledge Distribution:")
    for category, items in categories.items():
        print(f"   {category}: {len(items)} items")
        for item in items:
            print(f"      - {item['principle_name']}")
    
    return essential_knowledge

if __name__ == "__main__":
    essential_knowledge = expand_world_model_empirical_constants()
    
    print(f"\n✅ Manual empirical constants curation successful!")
    print(f"🚀 Ready to integrate with NWTN WorldModelCore system!")
    print(f"📊 Total knowledge items: {len(essential_knowledge)}")
    print(f"🎯 Next step: Update WorldModelCore._initialize_empirical_constants()")
    print(f"🔬 Ready for molecular & cell biology ZIM processing!")