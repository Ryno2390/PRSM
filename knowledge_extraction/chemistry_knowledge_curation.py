#!/usr/bin/env python3
"""
Chemistry Knowledge Curation for NWTN WorldModelCore
====================================================

Creates essential chemistry knowledge items manually from authoritative sources
to expand the WorldModelCore with fundamental chemical principles.
"""

import json
import os
from typing import Dict, List, Any

def create_essential_chemistry_knowledge() -> List[Dict[str, Any]]:
    """
    Create essential chemistry knowledge items manually from authoritative sources
    
    Returns:
        List of essential chemistry knowledge items
    """
    
    essential_knowledge = [
        # Atomic Theory
        {
            'content': 'All matter is composed of atoms, which are the basic units of chemical elements',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'atomic_theory',
            'mathematical_form': 'Matter = Atoms (indivisible units)',
            'dependencies': [],
            'references': ['General Chemistry textbooks', 'Atomic theory', 'Dalton'],
            'applicable_conditions': ['chemical matter'],
            'principle_name': 'Atomic Theory'
        },
        {
            'content': 'Atoms of the same element have the same number of protons',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'atomic_theory',
            'mathematical_form': 'Atomic number = Number of protons',
            'dependencies': [],
            'references': ['General Chemistry textbooks', 'Atomic structure'],
            'applicable_conditions': ['chemical elements'],
            'principle_name': 'Atomic Number Definition'
        },
        {
            'content': 'Isotopes are atoms of the same element with different numbers of neutrons',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'atomic_theory',
            'mathematical_form': 'Same protons, different neutrons → Isotopes',
            'dependencies': [],
            'references': ['General Chemistry textbooks', 'Isotopes'],
            'applicable_conditions': ['atomic nuclei'],
            'principle_name': 'Isotope Definition'
        },
        
        # Periodic Table
        {
            'content': 'Elements are arranged in the periodic table by increasing atomic number',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'periodic_table',
            'mathematical_form': 'Periodic Table order = Atomic number sequence',
            'dependencies': [],
            'references': ['General Chemistry textbooks', 'Periodic table', 'Mendeleev'],
            'applicable_conditions': ['chemical elements'],
            'principle_name': 'Periodic Table Organization'
        },
        {
            'content': 'Element properties vary periodically with atomic number',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'periodic_table',
            'mathematical_form': 'Properties = f(Atomic number, periodic)',
            'dependencies': [],
            'references': ['General Chemistry textbooks', 'Periodic law'],
            'applicable_conditions': ['chemical elements'],
            'principle_name': 'Periodic Law'
        },
        {
            'content': 'Electron configuration determines chemical properties of elements',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'periodic_table',
            'mathematical_form': 'Electron configuration → Chemical properties',
            'dependencies': [],
            'references': ['General Chemistry textbooks', 'Electron configuration'],
            'applicable_conditions': ['atoms and ions'],
            'principle_name': 'Electron Configuration Principle'
        },
        
        # Chemical Bonding
        {
            'content': 'Atoms form ionic bonds by transferring electrons',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'chemical_bonding',
            'mathematical_form': 'Metal + Nonmetal → Ionic compound',
            'dependencies': [],
            'references': ['General Chemistry textbooks', 'Ionic bonding'],
            'applicable_conditions': ['metals and nonmetals'],
            'principle_name': 'Ionic Bonding'
        },
        {
            'content': 'Atoms form covalent bonds by sharing electrons',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'chemical_bonding',
            'mathematical_form': 'Nonmetal + Nonmetal → Covalent compound',
            'dependencies': [],
            'references': ['General Chemistry textbooks', 'Covalent bonding'],
            'applicable_conditions': ['nonmetals'],
            'principle_name': 'Covalent Bonding'
        },
        {
            'content': 'Molecular geometry is determined by electron pair repulsion',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'chemical_bonding',
            'mathematical_form': 'Electron pairs → Geometry (VSEPR)',
            'dependencies': [],
            'references': ['General Chemistry textbooks', 'VSEPR theory'],
            'applicable_conditions': ['molecules'],
            'principle_name': 'VSEPR Theory'
        },
        
        # Stoichiometry
        {
            'content': 'Mass is conserved in chemical reactions',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'stoichiometry',
            'mathematical_form': 'Mass reactants = Mass products',
            'dependencies': [],
            'references': ['General Chemistry textbooks', 'Conservation of mass'],
            'applicable_conditions': ['chemical reactions'],
            'principle_name': 'Conservation of Mass'
        },
        {
            'content': 'Chemical equations must be balanced to conserve atoms',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'stoichiometry',
            'mathematical_form': 'Atoms in = Atoms out',
            'dependencies': [],
            'references': ['General Chemistry textbooks', 'Balancing equations'],
            'applicable_conditions': ['chemical equations'],
            'principle_name': 'Balanced Chemical Equations'
        },
        {
            'content': 'Molar ratios in balanced equations determine stoichiometric relationships',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'stoichiometry',
            'mathematical_form': 'Mole ratio = Coefficient ratio',
            'dependencies': [],
            'references': ['General Chemistry textbooks', 'Stoichiometry'],
            'applicable_conditions': ['chemical reactions'],
            'principle_name': 'Stoichiometric Relationships'
        },
        
        # Thermodynamics
        {
            'content': 'Enthalpy change measures heat absorbed or released in reactions',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'thermodynamics',
            'mathematical_form': 'ΔH = Heat absorbed or released',
            'dependencies': [],
            'references': ['Physical Chemistry textbooks', 'Thermodynamics'],
            'applicable_conditions': ['chemical reactions'],
            'principle_name': 'Enthalpy Change'
        },
        {
            'content': 'Entropy measures the disorder or randomness of a system',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'thermodynamics',
            'mathematical_form': 'ΔS = Change in disorder',
            'dependencies': [],
            'references': ['Physical Chemistry textbooks', 'Thermodynamics'],
            'applicable_conditions': ['chemical systems'],
            'principle_name': 'Entropy'
        },
        {
            'content': 'Gibbs free energy determines spontaneity of reactions',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'thermodynamics',
            'mathematical_form': 'ΔG = ΔH - TΔS',
            'dependencies': [],
            'references': ['Physical Chemistry textbooks', 'Thermodynamics'],
            'applicable_conditions': ['chemical reactions'],
            'principle_name': 'Gibbs Free Energy'
        },
        
        # Kinetics
        {
            'content': 'Reaction rate depends on concentration of reactants',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'kinetics',
            'mathematical_form': 'Rate = k[A]^m[B]^n',
            'dependencies': [],
            'references': ['Physical Chemistry textbooks', 'Kinetics'],
            'applicable_conditions': ['chemical reactions'],
            'principle_name': 'Rate Law'
        },
        {
            'content': 'Catalysts increase reaction rate without being consumed',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'kinetics',
            'mathematical_form': 'Catalyst: Lower activation energy',
            'dependencies': [],
            'references': ['Physical Chemistry textbooks', 'Catalysis'],
            'applicable_conditions': ['chemical reactions'],
            'principle_name': 'Catalysis'
        },
        {
            'content': 'Temperature increases reaction rate exponentially',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'kinetics',
            'mathematical_form': 'k = A⋅e^(-Ea/RT)',
            'dependencies': [],
            'references': ['Physical Chemistry textbooks', 'Arrhenius equation'],
            'applicable_conditions': ['chemical reactions'],
            'principle_name': 'Arrhenius Equation'
        },
        
        # Equilibrium
        {
            'content': 'Chemical equilibrium occurs when forward and reverse reaction rates are equal',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'equilibrium',
            'mathematical_form': 'Rate forward = Rate reverse',
            'dependencies': [],
            'references': ['Physical Chemistry textbooks', 'Equilibrium'],
            'applicable_conditions': ['reversible reactions'],
            'principle_name': 'Chemical Equilibrium'
        },
        {
            'content': 'Equilibrium constant depends only on temperature',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'equilibrium',
            'mathematical_form': 'K = f(Temperature only)',
            'dependencies': [],
            'references': ['Physical Chemistry textbooks', 'Equilibrium constant'],
            'applicable_conditions': ['equilibrium systems'],
            'principle_name': 'Equilibrium Constant'
        },
        {
            'content': 'Le Chatelier\'s principle predicts equilibrium shifts',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'equilibrium',
            'mathematical_form': 'Stress → Equilibrium shift to relieve stress',
            'dependencies': [],
            'references': ['Physical Chemistry textbooks', 'Le Chatelier\'s principle'],
            'applicable_conditions': ['equilibrium systems'],
            'principle_name': 'Le Chatelier\'s Principle'
        },
        
        # Acids and Bases
        {
            'content': 'Acids donate protons and bases accept protons',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'acids_bases',
            'mathematical_form': 'HA → H⁺ + A⁻ (acid); B + H⁺ → BH⁺ (base)',
            'dependencies': [],
            'references': ['General Chemistry textbooks', 'Brønsted-Lowry theory'],
            'applicable_conditions': ['aqueous solutions'],
            'principle_name': 'Brønsted-Lowry Theory'
        },
        {
            'content': 'pH measures hydrogen ion concentration',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'acids_bases',
            'mathematical_form': 'pH = -log[H⁺]',
            'dependencies': [],
            'references': ['General Chemistry textbooks', 'pH scale'],
            'applicable_conditions': ['aqueous solutions'],
            'principle_name': 'pH Scale'
        },
        {
            'content': 'Water autoionizes to produce equal concentrations of H⁺ and OH⁻',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'acids_bases',
            'mathematical_form': 'H₂O ⇌ H⁺ + OH⁻; [H⁺][OH⁻] = 10⁻¹⁴',
            'dependencies': [],
            'references': ['General Chemistry textbooks', 'Water autoionization'],
            'applicable_conditions': ['aqueous solutions'],
            'principle_name': 'Water Autoionization'
        },
        
        # Organic Chemistry
        {
            'content': 'Carbon forms four covalent bonds in stable compounds',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'organic_chemistry',
            'mathematical_form': 'C forms 4 bonds',
            'dependencies': [],
            'references': ['Organic Chemistry textbooks', 'Carbon bonding'],
            'applicable_conditions': ['organic compounds'],
            'principle_name': 'Carbon Tetravalency'
        },
        {
            'content': 'Hydrocarbons are compounds containing only carbon and hydrogen',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'organic_chemistry',
            'mathematical_form': 'CₓHᵧ compounds',
            'dependencies': [],
            'references': ['Organic Chemistry textbooks', 'Hydrocarbons'],
            'applicable_conditions': ['organic compounds'],
            'principle_name': 'Hydrocarbon Definition'
        },
        {
            'content': 'Functional groups determine chemical properties of organic compounds',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'organic_chemistry',
            'mathematical_form': 'Functional group → Chemical behavior',
            'dependencies': [],
            'references': ['Organic Chemistry textbooks', 'Functional groups'],
            'applicable_conditions': ['organic compounds'],
            'principle_name': 'Functional Group Concept'
        },
        
        # Electrochemistry
        {
            'content': 'Oxidation involves loss of electrons, reduction involves gain of electrons',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'electrochemistry',
            'mathematical_form': 'Oxidation: lose e⁻; Reduction: gain e⁻',
            'dependencies': [],
            'references': ['General Chemistry textbooks', 'Redox reactions'],
            'applicable_conditions': ['redox reactions'],
            'principle_name': 'Redox Reactions'
        },
        {
            'content': 'Electrochemical cells convert chemical energy to electrical energy',
            'certainty': 0.9999,
            'domain': 'chemistry',
            'category': 'electrochemistry',
            'mathematical_form': 'Chemical energy → Electrical energy',
            'dependencies': [],
            'references': ['Physical Chemistry textbooks', 'Electrochemistry'],
            'applicable_conditions': ['electrochemical cells'],
            'principle_name': 'Electrochemical Cells'
        }
    ]
    
    return essential_knowledge

def expand_world_model_chemistry():
    """Expand the WorldModelCore with essential chemistry knowledge"""
    
    print("⚗️ Creating Essential Chemistry Knowledge Base")
    print("=" * 60)
    
    # Create essential chemistry knowledge
    essential_knowledge = create_essential_chemistry_knowledge()
    
    print(f"📚 Created {len(essential_knowledge)} essential chemistry knowledge items")
    
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
    output_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/processed_knowledge/chemistry_essential_manual_v1.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(essential_knowledge, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Essential chemistry knowledge saved to:")
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
            '0.95-0.99': len([item for item in essential_knowledge if 0.95 <= item['certainty'] < 0.99])
        },
        'expansion_impact': f"WorldModelCore will expand from ~152 to {152 + len(essential_knowledge)} items"
    }
    
    summary_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/chemistry/essential_chemistry_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Summary saved to: {summary_file}")
    
    print(f"\n🎉 Essential Chemistry Knowledge Base Complete!")
    print(f"💡 WorldModelCore will expand from ~152 to {152 + len(essential_knowledge)} items!")
    print(f"🧠 High-quality knowledge with {avg_certainty:.1%} average certainty")
    
    # Show knowledge distribution
    print(f"\n📋 Knowledge Distribution:")
    for category, items in categories.items():
        print(f"   {category}: {len(items)} items")
        for item in items:
            print(f"      - {item['principle_name']}")
    
    return essential_knowledge

if __name__ == "__main__":
    essential_knowledge = expand_world_model_chemistry()
    
    print(f"\n✅ Manual chemistry knowledge curation successful!")
    print(f"🚀 Ready to integrate with NWTN WorldModelCore system!")
    print(f"📊 Total knowledge items: {len(essential_knowledge)}")
    print(f"🎯 Next step: Update WorldModelCore._initialize_chemical_principles()")
    print(f"💻 Ready for computer science ZIM processing!")