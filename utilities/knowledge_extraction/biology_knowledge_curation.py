#!/usr/bin/env python3
"""
Biology Knowledge Curation for NWTN WorldModelCore
==================================================

Creates essential biology knowledge items manually from authoritative sources
to expand the WorldModelCore with fundamental biological principles.
"""

import json
import os
from typing import Dict, List, Any

def create_essential_biology_knowledge() -> List[Dict[str, Any]]:
    """
    Create essential biology knowledge items manually from authoritative sources
    
    Returns:
        List of essential biology knowledge items
    """
    
    essential_knowledge = [
        # Cell Biology Fundamentals
        {
            'content': 'The cell is the fundamental unit of life and organization in organisms',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'cell_biology',
            'mathematical_form': 'Cell = Basic Unit of Life',
            'dependencies': [],
            'references': ['Cell Biology textbooks', 'Cell theory', 'Molecular Biology'],
            'applicable_conditions': ['all living organisms'],
            'principle_name': 'Cell Theory'
        },
        {
            'content': 'All living cells arise from pre-existing cells',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'cell_biology',
            'mathematical_form': 'Cell → Cell (division)',
            'dependencies': [],
            'references': ['Cell Biology textbooks', 'Cell theory', 'Rudolf Virchow'],
            'applicable_conditions': ['all living organisms'],
            'principle_name': 'Cell Division Principle'
        },
        {
            'content': 'DNA contains the genetic information of organisms',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'molecular_biology',
            'mathematical_form': 'DNA: A-T, G-C base pairing',
            'dependencies': [],
            'references': ['Molecular Biology textbooks', 'Watson-Crick', 'Genetics'],
            'applicable_conditions': ['all living organisms'],
            'principle_name': 'DNA as Genetic Material'
        },
        {
            'content': 'DNA is replicated in a semi-conservative manner',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'molecular_biology',
            'mathematical_form': 'DNA → 2 DNA (each with one original strand)',
            'dependencies': [],
            'references': ['Molecular Biology textbooks', 'Meselson-Stahl experiment'],
            'applicable_conditions': ['DNA replication'],
            'principle_name': 'Semi-Conservative DNA Replication'
        },
        {
            'content': 'The genetic code is universal and consists of triplet codons',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'molecular_biology',
            'mathematical_form': '3 nucleotides = 1 codon = 1 amino acid',
            'dependencies': [],
            'references': ['Molecular Biology textbooks', 'Genetic code', 'Protein synthesis'],
            'applicable_conditions': ['protein synthesis'],
            'principle_name': 'Universal Genetic Code'
        },
        
        # Protein Biology
        {
            'content': 'Proteins are polymers of amino acids linked by peptide bonds',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'protein_biology',
            'mathematical_form': 'Amino acids + peptide bonds → Protein',
            'dependencies': [],
            'references': ['Biochemistry textbooks', 'Protein Chemistry'],
            'applicable_conditions': ['protein structure'],
            'principle_name': 'Protein Structure'
        },
        {
            'content': 'Protein function is determined by its three-dimensional structure',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'protein_biology',
            'mathematical_form': 'Structure → Function',
            'dependencies': [],
            'references': ['Biochemistry textbooks', 'Protein folding'],
            'applicable_conditions': ['protein function'],
            'principle_name': 'Structure-Function Relationship'
        },
        {
            'content': 'Enzymes are biological catalysts that lower activation energy',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'protein_biology',
            'mathematical_form': 'Enzyme + Substrate → Enzyme-Substrate Complex → Product',
            'dependencies': [],
            'references': ['Biochemistry textbooks', 'Enzyme kinetics'],
            'applicable_conditions': ['enzymatic reactions'],
            'principle_name': 'Enzyme Catalysis'
        },
        
        # Metabolism
        {
            'content': 'Cellular respiration converts glucose and oxygen to ATP, CO2, and water',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'metabolism',
            'mathematical_form': 'C6H12O6 + 6O2 → 6CO2 + 6H2O + ATP',
            'dependencies': [],
            'references': ['Biochemistry textbooks', 'Cell Biology', 'Metabolism'],
            'applicable_conditions': ['aerobic organisms'],
            'principle_name': 'Cellular Respiration'
        },
        {
            'content': 'Photosynthesis converts light energy, CO2, and water to glucose and oxygen',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'metabolism',
            'mathematical_form': '6CO2 + 6H2O + light energy → C6H12O6 + 6O2',
            'dependencies': [],
            'references': ['Plant Biology textbooks', 'Biochemistry', 'Photosynthesis'],
            'applicable_conditions': ['photosynthetic organisms'],
            'principle_name': 'Photosynthesis'
        },
        {
            'content': 'ATP is the universal energy currency of cells',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'metabolism',
            'mathematical_form': 'ATP → ADP + Pi + Energy',
            'dependencies': [],
            'references': ['Biochemistry textbooks', 'Bioenergetics'],
            'applicable_conditions': ['all living cells'],
            'principle_name': 'ATP Energy Currency'
        },
        
        # Genetics
        {
            'content': 'Genes are discrete units of heredity that determine traits',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'genetics',
            'mathematical_form': 'Gene → Trait',
            'dependencies': [],
            'references': ['Genetics textbooks', 'Mendel', 'Heredity'],
            'applicable_conditions': ['inherited traits'],
            'principle_name': 'Gene Concept'
        },
        {
            'content': 'Alleles segregate during gamete formation and reunite during fertilization',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'genetics',
            'mathematical_form': 'Aa → A or a (gametes) → AA, Aa, aa (offspring)',
            'dependencies': [],
            'references': ['Genetics textbooks', 'Mendel\'s Laws'],
            'applicable_conditions': ['sexual reproduction'],
            'principle_name': 'Law of Segregation'
        },
        {
            'content': 'Different genes assort independently during gamete formation',
            'certainty': 0.999,
            'domain': 'biology',
            'category': 'genetics',
            'mathematical_form': 'AaBb → AB, Ab, aB, ab (independent)',
            'dependencies': [],
            'references': ['Genetics textbooks', 'Mendel\'s Laws'],
            'applicable_conditions': ['unlinked genes'],
            'principle_name': 'Law of Independent Assortment'
        },
        
        # Evolution
        {
            'content': 'All species have descended from common ancestors through evolution',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'evolution',
            'mathematical_form': 'Common Ancestor → Divergent Species',
            'dependencies': [],
            'references': ['Evolution textbooks', 'Darwin', 'Phylogeny'],
            'applicable_conditions': ['all life forms'],
            'principle_name': 'Common Descent'
        },
        {
            'content': 'Natural selection acts on heritable variation to drive evolutionary change',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'evolution',
            'mathematical_form': 'Variation + Selection → Evolution',
            'dependencies': [],
            'references': ['Evolution textbooks', 'Darwin', 'Natural Selection'],
            'applicable_conditions': ['populations with variation'],
            'principle_name': 'Natural Selection'
        },
        {
            'content': 'Mutations are the ultimate source of genetic variation',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'evolution',
            'mathematical_form': 'DNA → Mutated DNA → Variation',
            'dependencies': [],
            'references': ['Evolution textbooks', 'Genetics', 'Mutation'],
            'applicable_conditions': ['genetic material'],
            'principle_name': 'Mutation as Source of Variation'
        },
        
        # Ecology
        {
            'content': 'Energy flows through ecosystems from producers to consumers',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'ecology',
            'mathematical_form': 'Producers → Primary Consumers → Secondary Consumers',
            'dependencies': [],
            'references': ['Ecology textbooks', 'Ecosystem dynamics'],
            'applicable_conditions': ['ecological systems'],
            'principle_name': 'Energy Flow'
        },
        {
            'content': 'Nutrients cycle through ecosystems and are recycled',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'ecology',
            'mathematical_form': 'Nutrients: Biotic ↔ Abiotic pools',
            'dependencies': [],
            'references': ['Ecology textbooks', 'Biogeochemical cycles'],
            'applicable_conditions': ['ecological systems'],
            'principle_name': 'Nutrient Cycling'
        },
        {
            'content': 'Population growth is limited by carrying capacity',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'ecology',
            'mathematical_form': 'dN/dt = rN(1 - N/K)',
            'dependencies': [],
            'references': ['Ecology textbooks', 'Population dynamics'],
            'applicable_conditions': ['population growth'],
            'principle_name': 'Carrying Capacity'
        },
        
        # Physiology
        {
            'content': 'Homeostasis maintains internal conditions within narrow ranges',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'physiology',
            'mathematical_form': 'Stimulus → Response → Equilibrium',
            'dependencies': [],
            'references': ['Physiology textbooks', 'Homeostasis'],
            'applicable_conditions': ['living organisms'],
            'principle_name': 'Homeostasis'
        },
        {
            'content': 'Negative feedback loops maintain physiological stability',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'physiology',
            'mathematical_form': 'Deviation → Opposite Response → Correction',
            'dependencies': [],
            'references': ['Physiology textbooks', 'Feedback mechanisms'],
            'applicable_conditions': ['physiological systems'],
            'principle_name': 'Negative Feedback'
        },
        {
            'content': 'Osmosis is the movement of water across semipermeable membranes',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'physiology',
            'mathematical_form': 'Water: High concentration → Low concentration',
            'dependencies': [],
            'references': ['Cell Biology textbooks', 'Membrane transport'],
            'applicable_conditions': ['cellular membranes'],
            'principle_name': 'Osmosis'
        },
        
        # Developmental Biology
        {
            'content': 'Development involves cell division, differentiation, and morphogenesis',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'developmental_biology',
            'mathematical_form': 'Zygote → Differentiated cells → Organism',
            'dependencies': [],
            'references': ['Developmental Biology textbooks', 'Embryology'],
            'applicable_conditions': ['multicellular organisms'],
            'principle_name': 'Developmental Processes'
        },
        {
            'content': 'Gene expression is regulated during development',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'developmental_biology',
            'mathematical_form': 'Gene regulation → Cell fate determination',
            'dependencies': [],
            'references': ['Developmental Biology textbooks', 'Gene regulation'],
            'applicable_conditions': ['developmental processes'],
            'principle_name': 'Developmental Gene Regulation'
        },
        
        # Immunology
        {
            'content': 'The immune system distinguishes self from non-self',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'immunology',
            'mathematical_form': 'Self-recognition → Tolerance; Non-self → Response',
            'dependencies': [],
            'references': ['Immunology textbooks', 'Immune recognition'],
            'applicable_conditions': ['immune systems'],
            'principle_name': 'Self/Non-Self Recognition'
        },
        {
            'content': 'Antibodies are specific proteins that bind to antigens',
            'certainty': 0.9999,
            'domain': 'biology',
            'category': 'immunology',
            'mathematical_form': 'Antibody + Antigen → Antibody-Antigen Complex',
            'dependencies': [],
            'references': ['Immunology textbooks', 'Antibody structure'],
            'applicable_conditions': ['adaptive immunity'],
            'principle_name': 'Antibody-Antigen Binding'
        }
    ]
    
    return essential_knowledge

def expand_world_model_biology():
    """Expand the WorldModelCore with essential biology knowledge"""
    
    print("🧬 Creating Essential Biology Knowledge Base")
    print("=" * 60)
    
    # Create essential biology knowledge
    essential_knowledge = create_essential_biology_knowledge()
    
    print(f"📚 Created {len(essential_knowledge)} essential biology knowledge items")
    
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
    output_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/processed_knowledge/biology_essential_manual_v1.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(essential_knowledge, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Essential biology knowledge saved to:")
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
        'expansion_impact': f"WorldModelCore will expand from ~125 to {125 + len(essential_knowledge)} items"
    }
    
    summary_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/biology/essential_biology_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Summary saved to: {summary_file}")
    
    print(f"\n🎉 Essential Biology Knowledge Base Complete!")
    print(f"💡 WorldModelCore will expand from ~125 to {125 + len(essential_knowledge)} items!")
    print(f"🧠 High-quality knowledge with {avg_certainty:.1%} average certainty")
    
    # Show knowledge distribution
    print(f"\n📋 Knowledge Distribution:")
    for category, items in categories.items():
        print(f"   {category}: {len(items)} items")
        for item in items:
            print(f"      - {item['principle_name']}")
    
    return essential_knowledge

if __name__ == "__main__":
    essential_knowledge = expand_world_model_biology()
    
    print(f"\n✅ Manual biology knowledge curation successful!")
    print(f"🚀 Ready to integrate with NWTN WorldModelCore system!")
    print(f"📊 Total knowledge items: {len(essential_knowledge)}")
    print(f"🎯 Next step: Update WorldModelCore._initialize_biological_foundations()")
    print(f"⚗️ Ready for chemistry ZIM processing!")