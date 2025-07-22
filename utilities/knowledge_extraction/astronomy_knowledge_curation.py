#!/usr/bin/env python3
"""
Astronomy Knowledge Curation for NWTN WorldModelCore
====================================================

Creates essential astronomy knowledge items manually from authoritative sources.
"""

import json
import os
from typing import Dict, List, Any

def create_essential_astronomy_knowledge() -> List[Dict[str, Any]]:
    """Create essential astronomy knowledge items"""
    
    essential_knowledge = [
        # Solar System
        {
            'content': 'The Sun is a G-type main-sequence star at the center of our solar system',
            'certainty': 0.9999,
            'domain': 'astronomy',
            'category': 'solar_system',
            'mathematical_form': 'Sun: G-type star + Main sequence',
            'dependencies': [],
            'references': ['Astronomy textbooks', 'Solar physics'],
            'applicable_conditions': ['solar system'],
            'principle_name': 'Solar Classification'
        },
        {
            'content': 'Planets orbit the Sun in elliptical paths following Kepler\'s laws',
            'certainty': 0.9999,
            'domain': 'astronomy',
            'category': 'solar_system',
            'mathematical_form': 'T² ∝ a³ (Kepler\'s Third Law)',
            'dependencies': [],
            'references': ['Kepler\'s laws', 'Orbital mechanics'],
            'applicable_conditions': ['planetary motion'],
            'principle_name': 'Kepler\'s Laws'
        },
        {
            'content': 'Earth is the third planet from the Sun in the habitable zone',
            'certainty': 0.9999,
            'domain': 'astronomy',
            'category': 'solar_system',
            'mathematical_form': 'Earth: Third planet + Habitable zone',
            'dependencies': [],
            'references': ['Astronomy textbooks', 'Planetary science'],
            'applicable_conditions': ['solar system'],
            'principle_name': 'Earth\'s Position'
        },
        
        # Stellar Evolution
        {
            'content': 'Stars form from gravitational collapse of gas and dust clouds',
            'certainty': 0.9999,
            'domain': 'astronomy',
            'category': 'stellar_evolution',
            'mathematical_form': 'Gas cloud + Gravity → Star formation',
            'dependencies': [],
            'references': ['Stellar evolution textbooks', 'Star formation'],
            'applicable_conditions': ['stellar birth'],
            'principle_name': 'Star Formation'
        },
        {
            'content': 'Nuclear fusion in stellar cores converts hydrogen to helium',
            'certainty': 0.9999,
            'domain': 'astronomy',
            'category': 'stellar_evolution',
            'mathematical_form': '4H → He + Energy',
            'dependencies': [],
            'references': ['Nuclear astrophysics', 'Stellar physics'],
            'applicable_conditions': ['main sequence stars'],
            'principle_name': 'Stellar Nuclear Fusion'
        },
        {
            'content': 'Massive stars end their lives as supernovae, neutron stars, or black holes',
            'certainty': 0.9999,
            'domain': 'astronomy',
            'category': 'stellar_evolution',
            'mathematical_form': 'Massive star → Supernova → Neutron star/Black hole',
            'dependencies': [],
            'references': ['Stellar evolution textbooks', 'Supernova physics'],
            'applicable_conditions': ['massive stars'],
            'principle_name': 'Stellar Death'
        },
        
        # Galaxies
        {
            'content': 'Galaxies are gravitationally bound systems of stars, gas, and dark matter',
            'certainty': 0.9999,
            'domain': 'astronomy',
            'category': 'galaxies',
            'mathematical_form': 'Galaxy = Stars + Gas + Dark matter',
            'dependencies': [],
            'references': ['Galaxy textbooks', 'Galactic astronomy'],
            'applicable_conditions': ['galactic systems'],
            'principle_name': 'Galaxy Definition'
        },
        {
            'content': 'The Milky Way is a barred spiral galaxy containing our solar system',
            'certainty': 0.9999,
            'domain': 'astronomy',
            'category': 'galaxies',
            'mathematical_form': 'Milky Way: Barred spiral + Solar system',
            'dependencies': [],
            'references': ['Galactic astronomy', 'Milky Way studies'],
            'applicable_conditions': ['local galaxy'],
            'principle_name': 'Milky Way Structure'
        },
        {
            'content': 'Galaxies are moving away from us, indicating universe expansion',
            'certainty': 0.9999,
            'domain': 'astronomy',
            'category': 'galaxies',
            'mathematical_form': 'v = H₀ × d (Hubble\'s Law)',
            'dependencies': [],
            'references': ['Cosmology textbooks', 'Hubble\'s observations'],
            'applicable_conditions': ['cosmological distances'],
            'principle_name': 'Hubble\'s Law'
        },
        
        # Cosmology
        {
            'content': 'The universe began with the Big Bang approximately 13.8 billion years ago',
            'certainty': 0.999,
            'domain': 'astronomy',
            'category': 'cosmology',
            'mathematical_form': 'Universe age ≈ 13.8 Gyr',
            'dependencies': [],
            'references': ['Big Bang cosmology', 'CMB observations'],
            'applicable_conditions': ['cosmic history'],
            'principle_name': 'Big Bang Theory'
        },
        {
            'content': 'The cosmic microwave background is radiation left over from the Big Bang',
            'certainty': 0.9999,
            'domain': 'astronomy',
            'category': 'cosmology',
            'mathematical_form': 'CMB: Primordial radiation + T ≈ 2.7 K',
            'dependencies': [],
            'references': ['CMB discovery', 'Cosmology textbooks'],
            'applicable_conditions': ['early universe'],
            'principle_name': 'Cosmic Microwave Background'
        },
        {
            'content': 'Dark matter comprises approximately 27% of the universe\'s mass-energy',
            'certainty': 0.99,
            'domain': 'astronomy',
            'category': 'cosmology',
            'mathematical_form': 'Dark matter ≈ 27% of universe',
            'dependencies': [],
            'references': ['Dark matter studies', 'Cosmological surveys'],
            'applicable_conditions': ['cosmic composition'],
            'principle_name': 'Dark Matter'
        },
        {
            'content': 'Dark energy drives accelerating expansion of the universe',
            'certainty': 0.99,
            'domain': 'astronomy',
            'category': 'cosmology',
            'mathematical_form': 'Dark energy → Accelerating expansion',
            'dependencies': [],
            'references': ['Dark energy studies', 'Supernova observations'],
            'applicable_conditions': ['cosmic acceleration'],
            'principle_name': 'Dark Energy'
        },
        
        # Exoplanets
        {
            'content': 'Exoplanets are planets orbiting stars other than the Sun',
            'certainty': 0.9999,
            'domain': 'astronomy',
            'category': 'exoplanets',
            'mathematical_form': 'Exoplanet: Planet + Other star system',
            'dependencies': [],
            'references': ['Exoplanet textbooks', 'Planet detection'],
            'applicable_conditions': ['extrasolar systems'],
            'principle_name': 'Exoplanet Definition'
        },
        {
            'content': 'Transit method detects exoplanets by observing stellar brightness dips',
            'certainty': 0.9999,
            'domain': 'astronomy',
            'category': 'exoplanets',
            'mathematical_form': 'Transit: Planet blocks starlight → Brightness dip',
            'dependencies': [],
            'references': ['Transit photometry', 'Exoplanet detection'],
            'applicable_conditions': ['planet detection'],
            'principle_name': 'Transit Method'
        },
        {
            'content': 'Habitable zone is the region around a star where liquid water can exist',
            'certainty': 0.9999,
            'domain': 'astronomy',
            'category': 'exoplanets',
            'mathematical_form': 'Habitable zone: Liquid water possible',
            'dependencies': [],
            'references': ['Astrobiology', 'Habitability studies'],
            'applicable_conditions': ['planetary habitability'],
            'principle_name': 'Habitable Zone'
        },
        
        # Observational Astronomy
        {
            'content': 'Telescopes collect electromagnetic radiation to observe celestial objects',
            'certainty': 0.9999,
            'domain': 'astronomy',
            'category': 'observational',
            'mathematical_form': 'Telescope: Collect radiation → Observe objects',
            'dependencies': [],
            'references': ['Observational astronomy', 'Telescope design'],
            'applicable_conditions': ['astronomical observation'],
            'principle_name': 'Telescope Function'
        },
        {
            'content': 'Doppler shift reveals motion of celestial objects toward or away from us',
            'certainty': 0.9999,
            'domain': 'astronomy',
            'category': 'observational',
            'mathematical_form': 'Δλ/λ = v/c',
            'dependencies': [],
            'references': ['Doppler effect', 'Spectroscopy'],
            'applicable_conditions': ['radial velocity'],
            'principle_name': 'Doppler Shift'
        },
        {
            'content': 'Parallax measures stellar distances using Earth\'s orbital motion',
            'certainty': 0.9999,
            'domain': 'astronomy',
            'category': 'observational',
            'mathematical_form': 'Distance = 1/parallax (in parsecs)',
            'dependencies': [],
            'references': ['Astrometry', 'Distance measurement'],
            'applicable_conditions': ['stellar distances'],
            'principle_name': 'Stellar Parallax'
        },
        
        # Planetary Science
        {
            'content': 'Terrestrial planets are rocky worlds in the inner solar system',
            'certainty': 0.9999,
            'domain': 'astronomy',
            'category': 'planetary_science',
            'mathematical_form': 'Terrestrial: Rocky + Inner solar system',
            'dependencies': [],
            'references': ['Planetary science', 'Solar system formation'],
            'applicable_conditions': ['rocky planets'],
            'principle_name': 'Terrestrial Planets'
        },
        {
            'content': 'Gas giants are massive planets composed primarily of hydrogen and helium',
            'certainty': 0.9999,
            'domain': 'astronomy',
            'category': 'planetary_science',
            'mathematical_form': 'Gas giant: Massive + H/He composition',
            'dependencies': [],
            'references': ['Planetary science', 'Gas giant studies'],
            'applicable_conditions': ['gas giant planets'],
            'principle_name': 'Gas Giants'
        },
        {
            'content': 'Impact cratering shapes planetary surfaces throughout the solar system',
            'certainty': 0.9999,
            'domain': 'astronomy',
            'category': 'planetary_science',
            'mathematical_form': 'Impact → Crater formation',
            'dependencies': [],
            'references': ['Impact cratering', 'Planetary geology'],
            'applicable_conditions': ['planetary surfaces'],
            'principle_name': 'Impact Cratering'
        }
    ]
    
    return essential_knowledge

def expand_world_model_astronomy():
    """Expand the WorldModelCore with essential astronomy knowledge"""
    
    print("🌌 Creating Essential Astronomy Knowledge Base")
    print("=" * 60)
    
    # Create essential astronomy knowledge
    essential_knowledge = create_essential_astronomy_knowledge()
    
    print(f"📚 Created {len(essential_knowledge)} essential astronomy knowledge items")
    
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
    os.makedirs("/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/astronomy", exist_ok=True)
    output_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/processed_knowledge/astronomy_essential_manual_v1.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(essential_knowledge, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Essential astronomy knowledge saved to:")
    print(f"   {output_file}")
    
    # Show sample items
    print(f"\n📄 Sample Knowledge Items:")
    for i, item in enumerate(essential_knowledge[:3]):
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
        'expansion_impact': f"WorldModelCore will expand significantly with astronomy knowledge"
    }
    
    summary_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/astronomy/essential_astronomy_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Summary saved to: {summary_file}")
    
    print(f"\n🎉 Essential Astronomy Knowledge Base Complete!")
    print(f"💡 WorldModelCore ready for astronomy integration!")
    print(f"🧠 High-quality knowledge with {avg_certainty:.1%} average certainty")
    
    return essential_knowledge

if __name__ == "__main__":
    essential_knowledge = expand_world_model_astronomy()
    
    print(f"\n✅ Manual astronomy knowledge curation successful!")
    print(f"🚀 Ready to integrate with NWTN WorldModelCore system!")
    print(f"📊 Total knowledge items: {len(essential_knowledge)}")
    print(f"🏥 Ready for medicine knowledge curation!")