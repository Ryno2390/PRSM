#!/usr/bin/env python3
"""
Mathematics Knowledge Curation for NWTN WorldModelCore
======================================================

Creates essential mathematics knowledge items manually from authoritative sources
to expand the WorldModelCore with fundamental mathematical principles.
"""

import json
import os
from typing import Dict, List, Any

def create_essential_mathematics_knowledge() -> List[Dict[str, Any]]:
    """
    Create essential mathematics knowledge items manually from authoritative sources
    
    Returns:
        List of essential mathematics knowledge items
    """
    
    essential_knowledge = [
        # Basic Algebra
        {
            'content': 'For any real numbers a and b: a + b = b + a',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'algebra',
            'mathematical_form': 'a + b = b + a',
            'dependencies': [],
            'references': ['Abstract Algebra textbooks', 'Field axioms'],
            'applicable_conditions': ['real numbers', 'addition operation'],
            'principle_name': 'Commutative Property of Addition'
        },
        {
            'content': 'For any real numbers a and b: a × b = b × a',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'algebra',
            'mathematical_form': 'a × b = b × a',
            'dependencies': [],
            'references': ['Abstract Algebra textbooks', 'Field axioms'],
            'applicable_conditions': ['real numbers', 'multiplication operation'],
            'principle_name': 'Commutative Property of Multiplication'
        },
        {
            'content': 'For any real numbers a, b, and c: (a + b) + c = a + (b + c)',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'algebra',
            'mathematical_form': '(a + b) + c = a + (b + c)',
            'dependencies': [],
            'references': ['Abstract Algebra textbooks', 'Group axioms'],
            'applicable_conditions': ['real numbers', 'addition operation'],
            'principle_name': 'Associative Property of Addition'
        },
        {
            'content': 'For any real numbers a, b, and c: a × (b + c) = (a × b) + (a × c)',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'algebra',
            'mathematical_form': 'a × (b + c) = (a × b) + (a × c)',
            'dependencies': [],
            'references': ['Abstract Algebra textbooks', 'Distributive law'],
            'applicable_conditions': ['real numbers', 'multiplication and addition'],
            'principle_name': 'Distributive Property'
        },
        {
            'content': 'For any real number a: a + 0 = a',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'algebra',
            'mathematical_form': 'a + 0 = a',
            'dependencies': [],
            'references': ['Abstract Algebra textbooks', 'Additive identity'],
            'applicable_conditions': ['real numbers', 'zero element'],
            'principle_name': 'Additive Identity'
        },
        {
            'content': 'For any real number a: a × 1 = a',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'algebra',
            'mathematical_form': 'a × 1 = a',
            'dependencies': [],
            'references': ['Abstract Algebra textbooks', 'Multiplicative identity'],
            'applicable_conditions': ['real numbers', 'unity element'],
            'principle_name': 'Multiplicative Identity'
        },
        
        # Geometry
        {
            'content': 'The sum of angles in any triangle is 180 degrees',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'geometry',
            'mathematical_form': 'α + β + γ = 180°',
            'dependencies': [],
            'references': ['Euclidean Geometry', 'Geometric theorems'],
            'applicable_conditions': ['Euclidean space', 'triangles'],
            'principle_name': 'Triangle Angle Sum Theorem'
        },
        {
            'content': 'In a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'geometry',
            'mathematical_form': 'c² = a² + b²',
            'dependencies': [],
            'references': ['Pythagorean Theorem', 'Euclidean Geometry'],
            'applicable_conditions': ['right triangles', 'Euclidean space'],
            'principle_name': 'Pythagorean Theorem'
        },
        {
            'content': 'The area of a circle is pi times the radius squared',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'geometry',
            'mathematical_form': 'A = πr²',
            'dependencies': [],
            'references': ['Circle geometry', 'Calculus derivation'],
            'applicable_conditions': ['circles', 'Euclidean space'],
            'principle_name': 'Circle Area Formula'
        },
        {
            'content': 'The circumference of a circle is pi times the diameter',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'geometry',
            'mathematical_form': 'C = πd = 2πr',
            'dependencies': [],
            'references': ['Circle geometry', 'Definition of pi'],
            'applicable_conditions': ['circles', 'Euclidean space'],
            'principle_name': 'Circle Circumference Formula'
        },
        
        # Calculus
        {
            'content': 'The derivative of a constant is zero',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'calculus',
            'mathematical_form': 'd/dx[c] = 0',
            'dependencies': [],
            'references': ['Calculus textbooks', 'Derivative rules'],
            'applicable_conditions': ['constant functions', 'differentiable functions'],
            'principle_name': 'Derivative of Constant Rule'
        },
        {
            'content': 'The derivative of x raised to the power n is n times x raised to the power (n-1)',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'calculus',
            'mathematical_form': 'd/dx[xⁿ] = nxⁿ⁻¹',
            'dependencies': [],
            'references': ['Calculus textbooks', 'Power rule'],
            'applicable_conditions': ['polynomial functions', 'differentiable functions'],
            'principle_name': 'Power Rule for Derivatives'
        },
        {
            'content': 'The derivative of a sum is the sum of the derivatives',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'calculus',
            'mathematical_form': 'd/dx[f(x) + g(x)] = f\'(x) + g\'(x)',
            'dependencies': [],
            'references': ['Calculus textbooks', 'Linearity of derivatives'],
            'applicable_conditions': ['differentiable functions', 'sum of functions'],
            'principle_name': 'Sum Rule for Derivatives'
        },
        {
            'content': 'The fundamental theorem of calculus relates differentiation and integration',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'calculus',
            'mathematical_form': '∫ₐᵇ f\'(x)dx = f(b) - f(a)',
            'dependencies': [],
            'references': ['Fundamental Theorem of Calculus', 'Calculus textbooks'],
            'applicable_conditions': ['continuous functions', 'differentiable functions'],
            'principle_name': 'Fundamental Theorem of Calculus'
        },
        
        # Statistics and Probability
        {
            'content': 'The probability of any event is between 0 and 1, inclusive',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'statistics',
            'mathematical_form': '0 ≤ P(A) ≤ 1',
            'dependencies': [],
            'references': ['Probability Theory', 'Kolmogorov axioms'],
            'applicable_conditions': ['probability spaces', 'events'],
            'principle_name': 'Probability Bounds'
        },
        {
            'content': 'The probabilities of all possible outcomes sum to 1',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'statistics',
            'mathematical_form': 'Σ P(Aᵢ) = 1',
            'dependencies': [],
            'references': ['Probability Theory', 'Kolmogorov axioms'],
            'applicable_conditions': ['complete probability space', 'mutually exclusive events'],
            'principle_name': 'Total Probability Law'
        },
        {
            'content': 'The mean of a dataset is the sum of all values divided by the number of values',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'statistics',
            'mathematical_form': 'μ = (Σxᵢ)/n',
            'dependencies': [],
            'references': ['Statistics textbooks', 'Descriptive statistics'],
            'applicable_conditions': ['finite datasets', 'numerical data'],
            'principle_name': 'Arithmetic Mean'
        },
        
        # Number Theory
        {
            'content': 'Every integer greater than 1 is either prime or can be expressed as a product of primes',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'number_theory',
            'mathematical_form': 'n = p₁^α₁ × p₂^α₂ × ... × pₖ^αₖ',
            'dependencies': [],
            'references': ['Fundamental Theorem of Arithmetic', 'Number Theory textbooks'],
            'applicable_conditions': ['integers greater than 1', 'prime factorization'],
            'principle_name': 'Fundamental Theorem of Arithmetic'
        },
        {
            'content': 'There are infinitely many prime numbers',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'number_theory',
            'mathematical_form': '|{p : p is prime}| = ∞',
            'dependencies': [],
            'references': ['Euclid\'s theorem', 'Number Theory textbooks'],
            'applicable_conditions': ['prime numbers', 'infinite sets'],
            'principle_name': 'Infinitude of Primes'
        },
        
        # Logic and Set Theory
        {
            'content': 'For any proposition P: P is either true or false, but not both',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'logic',
            'mathematical_form': 'P ∨ ¬P and ¬(P ∧ ¬P)',
            'dependencies': [],
            'references': ['Classical Logic', 'Propositional Logic'],
            'applicable_conditions': ['classical logic', 'propositions'],
            'principle_name': 'Law of Excluded Middle'
        },
        {
            'content': 'A set is either a subset of another set or it is not',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'set_theory',
            'mathematical_form': 'A ⊆ B or A ⊄ B',
            'dependencies': [],
            'references': ['Set Theory textbooks', 'ZFC axioms'],
            'applicable_conditions': ['sets', 'subset relation'],
            'principle_name': 'Subset Relation Property'
        },
        {
            'content': 'The intersection of a set with itself is the set itself',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'set_theory',
            'mathematical_form': 'A ∩ A = A',
            'dependencies': [],
            'references': ['Set Theory textbooks', 'Set operations'],
            'applicable_conditions': ['sets', 'intersection operation'],
            'principle_name': 'Idempotent Property of Intersection'
        },
        {
            'content': 'The union of a set with the empty set is the set itself',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'set_theory',
            'mathematical_form': 'A ∪ ∅ = A',
            'dependencies': [],
            'references': ['Set Theory textbooks', 'Set operations'],
            'applicable_conditions': ['sets', 'union operation', 'empty set'],
            'principle_name': 'Identity Property of Union'
        },
        
        # Mathematical Constants
        {
            'content': 'Pi is the ratio of a circle\'s circumference to its diameter',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'constants',
            'mathematical_form': 'π ≈ 3.14159265359',
            'dependencies': [],
            'references': ['Mathematical constants', 'Circle geometry'],
            'applicable_conditions': ['circles', 'Euclidean geometry'],
            'principle_name': 'Pi Constant'
        },
        {
            'content': 'Euler\'s number e is the base of the natural logarithm',
            'certainty': 0.9999,
            'domain': 'mathematics',
            'category': 'constants',
            'mathematical_form': 'e ≈ 2.71828182846',
            'dependencies': [],
            'references': ['Mathematical constants', 'Natural logarithm'],
            'applicable_conditions': ['exponential functions', 'natural logarithm'],
            'principle_name': 'Euler\'s Number'
        }
    ]
    
    return essential_knowledge

def expand_world_model_mathematics():
    """Expand the WorldModelCore with essential mathematics knowledge"""
    
    print("🔢 Creating Essential Mathematics Knowledge Base")
    print("=" * 60)
    
    # Create essential mathematics knowledge
    essential_knowledge = create_essential_mathematics_knowledge()
    
    print(f"📚 Created {len(essential_knowledge)} essential mathematics knowledge items")
    
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
    output_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/processed_knowledge/mathematics_essential_manual_v1.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(essential_knowledge, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Essential mathematics knowledge saved to:")
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
        'expansion_impact': f"WorldModelCore will expand from ~65 to {65 + len(essential_knowledge)} items"
    }
    
    summary_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/mathematics/essential_mathematics_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Summary saved to: {summary_file}")
    
    print(f"\n🎉 Essential Mathematics Knowledge Base Complete!")
    print(f"💡 WorldModelCore will expand from ~65 to {65 + len(essential_knowledge)} items!")
    print(f"🧠 High-quality knowledge with {avg_certainty:.1%} average certainty")
    
    # Show knowledge distribution
    print(f"\n📋 Knowledge Distribution:")
    for category, items in categories.items():
        print(f"   {category}: {len(items)} items")
        for item in items:
            print(f"      - {item['principle_name']}")
    
    return essential_knowledge

if __name__ == "__main__":
    essential_knowledge = expand_world_model_mathematics()
    
    print(f"\n✅ Manual mathematics knowledge curation successful!")
    print(f"🚀 Ready to integrate with NWTN WorldModelCore system!")
    print(f"📊 Total knowledge items: {len(essential_knowledge)}")
    print(f"🎯 Next step: Update WorldModelCore._initialize_mathematical_truths()")