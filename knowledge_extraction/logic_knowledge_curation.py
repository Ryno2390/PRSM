#!/usr/bin/env python3
"""
Logic Knowledge Curation for NWTN WorldModelCore
================================================

Creates essential logic knowledge items manually from authoritative sources
to expand the WorldModelCore with fundamental logical principles.
"""

import json
import os
from typing import Dict, List, Any

def create_essential_logic_knowledge() -> List[Dict[str, Any]]:
    """
    Create essential logic knowledge items manually from authoritative sources
    
    Returns:
        List of essential logic knowledge items
    """
    
    essential_knowledge = [
        # Classical Logic Principles
        {
            'content': 'A proposition is either true or false, but not both',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'propositional_logic',
            'mathematical_form': 'P ∨ ¬P ∧ ¬(P ∧ ¬P)',
            'dependencies': [],
            'references': ['Classical Logic', 'Aristotelian Logic', 'Logic textbooks'],
            'applicable_conditions': ['classical logic', 'bivalent logic'],
            'principle_name': 'Law of Excluded Middle'
        },
        {
            'content': 'A statement cannot be both true and false simultaneously',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'propositional_logic',
            'mathematical_form': '¬(P ∧ ¬P)',
            'dependencies': [],
            'references': ['Classical Logic', 'Aristotelian Logic', 'Logic textbooks'],
            'applicable_conditions': ['classical logic', 'non-contradiction'],
            'principle_name': 'Law of Non-Contradiction'
        },
        {
            'content': 'Everything is identical to itself',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'identity_logic',
            'mathematical_form': 'A = A',
            'dependencies': [],
            'references': ['Classical Logic', 'Identity theory', 'Logic textbooks'],
            'applicable_conditions': ['classical logic', 'identity relation'],
            'principle_name': 'Law of Identity'
        },
        
        # Logical Inference Rules
        {
            'content': 'If P implies Q and P is true, then Q is true',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'inference_rules',
            'mathematical_form': '((P → Q) ∧ P) → Q',
            'dependencies': [],
            'references': ['Propositional Logic', 'Deductive reasoning', 'Logic textbooks'],
            'applicable_conditions': ['deductive reasoning', 'material implication'],
            'principle_name': 'Modus Ponens'
        },
        {
            'content': 'If P implies Q and Q is false, then P is false',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'inference_rules',
            'mathematical_form': '((P → Q) ∧ ¬Q) → ¬P',
            'dependencies': [],
            'references': ['Propositional Logic', 'Deductive reasoning', 'Logic textbooks'],
            'applicable_conditions': ['deductive reasoning', 'material implication'],
            'principle_name': 'Modus Tollens'
        },
        {
            'content': 'If P implies Q and Q implies R, then P implies R',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'inference_rules',
            'mathematical_form': '((P → Q) ∧ (Q → R)) → (P → R)',
            'dependencies': [],
            'references': ['Propositional Logic', 'Transitivity', 'Logic textbooks'],
            'applicable_conditions': ['deductive reasoning', 'transitivity'],
            'principle_name': 'Hypothetical Syllogism'
        },
        {
            'content': 'If P or Q is true and P is false, then Q is true',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'inference_rules',
            'mathematical_form': '((P ∨ Q) ∧ ¬P) → Q',
            'dependencies': [],
            'references': ['Propositional Logic', 'Disjunctive syllogism', 'Logic textbooks'],
            'applicable_conditions': ['deductive reasoning', 'disjunction'],
            'principle_name': 'Disjunctive Syllogism'
        },
        
        # Logical Equivalences
        {
            'content': 'Double negation of a statement is equivalent to the original statement',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'logical_equivalences',
            'mathematical_form': '¬¬P ≡ P',
            'dependencies': [],
            'references': ['Propositional Logic', 'Logical equivalences', 'Logic textbooks'],
            'applicable_conditions': ['classical logic', 'negation'],
            'principle_name': 'Double Negation Law'
        },
        {
            'content': 'The conjunction of P and Q is equivalent to the conjunction of Q and P',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'logical_equivalences',
            'mathematical_form': 'P ∧ Q ≡ Q ∧ P',
            'dependencies': [],
            'references': ['Propositional Logic', 'Commutative laws', 'Logic textbooks'],
            'applicable_conditions': ['classical logic', 'conjunction'],
            'principle_name': 'Commutative Law of Conjunction'
        },
        {
            'content': 'The disjunction of P and Q is equivalent to the disjunction of Q and P',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'logical_equivalences',
            'mathematical_form': 'P ∨ Q ≡ Q ∨ P',
            'dependencies': [],
            'references': ['Propositional Logic', 'Commutative laws', 'Logic textbooks'],
            'applicable_conditions': ['classical logic', 'disjunction'],
            'principle_name': 'Commutative Law of Disjunction'
        },
        {
            'content': 'The conjunction is distributive over disjunction',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'logical_equivalences',
            'mathematical_form': 'P ∧ (Q ∨ R) ≡ (P ∧ Q) ∨ (P ∧ R)',
            'dependencies': [],
            'references': ['Propositional Logic', 'Distributive laws', 'Logic textbooks'],
            'applicable_conditions': ['classical logic', 'conjunction', 'disjunction'],
            'principle_name': 'Distributive Law of Conjunction over Disjunction'
        },
        {
            'content': 'The disjunction is distributive over conjunction',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'logical_equivalences',
            'mathematical_form': 'P ∨ (Q ∧ R) ≡ (P ∨ Q) ∧ (P ∨ R)',
            'dependencies': [],
            'references': ['Propositional Logic', 'Distributive laws', 'Logic textbooks'],
            'applicable_conditions': ['classical logic', 'conjunction', 'disjunction'],
            'principle_name': 'Distributive Law of Disjunction over Conjunction'
        },
        
        # De Morgan's Laws
        {
            'content': 'The negation of a conjunction is equivalent to the disjunction of the negations',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'logical_equivalences',
            'mathematical_form': '¬(P ∧ Q) ≡ ¬P ∨ ¬Q',
            'dependencies': [],
            'references': ['De Morgan\'s Laws', 'Propositional Logic', 'Logic textbooks'],
            'applicable_conditions': ['classical logic', 'negation', 'conjunction', 'disjunction'],
            'principle_name': 'De Morgan\'s Law for Conjunction'
        },
        {
            'content': 'The negation of a disjunction is equivalent to the conjunction of the negations',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'logical_equivalences',
            'mathematical_form': '¬(P ∨ Q) ≡ ¬P ∧ ¬Q',
            'dependencies': [],
            'references': ['De Morgan\'s Laws', 'Propositional Logic', 'Logic textbooks'],
            'applicable_conditions': ['classical logic', 'negation', 'conjunction', 'disjunction'],
            'principle_name': 'De Morgan\'s Law for Disjunction'
        },
        
        # Quantifier Logic
        {
            'content': 'If a property holds for all elements in a domain, then it holds for any specific element',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'quantifier_logic',
            'mathematical_form': '∀x P(x) → P(a)',
            'dependencies': [],
            'references': ['Predicate Logic', 'Universal quantification', 'Logic textbooks'],
            'applicable_conditions': ['predicate logic', 'universal quantification'],
            'principle_name': 'Universal Instantiation'
        },
        {
            'content': 'If a property holds for a specific element, then there exists an element with that property',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'quantifier_logic',
            'mathematical_form': 'P(a) → ∃x P(x)',
            'dependencies': [],
            'references': ['Predicate Logic', 'Existential quantification', 'Logic textbooks'],
            'applicable_conditions': ['predicate logic', 'existential quantification'],
            'principle_name': 'Existential Generalization'
        },
        
        # Logical Implication
        {
            'content': 'Material implication is equivalent to disjunction with negation of antecedent',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'implication',
            'mathematical_form': 'P → Q ≡ ¬P ∨ Q',
            'dependencies': [],
            'references': ['Propositional Logic', 'Material implication', 'Logic textbooks'],
            'applicable_conditions': ['classical logic', 'material implication'],
            'principle_name': 'Material Implication Equivalence'
        },
        {
            'content': 'Biconditional is equivalent to conjunction of both implications',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'implication',
            'mathematical_form': 'P ↔ Q ≡ (P → Q) ∧ (Q → P)',
            'dependencies': [],
            'references': ['Propositional Logic', 'Biconditional', 'Logic textbooks'],
            'applicable_conditions': ['classical logic', 'biconditional'],
            'principle_name': 'Biconditional Equivalence'
        },
        
        # Logical Consistency
        {
            'content': 'A set of statements is consistent if they can all be true simultaneously',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'consistency',
            'mathematical_form': 'Consistent(S) ↔ ∃I (∀p∈S: I(p) = true)',
            'dependencies': [],
            'references': ['Logic textbooks', 'Consistency theory', 'Model theory'],
            'applicable_conditions': ['classical logic', 'consistency'],
            'principle_name': 'Logical Consistency'
        },
        {
            'content': 'A set of statements is inconsistent if it leads to a contradiction',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'consistency',
            'mathematical_form': 'Inconsistent(S) ↔ S ⊢ P ∧ ¬P',
            'dependencies': [],
            'references': ['Logic textbooks', 'Consistency theory', 'Proof theory'],
            'applicable_conditions': ['classical logic', 'contradiction'],
            'principle_name': 'Logical Inconsistency'
        },
        
        # Proof Theory
        {
            'content': 'A valid argument preserves truth from premises to conclusion',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'proof_theory',
            'mathematical_form': 'Valid(P₁,...,Pₙ ⊢ Q) ↔ (P₁ ∧ ... ∧ Pₙ) → Q is tautology',
            'dependencies': [],
            'references': ['Logic textbooks', 'Proof theory', 'Validity theory'],
            'applicable_conditions': ['deductive reasoning', 'validity'],
            'principle_name': 'Logical Validity'
        },
        {
            'content': 'A sound argument is both valid and has true premises',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'proof_theory',
            'mathematical_form': 'Sound(P₁,...,Pₙ ⊢ Q) ↔ Valid(P₁,...,Pₙ ⊢ Q) ∧ (P₁ ∧ ... ∧ Pₙ)',
            'dependencies': [],
            'references': ['Logic textbooks', 'Proof theory', 'Soundness theory'],
            'applicable_conditions': ['deductive reasoning', 'soundness'],
            'principle_name': 'Logical Soundness'
        },
        
        # Completeness and Decidability
        {
            'content': 'In propositional logic, every valid formula is provable',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'completeness',
            'mathematical_form': '⊨ P ↔ ⊢ P',
            'dependencies': [],
            'references': ['Completeness theorem', 'Propositional logic', 'Logic textbooks'],
            'applicable_conditions': ['propositional logic', 'completeness'],
            'principle_name': 'Completeness of Propositional Logic'
        },
        {
            'content': 'Propositional logic is decidable - there exists an algorithm to determine validity',
            'certainty': 0.9999,
            'domain': 'logic',
            'category': 'decidability',
            'mathematical_form': 'Decidable(PropositionalLogic)',
            'dependencies': [],
            'references': ['Decidability theory', 'Propositional logic', 'Logic textbooks'],
            'applicable_conditions': ['propositional logic', 'decidability'],
            'principle_name': 'Decidability of Propositional Logic'
        }
    ]
    
    return essential_knowledge

def expand_world_model_logic():
    """Expand the WorldModelCore with essential logic knowledge"""
    
    print("🔣 Creating Essential Logic Knowledge Base")
    print("=" * 60)
    
    # Create essential logic knowledge
    essential_knowledge = create_essential_logic_knowledge()
    
    print(f"📚 Created {len(essential_knowledge)} essential logic knowledge items")
    
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
    output_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/processed_knowledge/logic_essential_manual_v1.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(essential_knowledge, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Essential logic knowledge saved to:")
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
        'expansion_impact': f"WorldModelCore will expand from ~90 to {90 + len(essential_knowledge)} items"
    }
    
    summary_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/logic/essential_logic_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Summary saved to: {summary_file}")
    
    print(f"\n🎉 Essential Logic Knowledge Base Complete!")
    print(f"💡 WorldModelCore will expand from ~90 to {90 + len(essential_knowledge)} items!")
    print(f"🧠 High-quality knowledge with {avg_certainty:.1%} average certainty")
    
    # Show knowledge distribution
    print(f"\n📋 Knowledge Distribution:")
    for category, items in categories.items():
        print(f"   {category}: {len(items)} items")
        for item in items:
            print(f"      - {item['principle_name']}")
    
    return essential_knowledge

if __name__ == "__main__":
    essential_knowledge = expand_world_model_logic()
    
    print(f"\n✅ Manual logic knowledge curation successful!")
    print(f"🚀 Ready to integrate with NWTN WorldModelCore system!")
    print(f"📊 Total knowledge items: {len(essential_knowledge)}")
    print(f"🎯 Next step: Update WorldModelCore._initialize_logical_principles()")