#!/usr/bin/env python3
"""
Computer Science Knowledge Curation for NWTN WorldModelCore
===========================================================

Creates essential computer science knowledge items manually from authoritative sources
to expand the WorldModelCore with fundamental computational principles.
"""

import json
import os
from typing import Dict, List, Any

def create_essential_computer_science_knowledge() -> List[Dict[str, Any]]:
    """
    Create essential computer science knowledge items manually from authoritative sources
    
    Returns:
        List of essential computer science knowledge items
    """
    
    essential_knowledge = [
        # Algorithms and Data Structures
        {
            'content': 'Algorithms are step-by-step procedures for solving computational problems',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'algorithms',
            'mathematical_form': 'Algorithm: Input → Process → Output',
            'dependencies': [],
            'references': ['Algorithm textbooks', 'Computer Science fundamentals'],
            'applicable_conditions': ['computational problems'],
            'principle_name': 'Algorithm Definition'
        },
        {
            'content': 'Time complexity measures how algorithm performance scales with input size',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'algorithms',
            'mathematical_form': 'T(n) = O(f(n))',
            'dependencies': [],
            'references': ['Algorithm Analysis', 'Big O notation'],
            'applicable_conditions': ['algorithm analysis'],
            'principle_name': 'Time Complexity'
        },
        {
            'content': 'Space complexity measures memory usage of algorithms',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'algorithms',
            'mathematical_form': 'S(n) = O(g(n))',
            'dependencies': [],
            'references': ['Algorithm Analysis', 'Big O notation'],
            'applicable_conditions': ['algorithm analysis'],
            'principle_name': 'Space Complexity'
        },
        {
            'content': 'Binary search has O(log n) time complexity for sorted arrays',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'algorithms',
            'mathematical_form': 'Binary Search: O(log n)',
            'dependencies': [],
            'references': ['Algorithm textbooks', 'Search algorithms'],
            'applicable_conditions': ['sorted data'],
            'principle_name': 'Binary Search Complexity'
        },
        {
            'content': 'Sorting algorithms arrange data in specified order',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'algorithms',
            'mathematical_form': 'Sort: Unordered → Ordered',
            'dependencies': [],
            'references': ['Algorithm textbooks', 'Sorting algorithms'],
            'applicable_conditions': ['comparable data'],
            'principle_name': 'Sorting Algorithms'
        },
        
        # Programming Languages
        {
            'content': 'Programming languages provide syntax and semantics for expressing computations',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'programming_languages',
            'mathematical_form': 'Language: Syntax + Semantics → Computation',
            'dependencies': [],
            'references': ['Programming Language textbooks', 'Language theory'],
            'applicable_conditions': ['software development'],
            'principle_name': 'Programming Language Definition'
        },
        {
            'content': 'Compiled languages translate source code to machine code before execution',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'programming_languages',
            'mathematical_form': 'Source Code → Compiler → Machine Code',
            'dependencies': [],
            'references': ['Compiler textbooks', 'Programming languages'],
            'applicable_conditions': ['compiled languages'],
            'principle_name': 'Compilation Process'
        },
        {
            'content': 'Interpreted languages execute source code directly through an interpreter',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'programming_languages',
            'mathematical_form': 'Source Code → Interpreter → Execution',
            'dependencies': [],
            'references': ['Interpreter textbooks', 'Programming languages'],
            'applicable_conditions': ['interpreted languages'],
            'principle_name': 'Interpretation Process'
        },
        {
            'content': 'Variables store data values that can be referenced and modified',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'programming_languages',
            'mathematical_form': 'Variable: Name → Memory Location → Value',
            'dependencies': [],
            'references': ['Programming textbooks', 'Language fundamentals'],
            'applicable_conditions': ['programming'],
            'principle_name': 'Variable Concept'
        },
        
        # Data Structures
        {
            'content': 'Arrays store elements in contiguous memory locations',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'data_structures',
            'mathematical_form': 'Array[i] = Base Address + i × Element Size',
            'dependencies': [],
            'references': ['Data Structure textbooks', 'Array implementation'],
            'applicable_conditions': ['sequential data'],
            'principle_name': 'Array Structure'
        },
        {
            'content': 'Linked lists store elements in nodes connected by pointers',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'data_structures',
            'mathematical_form': 'Node: Data + Pointer → Next Node',
            'dependencies': [],
            'references': ['Data Structure textbooks', 'Linked list implementation'],
            'applicable_conditions': ['dynamic data'],
            'principle_name': 'Linked List Structure'
        },
        {
            'content': 'Stacks follow Last-In-First-Out (LIFO) principle',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'data_structures',
            'mathematical_form': 'Stack: LIFO (Last In, First Out)',
            'dependencies': [],
            'references': ['Data Structure textbooks', 'Stack implementation'],
            'applicable_conditions': ['LIFO operations'],
            'principle_name': 'Stack LIFO Principle'
        },
        {
            'content': 'Queues follow First-In-First-Out (FIFO) principle',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'data_structures',
            'mathematical_form': 'Queue: FIFO (First In, First Out)',
            'dependencies': [],
            'references': ['Data Structure textbooks', 'Queue implementation'],
            'applicable_conditions': ['FIFO operations'],
            'principle_name': 'Queue FIFO Principle'
        },
        
        # Computer Systems
        {
            'content': 'Computer systems consist of hardware and software components',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'computer_systems',
            'mathematical_form': 'Computer System = Hardware + Software',
            'dependencies': [],
            'references': ['Computer Systems textbooks', 'Computer Architecture'],
            'applicable_conditions': ['computing systems'],
            'principle_name': 'Computer System Architecture'
        },
        {
            'content': 'CPU executes instructions in fetch-decode-execute cycle',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'computer_systems',
            'mathematical_form': 'CPU Cycle: Fetch → Decode → Execute',
            'dependencies': [],
            'references': ['Computer Architecture textbooks', 'CPU design'],
            'applicable_conditions': ['CPU operations'],
            'principle_name': 'CPU Instruction Cycle'
        },
        {
            'content': 'Memory hierarchy balances speed, capacity, and cost',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'computer_systems',
            'mathematical_form': 'Memory: Cache → RAM → Storage',
            'dependencies': [],
            'references': ['Computer Architecture textbooks', 'Memory systems'],
            'applicable_conditions': ['memory design'],
            'principle_name': 'Memory Hierarchy'
        },
        
        # Operating Systems
        {
            'content': 'Operating systems manage computer hardware and software resources',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'operating_systems',
            'mathematical_form': 'OS: Resource Management + Abstraction',
            'dependencies': [],
            'references': ['Operating Systems textbooks', 'OS design'],
            'applicable_conditions': ['computer systems'],
            'principle_name': 'Operating System Function'
        },
        {
            'content': 'Processes are programs in execution with allocated resources',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'operating_systems',
            'mathematical_form': 'Process = Program + Execution Context',
            'dependencies': [],
            'references': ['Operating Systems textbooks', 'Process management'],
            'applicable_conditions': ['program execution'],
            'principle_name': 'Process Concept'
        },
        {
            'content': 'Threads are lightweight processes that share memory space',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'operating_systems',
            'mathematical_form': 'Thread: Lightweight Process + Shared Memory',
            'dependencies': [],
            'references': ['Operating Systems textbooks', 'Thread management'],
            'applicable_conditions': ['concurrent execution'],
            'principle_name': 'Thread Concept'
        },
        
        # Networks
        {
            'content': 'Computer networks connect devices to enable communication',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'networks',
            'mathematical_form': 'Network: Nodes + Links + Protocols',
            'dependencies': [],
            'references': ['Computer Networks textbooks', 'Network protocols'],
            'applicable_conditions': ['distributed systems'],
            'principle_name': 'Computer Network Definition'
        },
        {
            'content': 'TCP/IP provides reliable, ordered delivery of data packets',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'networks',
            'mathematical_form': 'TCP/IP: Reliable + Ordered + Packet Delivery',
            'dependencies': [],
            'references': ['Network protocol textbooks', 'TCP/IP specification'],
            'applicable_conditions': ['internet communication'],
            'principle_name': 'TCP/IP Protocol'
        },
        {
            'content': 'OSI model defines seven layers of network communication',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'networks',
            'mathematical_form': 'OSI: 7 Layers (Physical → Application)',
            'dependencies': [],
            'references': ['Network textbooks', 'OSI model'],
            'applicable_conditions': ['network design'],
            'principle_name': 'OSI Model'
        },
        
        # Databases
        {
            'content': 'Databases store, organize, and retrieve structured data',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'databases',
            'mathematical_form': 'Database: Data + Structure + Operations',
            'dependencies': [],
            'references': ['Database textbooks', 'Data management'],
            'applicable_conditions': ['data storage'],
            'principle_name': 'Database Definition'
        },
        {
            'content': 'Relational databases organize data in tables with relationships',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'databases',
            'mathematical_form': 'Relational DB: Tables + Relationships + SQL',
            'dependencies': [],
            'references': ['Database textbooks', 'Relational model'],
            'applicable_conditions': ['structured data'],
            'principle_name': 'Relational Database Model'
        },
        {
            'content': 'ACID properties ensure database transaction reliability',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'databases',
            'mathematical_form': 'ACID: Atomicity + Consistency + Isolation + Durability',
            'dependencies': [],
            'references': ['Database textbooks', 'Transaction processing'],
            'applicable_conditions': ['database transactions'],
            'principle_name': 'ACID Properties'
        },
        
        # Software Engineering
        {
            'content': 'Software engineering applies engineering principles to software development',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'software_engineering',
            'mathematical_form': 'Software Engineering = Engineering Principles + Software Development',
            'dependencies': [],
            'references': ['Software Engineering textbooks', 'Software development'],
            'applicable_conditions': ['software projects'],
            'principle_name': 'Software Engineering Definition'
        },
        {
            'content': 'Version control systems track changes to source code over time',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'software_engineering',
            'mathematical_form': 'Version Control: Code + History + Collaboration',
            'dependencies': [],
            'references': ['Software Engineering textbooks', 'Version control'],
            'applicable_conditions': ['software development'],
            'principle_name': 'Version Control'
        },
        {
            'content': 'Testing verifies that software meets specified requirements',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'software_engineering',
            'mathematical_form': 'Testing: Requirements → Verification → Quality',
            'dependencies': [],
            'references': ['Software Engineering textbooks', 'Software testing'],
            'applicable_conditions': ['software quality'],
            'principle_name': 'Software Testing'
        },
        
        # Security
        {
            'content': 'Computer security protects systems from unauthorized access and attacks',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'security',
            'mathematical_form': 'Security: Confidentiality + Integrity + Availability',
            'dependencies': [],
            'references': ['Computer Security textbooks', 'Cybersecurity'],
            'applicable_conditions': ['computing systems'],
            'principle_name': 'Computer Security CIA Triad'
        },
        {
            'content': 'Encryption transforms plaintext into ciphertext to protect data',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'security',
            'mathematical_form': 'Encryption: Plaintext + Key → Ciphertext',
            'dependencies': [],
            'references': ['Cryptography textbooks', 'Data encryption'],
            'applicable_conditions': ['data protection'],
            'principle_name': 'Data Encryption'
        },
        {
            'content': 'Authentication verifies the identity of users or systems',
            'certainty': 0.9999,
            'domain': 'computer_science',
            'category': 'security',
            'mathematical_form': 'Authentication: Identity Claims → Verification → Access',
            'dependencies': [],
            'references': ['Security textbooks', 'Authentication systems'],
            'applicable_conditions': ['access control'],
            'principle_name': 'Authentication'
        }
    ]
    
    return essential_knowledge

def expand_world_model_computer_science():
    """Expand the WorldModelCore with essential computer science knowledge"""
    
    print("💻 Creating Essential Computer Science Knowledge Base")
    print("=" * 60)
    
    # Create essential computer science knowledge
    essential_knowledge = create_essential_computer_science_knowledge()
    
    print(f"📚 Created {len(essential_knowledge)} essential computer science knowledge items")
    
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
    output_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/processed_knowledge/computer_science_essential_manual_v1.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(essential_knowledge, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Essential computer science knowledge saved to:")
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
        'expansion_impact': f"WorldModelCore will expand from ~181 to {181 + len(essential_knowledge)} items"
    }
    
    summary_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/computer_science/essential_computer_science_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Summary saved to: {summary_file}")
    
    print(f"\n🎉 Essential Computer Science Knowledge Base Complete!")
    print(f"💡 WorldModelCore will expand from ~181 to {181 + len(essential_knowledge)} items!")
    print(f"🧠 High-quality knowledge with {avg_certainty:.1%} average certainty")
    
    # Show knowledge distribution
    print(f"\n📋 Knowledge Distribution:")
    for category, items in categories.items():
        print(f"   {category}: {len(items)} items")
        for item in items:
            print(f"      - {item['principle_name']}")
    
    return essential_knowledge

if __name__ == "__main__":
    essential_knowledge = expand_world_model_computer_science()
    
    print(f"\n✅ Manual computer science knowledge curation successful!")
    print(f"🚀 Ready to integrate with NWTN WorldModelCore system!")
    print(f"📊 Total knowledge items: {len(essential_knowledge)}")
    print(f"🎯 Next step: Update WorldModelCore with computer science knowledge")
    print(f"🌌 Ready for astronomy ZIM processing!")