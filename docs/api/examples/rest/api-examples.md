# PRSM REST API Examples

**Comprehensive examples for interacting with PRSM APIs using HTTP requests**

## 🎯 Overview

This collection provides practical examples for integrating with PRSM using REST APIs. All examples include complete request/response cycles with explanations.

## 🔐 Authentication

All API requests require authentication. PRSM supports multiple methods:

### API Key Authentication
```bash
export PRSM_API_KEY="your_api_key_here"
```

### Bearer Token
```bash
curl -H "Authorization: Bearer $PRSM_API_KEY" \
     https://api.prsm.ai/v1/agents
```

### FTNS Token Authentication
```bash
curl -H "X-FTNS-Token: your_ftns_token" \
     https://api.prsm.ai/v1/agents
```

## 🤖 Agent Management Examples

### Create a Research Agent

**Request:**
```bash
curl -X POST https://api.prsm.ai/v1/agents \
  -H "Authorization: Bearer $PRSM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "biomedical_researcher",
    "description": "Specialized in biomedical research and clinical analysis",
    "type": "researcher",
    "model_provider": "openai",
    "model_name": "gpt-4",
    "capabilities": [
      "literature_search",
      "clinical_analysis",
      "data_interpretation",
      "hypothesis_generation"
    ],
    "specialized_knowledge": "biomedicine,clinical_research,pharmacology,genetics",
    "configuration": {
      "temperature": 0.7,
      "max_tokens": 2048,
      "top_p": 0.9
    },
    "tools": ["web_search", "file_upload", "data_analysis"],
    "memory_type": "persistent",
    "collaboration_mode": "team"
  }'
```

**Response:**
```json
{
  "id": "agent_bio_research_001",
  "name": "biomedical_researcher",
  "description": "Specialized in biomedical research and clinical analysis",
  "type": "researcher",
  "status": "active",
  "model_provider": "openai",
  "model_name": "gpt-4",
  "capabilities": [
    "literature_search",
    "clinical_analysis",
    "data_interpretation",
    "hypothesis_generation"
  ],
  "specialized_knowledge": "biomedicine,clinical_research,pharmacology,genetics",
  "configuration": {
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.9,
    "context_window": 8192
  },
  "tools": ["web_search", "file_upload", "data_analysis"],
  "memory_type": "persistent",
  "collaboration_mode": "team",
  "created_at": "2024-12-21T16:59:23Z",
  "updated_at": "2024-12-21T16:59:23Z",
  "performance_metrics": {
    "success_rate": 0.0,
    "avg_response_time": 0.0,
    "total_executions": 0,
    "error_rate": 0.0,
    "cost_per_execution": 0.0
  }
}
```

### Execute Research Task

**Request:**
```bash
curl -X POST https://api.prsm.ai/v1/agents/agent_bio_research_001/execute \
  -H "Authorization: Bearer $PRSM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Analyze the latest research on CAR-T cell therapy for treating acute lymphoblastic leukemia. Focus on: 1) Efficacy rates in pediatric vs adult populations, 2) Common adverse effects and management strategies, 3) Recent improvements in cell manufacturing, 4) Cost-effectiveness compared to traditional chemotherapy. Provide a comprehensive analysis with clinical recommendations.",
    "context": {
      "domain": "hematology_oncology",
      "priority": "high",
      "deadline": "2024-12-22T18:00:00Z",
      "target_audience": "clinical_researchers"
    },
    "data": {
      "urls": [
        "https://pubmed.ncbi.nlm.nih.gov/search?term=CAR-T+ALL+2024",
        "https://clinicaltrials.gov/search?term=CAR-T+leukemia"
      ],
      "parameters": {
        "focus_areas": ["efficacy", "safety", "manufacturing", "economics"],
        "time_frame": "2022-2024",
        "study_types": ["clinical_trials", "meta_analyses", "systematic_reviews"]
      }
    },
    "execution_options": {
      "timeout": 600,
      "max_iterations": 8,
      "stream_response": false,
      "save_to_memory": true
    }
  }'
```

**Response:**
```json
{
  "execution_id": "exec_cart_analysis_001",
  "status": "completed",
  "output": {
    "text": "# CAR-T Cell Therapy for Acute Lymphoblastic Leukemia: Comprehensive Analysis\n\n## Executive Summary\n\nCAR-T cell therapy has revolutionized treatment for relapsed/refractory acute lymphoblastic leukemia (ALL), with recent advances significantly improving outcomes...\n\n## 1. Efficacy Analysis: Pediatric vs Adult Populations\n\n### Pediatric Outcomes (Ages 3-21)\n- **Complete Remission Rate**: 81-94% across recent trials\n- **Overall Survival at 12 months**: 76-89%\n- **Event-Free Survival**: 58-73% at 12 months\n- **Key Studies**: ELIANA (tisagenlecleucel), ZUMA-3 (KTE-X19)\n\n### Adult Outcomes (Ages 22+)\n- **Complete Remission Rate**: 70-85% in recent studies\n- **Overall Survival at 12 months**: 65-78%\n- **Event-Free Survival**: 45-62% at 12 months\n- **Notable Difference**: Lower baseline performance status impacts outcomes\n\n## 2. Adverse Effects and Management\n\n### Cytokine Release Syndrome (CRS)\n- **Incidence**: 85-95% of patients experience some grade of CRS\n- **Severe CRS (Grade 3-4)**: 15-25% of cases\n- **Management**: Tocilizumab as first-line, corticosteroids for refractory cases\n- **Recent Improvements**: Enhanced monitoring protocols reduce severe cases by 30%\n\n### Immune Effector Cell-Associated Neurotoxicity Syndrome (ICANS)\n- **Incidence**: 40-65% across products\n- **Severe ICANS (Grade 3-4)**: 10-20%\n- **Management**: Early corticosteroid intervention, supportive care\n- **Innovation**: Predictive biomarkers (IL-6, ferritin) enable proactive management\n\n## 3. Manufacturing Improvements (2022-2024)\n\n### Process Enhancements\n- **Vein-to-Vein Time**: Reduced from 22-28 days to 14-18 days\n- **Manufacturing Success Rate**: Improved from 89% to 96%\n- **Point-of-Care Manufacturing**: Emerging technology reducing logistics\n\n### Quality Improvements\n- **Cell Viability**: Enhanced cryopreservation maintains >85% viability\n- **Potency Assays**: Standardized testing ensures consistent product quality\n- **Contamination Rates**: Reduced to <2% through enhanced sterility protocols\n\n## 4. Cost-Effectiveness Analysis\n\n### Direct Costs\n- **CAR-T Therapy**: $450,000-$500,000 per treatment\n- **Traditional Chemotherapy**: $150,000-$200,000 over treatment course\n- **Bone Marrow Transplant**: $300,000-$400,000 including complications\n\n### Value-Based Analysis\n- **Quality-Adjusted Life Years (QALYs)**: CAR-T provides 2.3-3.1 additional QALYs\n- **Cost per QALY**: $145,000-$195,000 (within acceptable thresholds)\n- **Long-term Savings**: Reduced hospitalization and supportive care costs\n\n## Clinical Recommendations\n\n### Patient Selection\n1. **Optimal Candidates**: Age <65, good performance status, adequate organ function\n2. **Timing**: Consider after first relapse in high-risk patients\n3. **Biomarkers**: CD19 expression >20%, absence of CNS involvement\n\n### Treatment Protocol\n1. **Pre-treatment**: Bridging therapy to control disease burden\n2. **Lymphodepletion**: Fludarabine + cyclophosphamide standard regimen\n3. **Post-CAR-T**: Structured monitoring for CRS/ICANS for 30 days\n\n### Future Directions\n1. **Combination Therapies**: CAR-T + checkpoint inhibitors under investigation\n2. **Allogeneic Products**: Off-the-shelf solutions to reduce cost and time\n3. **Enhanced CAR Designs**: Dual-target CARs to prevent antigen escape\n\n## Conclusion\n\nCAR-T cell therapy represents a paradigm shift in ALL treatment, with demonstrated superiority over conventional approaches in relapsed/refractory disease. Recent manufacturing improvements and adverse event management protocols have enhanced the therapeutic index. While cost remains a consideration, the value proposition is increasingly favorable given long-term outcomes and quality of life benefits.\n\n**Evidence Level**: A (Multiple high-quality RCTs and meta-analyses)\n**Recommendation Strength**: Strong for relapsed/refractory ALL patients meeting selection criteria",
    "data": {
      "confidence_score": 0.94,
      "evidence_quality": "high",
      "sources_reviewed": 47,
      "clinical_trials_analyzed": 12,
      "key_findings": [
        "CAR-T therapy shows superior efficacy in pediatric vs adult populations",
        "Manufacturing improvements have reduced production time by 35%",
        "Cost-effectiveness is favorable when considering long-term outcomes",
        "Adverse event management has significantly improved safety profile"
      ],
      "recommendations_summary": [
        "Consider CAR-T after first relapse in high-risk patients",
        "Implement enhanced monitoring protocols for CRS/ICANS",
        "Evaluate cost-effectiveness using QALY metrics",
        "Investigate combination therapies for enhanced efficacy"
      ]
    },
    "files": [
      "cart_efficacy_analysis.pdf",
      "adverse_events_summary.xlsx",
      "cost_effectiveness_model.json"
    ]
  },
  "metadata": {
    "execution_time": 287.4,
    "tokens_used": 3847,
    "cost": 0.42,
    "model_calls": 6,
    "sources_accessed": 47,
    "quality_score": 0.94
  },
  "started_at": "2024-12-21T17:50:00Z",
  "completed_at": "2024-12-21T17:54:47Z"
}
```

### List All Agents

**Request:**
```bash
curl -X GET "https://api.prsm.ai/v1/agents?limit=10&sort_by=created_at&sort_order=desc" \
  -H "Authorization: Bearer $PRSM_API_KEY"
```

**Response:**
```json
{
  "agents": [
    {
      "id": "agent_bio_research_001",
      "name": "biomedical_researcher",
      "type": "researcher",
      "status": "active",
      "model_provider": "openai",
      "created_at": "2024-12-21T16:59:23Z",
      "performance_metrics": {
        "success_rate": 0.97,
        "avg_response_time": 3.2,
        "total_executions": 34,
        "error_rate": 0.03,
        "cost_per_execution": 0.38
      }
    },
    {
      "id": "agent_data_analyst_002",
      "name": "clinical_data_analyst", 
      "type": "analyst",
      "status": "active",
      "model_provider": "anthropic",
      "created_at": "2024-12-21T15:30:11Z",
      "performance_metrics": {
        "success_rate": 0.95,
        "avg_response_time": 2.1,
        "total_executions": 67,
        "error_rate": 0.05,
        "cost_per_execution": 0.22
      }
    }
  ],
  "total": 25,
  "limit": 10,
  "offset": 0,
  "has_more": true
}
```

## 🌐 P2P Network Examples

### Join P2P Network

**Request:**
```bash
curl -X POST https://api.prsm.ai/v1/p2p/nodes \
  -H "Authorization: Bearer $PRSM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "node_type": "inference_provider",
    "capabilities": [
      "llama2-7b",
      "llama2-13b", 
      "stable-diffusion-xl",
      "whisper-large"
    ],
    "resources": {
      "gpu_memory": "24GB",
      "gpu_model": "RTX 4090",
      "cpu_cores": 16,
      "ram": "64GB",
      "storage": "2TB NVMe",
      "bandwidth": "1Gbps",
      "uptime": "99.5%"
    },
    "pricing": {
      "base_rate": 0.0008,
      "currency": "FTNS",
      "minimum_duration": 60,
      "availability_schedule": "24/7"
    },
    "network_config": {
      "max_concurrent_requests": 10,
      "timeout_seconds": 300,
      "retry_attempts": 3
    }
  }'
```

**Response:**
```json
{
  "node_id": "node_gpu_provider_789",
  "status": "active",
  "network_address": "192.168.1.100:8080",
  "capabilities": [
    "llama2-7b",
    "llama2-13b",
    "stable-diffusion-xl", 
    "whisper-large"
  ],
  "resources": {
    "gpu_memory": "24GB",
    "gpu_model": "RTX 4090",
    "cpu_cores": 16,
    "ram": "64GB",
    "storage": "2TB NVMe",
    "bandwidth": "1Gbps",
    "uptime": "99.5%"
  },
  "pricing": {
    "base_rate": 0.0008,
    "currency": "FTNS",
    "minimum_duration": 60,
    "availability_schedule": "24/7"
  },
  "network_metrics": {
    "peer_count": 1247,
    "reputation_score": 0.0,
    "total_earnings": 0.0,
    "successful_requests": 0,
    "avg_response_time": 0.0
  },
  "joined_at": "2024-12-21T18:15:00Z"
}
```

### Discover Available Peers

**Request:**
```bash
curl -X GET "https://api.prsm.ai/v1/p2p/peers?capability=llama2-7b&max_latency=100&min_reputation=0.8" \
  -H "Authorization: Bearer $PRSM_API_KEY"
```

**Response:**
```json
{
  "peers": [
    {
      "node_id": "node_fast_inference_123",
      "capabilities": ["llama2-7b", "llama2-13b"],
      "location": {
        "region": "us-west-2",
        "country": "United States"
      },
      "performance": {
        "avg_latency": 45,
        "success_rate": 0.98,
        "reputation_score": 0.94
      },
      "pricing": {
        "rate": 0.0006,
        "currency": "FTNS"
      },
      "availability": "available"
    },
    {
      "node_id": "node_europe_gpu_456",
      "capabilities": ["llama2-7b", "stable-diffusion-xl"],
      "location": {
        "region": "eu-central-1",
        "country": "Germany"
      },
      "performance": {
        "avg_latency": 78,
        "success_rate": 0.96,
        "reputation_score": 0.91
      },
      "pricing": {
        "rate": 0.0007,
        "currency": "FTNS"
      },
      "availability": "available"
    }
  ],
  "total_peers": 1247,
  "filtered_peers": 23,
  "query_time": 0.034
}
```

### Execute P2P Inference

**Request:**
```bash
curl -X POST https://api.prsm.ai/v1/p2p/inference \
  -H "Authorization: Bearer $PRSM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2-7b",
    "prompt": "Explain the mechanism of action of CRISPR-Cas9 gene editing technology in simple terms suitable for undergraduate students.",
    "parameters": {
      "max_tokens": 500,
      "temperature": 0.7,
      "top_p": 0.9
    },
    "routing": {
      "strategy": "latency_optimized",
      "max_latency": 100,
      "min_reputation": 0.85,
      "preferred_regions": ["us-west", "us-east"]
    },
    "payment": {
      "max_cost": 0.05,
      "currency": "FTNS"
    }
  }'
```

**Response:**
```json
{
  "inference_id": "inf_crispr_explanation_001",
  "status": "completed",
  "output": {
    "text": "CRISPR-Cas9 is like a highly precise molecular scissors and GPS system combined. Here's how it works:\n\n**The GPS System (Guide RNA):**\nImagine you want to edit a specific sentence in a massive library of books. The guide RNA acts like a GPS that searches through all the DNA in a cell to find exactly the right 'sentence' (gene sequence) you want to change.\n\n**The Molecular Scissors (Cas9 Protein):**\nOnce the guide RNA finds the target location, it signals to the Cas9 protein, which acts like extremely precise molecular scissors. Cas9 cuts the DNA at exactly the right spot - think of it as cutting between two specific letters in a word.\n\n**The Editing Process:**\nAfter the cut is made, the cell's natural repair mechanisms kick in. Scientists can:\n1. Let the cell repair itself (often introducing small changes)\n2. Provide a new piece of DNA to insert at the cut site\n3. Use the cell's repair system to fix a faulty gene\n\n**Why It's Revolutionary:**\n- **Precision**: Unlike previous methods, CRISPR can target almost any gene with incredible accuracy\n- **Speed**: What used to take months can now be done in days\n- **Cost**: Much cheaper than older gene editing techniques\n- **Versatility**: Works in many different types of cells and organisms\n\n**Applications:**\n- Treating genetic diseases like sickle cell anemia\n- Developing disease-resistant crops\n- Creating better models for studying diseases\n- Potential future treatments for cancer and other conditions\n\nThink of CRISPR as upgrading from using a sledgehammer to using a surgeon's scalpel for genetic modifications!",
    "tokens_generated": 347
  },
  "routing_info": {
    "selected_node": "node_fast_inference_123",
    "selection_reason": "optimal_latency_reputation_balance",
    "latency": 67,
    "reputation_score": 0.94
  },
  "cost": {
    "amount": 0.0278,
    "currency": "FTNS",
    "usd_equivalent": 0.139
  },
  "performance": {
    "total_time": 2.1,
    "queue_time": 0.3,
    "inference_time": 1.8,
    "network_latency": 0.067
  },
  "completed_at": "2024-12-21T18:25:15Z"
}
```

## 🔬 Workflow Management Examples

### Create Scientific Workflow

**Request:**
```bash
curl -X POST https://api.prsm.ai/v1/workflows \
  -H "Authorization: Bearer $PRSM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "drug_discovery_pipeline",
    "description": "End-to-end computational drug discovery workflow",
    "steps": [
      {
        "step_id": "target_identification",
        "name": "Protein Target Identification",
        "agent_requirements": {
          "type": "specialist",
          "domain": "structural_biology",
          "capabilities": ["protein_analysis", "binding_site_prediction"]
        },
        "input_schema": {
          "disease_pathway": "string",
          "protein_sequences": "array",
          "known_inhibitors": "array"
        },
        "output_schema": {
          "target_proteins": "array",
          "binding_sites": "array",
          "druggability_scores": "object"
        },
        "timeout": 1800
      },
      {
        "step_id": "compound_screening", 
        "name": "Virtual Compound Screening",
        "agent_requirements": {
          "type": "specialist",
          "domain": "cheminformatics",
          "capabilities": ["molecular_docking", "compound_filtering"]
        },
        "input_schema": {
          "target_proteins": "array",
          "binding_sites": "array",
          "compound_libraries": "array"
        },
        "output_schema": {
          "candidate_compounds": "array",
          "docking_scores": "object",
          "binding_affinities": "object"
        },
        "depends_on": ["target_identification"],
        "timeout": 3600
      },
      {
        "step_id": "admet_prediction",
        "name": "ADMET Property Prediction",
        "agent_requirements": {
          "type": "specialist", 
          "domain": "pharmacokinetics",
          "capabilities": ["admet_modeling", "toxicity_prediction"]
        },
        "input_schema": {
          "candidate_compounds": "array"
        },
        "output_schema": {
          "admet_properties": "object",
          "toxicity_predictions": "object",
          "filtered_compounds": "array"
        },
        "depends_on": ["compound_screening"],
        "timeout": 2400
      },
      {
        "step_id": "lead_optimization",
        "name": "Lead Compound Optimization",
        "agent_requirements": {
          "type": "specialist",
          "domain": "medicinal_chemistry", 
          "capabilities": ["structure_optimization", "synthesis_planning"]
        },
        "input_schema": {
          "filtered_compounds": "array",
          "optimization_criteria": "object"
        },
        "output_schema": {
          "optimized_compounds": "array",
          "synthesis_routes": "array",
          "improvement_analysis": "object"
        },
        "depends_on": ["admet_prediction"],
        "timeout": 2400
      }
    ],
    "global_settings": {
      "max_parallel_steps": 2,
      "retry_failed_steps": true,
      "max_retries": 3,
      "notification_endpoints": ["webhook://your-endpoint.com/notifications"]
    }
  }'
```

**Response:**
```json
{
  "workflow_id": "wf_drug_discovery_001",
  "name": "drug_discovery_pipeline",
  "description": "End-to-end computational drug discovery workflow",
  "status": "created",
  "steps": [
    {
      "step_id": "target_identification",
      "name": "Protein Target Identification",
      "status": "pending",
      "agent_assigned": null,
      "estimated_duration": 1800
    },
    {
      "step_id": "compound_screening",
      "name": "Virtual Compound Screening", 
      "status": "waiting",
      "agent_assigned": null,
      "estimated_duration": 3600
    },
    {
      "step_id": "admet_prediction",
      "name": "ADMET Property Prediction",
      "status": "waiting",
      "agent_assigned": null,
      "estimated_duration": 2400
    },
    {
      "step_id": "lead_optimization",
      "name": "Lead Compound Optimization",
      "status": "waiting", 
      "agent_assigned": null,
      "estimated_duration": 2400
    }
  ],
  "estimated_total_duration": 10200,
  "created_at": "2024-12-21T18:30:00Z",
  "updated_at": "2024-12-21T18:30:00Z"
}
```

### Execute Workflow

**Request:**
```bash
curl -X POST https://api.prsm.ai/v1/workflows/wf_drug_discovery_001/execute \
  -H "Authorization: Bearer $PRSM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input_data": {
      "disease_pathway": "Alzheimer disease amyloid pathway",
      "protein_sequences": [
        "MKTVRQERLKSDHISENDG...",
        "MLPGLALLLLAAWTARALEV..."
      ],
      "known_inhibitors": [
        "donepezil",
        "memantine",
        "rivastigmine"
      ],
      "compound_libraries": [
        "chembl_approved_drugs",
        "zinc_natural_products",
        "enamine_building_blocks"
      ],
      "optimization_criteria": {
        "target_selectivity": ">100-fold",
        "blood_brain_barrier": "permeable",
        "oral_bioavailability": ">30%",
        "half_life": ">8hours"
      }
    },
    "execution_options": {
      "priority": "high",
      "notify_on_completion": true,
      "save_intermediate_results": true
    }
  }'
```

**Response:**
```json
{
  "execution_id": "exec_wf_drug_001",
  "workflow_id": "wf_drug_discovery_001",
  "status": "running",
  "current_step": "target_identification",
  "progress": {
    "completed_steps": 0,
    "total_steps": 4,
    "percentage": 0
  },
  "step_status": [
    {
      "step_id": "target_identification",
      "status": "running",
      "agent_assigned": "agent_structural_bio_003",
      "started_at": "2024-12-21T18:32:00Z",
      "estimated_completion": "2024-12-21T19:02:00Z"
    },
    {
      "step_id": "compound_screening",
      "status": "waiting",
      "agent_assigned": null
    },
    {
      "step_id": "admet_prediction", 
      "status": "waiting",
      "agent_assigned": null
    },
    {
      "step_id": "lead_optimization",
      "status": "waiting",
      "agent_assigned": null
    }
  ],
  "started_at": "2024-12-21T18:32:00Z",
  "estimated_completion": "2024-12-21T21:22:00Z"
}
```

## 💰 Cost Management Examples

### Get Usage Analytics

**Request:**
```bash
curl -X GET "https://api.prsm.ai/v1/cost/usage?start_date=2024-12-01&end_date=2024-12-21&granularity=day" \
  -H "Authorization: Bearer $PRSM_API_KEY"
```

**Response:**
```json
{
  "period": {
    "start_date": "2024-12-01T00:00:00Z",
    "end_date": "2024-12-21T23:59:59Z",
    "granularity": "day"
  },
  "summary": {
    "total_cost": 1247.89,
    "total_requests": 8934,
    "total_tokens": 2847392,
    "avg_cost_per_request": 0.14,
    "avg_cost_per_token": 0.00044
  },
  "breakdown_by_service": {
    "agent_execution": {
      "cost": 892.34,
      "percentage": 71.5,
      "requests": 3421,
      "tokens": 1847293
    },
    "p2p_inference": {
      "cost": 234.67,
      "percentage": 18.8,
      "requests": 4123,
      "tokens": 756834
    },
    "workflow_execution": {
      "cost": 89.45,
      "percentage": 7.2,
      "requests": 234,
      "tokens": 178923
    },
    "data_storage": {
      "cost": 31.43,
      "percentage": 2.5,
      "storage_gb": 456.7
    }
  },
  "breakdown_by_provider": {
    "openai": {
      "cost": 567.89,
      "percentage": 45.5,
      "tokens": 1234567
    },
    "anthropic": {
      "cost": 334.12,
      "percentage": 26.8,
      "tokens": 890123
    },
    "huggingface": {
      "cost": 123.45,
      "percentage": 9.9,
      "tokens": 456789
    },
    "p2p_network": {
      "cost": 222.43,
      "percentage": 17.8,
      "tokens": 266013
    }
  },
  "daily_breakdown": [
    {
      "date": "2024-12-01",
      "cost": 45.67,
      "requests": 234,
      "tokens": 89234
    },
    {
      "date": "2024-12-02", 
      "cost": 67.89,
      "requests": 345,
      "tokens": 123456
    }
  ]
}
```

### Set Budget Alerts

**Request:**
```bash
curl -X POST https://api.prsm.ai/v1/cost/alerts \
  -H "Authorization: Bearer $PRSM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "monthly_budget_alert",
    "type": "budget_threshold",
    "conditions": {
      "budget_amount": 2000.00,
      "currency": "USD",
      "period": "monthly",
      "thresholds": [
        {
          "percentage": 75,
          "alert_type": "warning",
          "notification_channels": ["email", "webhook"]
        },
        {
          "percentage": 90,
          "alert_type": "critical",
          "notification_channels": ["email", "webhook", "sms"]
        },
        {
          "percentage": 100,
          "alert_type": "budget_exceeded",
          "notification_channels": ["email", "webhook", "sms"],
          "actions": ["pause_non_critical_workflows"]
        }
      ]
    },
    "notification_settings": {
      "email": ["admin@yourcompany.com", "finance@yourcompany.com"],
      "webhook": "https://your-webhook.com/cost-alerts",
      "sms": ["+1-555-0123"]
    }
  }'
```

**Response:**
```json
{
  "alert_id": "alert_budget_monthly_001",
  "name": "monthly_budget_alert",
  "type": "budget_threshold",
  "status": "active",
  "conditions": {
    "budget_amount": 2000.00,
    "currency": "USD",
    "period": "monthly",
    "current_spend": 1247.89,
    "percentage_used": 62.4,
    "thresholds": [
      {
        "percentage": 75,
        "alert_type": "warning",
        "triggered": false,
        "notification_channels": ["email", "webhook"]
      },
      {
        "percentage": 90, 
        "alert_type": "critical",
        "triggered": false,
        "notification_channels": ["email", "webhook", "sms"]
      },
      {
        "percentage": 100,
        "alert_type": "budget_exceeded",
        "triggered": false,
        "notification_channels": ["email", "webhook", "sms"],
        "actions": ["pause_non_critical_workflows"]
      }
    ]
  },
  "created_at": "2024-12-21T18:45:00Z",
  "next_evaluation": "2024-12-22T00:00:00Z"
}
```

## 📊 Real-time Monitoring Examples

### Stream Agent Execution

**WebSocket Connection:**
```javascript
// JavaScript WebSocket example
const ws = new WebSocket('wss://ws.prsm.ai/agents/agent_bio_research_001/executions/exec_cart_analysis_001/stream');

ws.onopen = function() {
    console.log('Connected to execution stream');
};

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    console.log('Execution update:', update);
    
    switch(update.type) {
        case 'progress':
            console.log(`Progress: ${update.progress}% - ${update.status}`);
            break;
        case 'partial_result':
            console.log('Partial output:', update.content);
            break;
        case 'completed':
            console.log('Execution completed!');
            console.log('Final result:', update.result);
            break;
        case 'error':
            console.error('Execution error:', update.error);
            break;
    }
};
```

**cURL to establish WebSocket (conceptual):**
```bash
# Note: WebSocket connections typically require a WebSocket client
# This shows the equivalent HTTP upgrade request headers

curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: your-websocket-key" \
  -H "Authorization: Bearer $PRSM_API_KEY" \
  "wss://ws.prsm.ai/agents/agent_bio_research_001/executions/exec_cart_analysis_001/stream"
```

### Subscribe to System Events

**Request:**
```bash
curl -X POST https://api.prsm.ai/v1/events/subscriptions \
  -H "Authorization: Bearer $PRSM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "research_team_events",
    "event_types": [
      "agent.execution.completed",
      "agent.execution.failed", 
      "workflow.step.completed",
      "cost.budget.threshold_reached",
      "p2p.node.status_changed"
    ],
    "filters": {
      "agent_types": ["researcher", "analyst"],
      "cost_threshold": 50.00,
      "priority": ["high", "critical"]
    },
    "delivery": {
      "method": "webhook",
      "endpoint": "https://your-app.com/prsm-events",
      "authentication": {
        "type": "bearer_token",
        "token": "your-webhook-auth-token"
      },
      "retry_policy": {
        "max_retries": 3,
        "backoff_strategy": "exponential"
      }
    }
  }'
```

**Response:**
```json
{
  "subscription_id": "sub_research_events_001",
  "name": "research_team_events",
  "status": "active",
  "event_types": [
    "agent.execution.completed",
    "agent.execution.failed",
    "workflow.step.completed", 
    "cost.budget.threshold_reached",
    "p2p.node.status_changed"
  ],
  "filters": {
    "agent_types": ["researcher", "analyst"],
    "cost_threshold": 50.00,
    "priority": ["high", "critical"]
  },
  "delivery": {
    "method": "webhook",
    "endpoint": "https://your-app.com/prsm-events",
    "authentication": {
      "type": "bearer_token"
    },
    "retry_policy": {
      "max_retries": 3,
      "backoff_strategy": "exponential"
    }
  },
  "created_at": "2024-12-21T19:00:00Z",
  "events_delivered": 0,
  "last_delivery": null
}
```

## 🔧 Advanced Integration Examples

### Batch Agent Creation

**Request:**
```bash
curl -X POST https://api.prsm.ai/v1/agents/batch \
  -H "Authorization: Bearer $PRSM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agents": [
      {
        "name": "literature_reviewer",
        "type": "researcher",
        "model_provider": "openai",
        "model_name": "gpt-4",
        "capabilities": ["literature_search", "citation_analysis"]
      },
      {
        "name": "data_processor",
        "type": "analyst", 
        "model_provider": "anthropic",
        "model_name": "claude-3-sonnet",
        "capabilities": ["data_cleaning", "statistical_analysis"]
      },
      {
        "name": "report_writer",
        "type": "specialist",
        "model_provider": "openai",
        "model_name": "gpt-4",
        "capabilities": ["technical_writing", "visualization"]
      }
    ],
    "batch_options": {
      "create_team": true,
      "team_name": "research_analysis_team",
      "shared_memory": true,
      "auto_assign_roles": true
    }
  }'
```

**Response:**
```json
{
  "batch_id": "batch_create_001",
  "status": "completed",
  "created_agents": [
    {
      "id": "agent_lit_review_001",
      "name": "literature_reviewer",
      "status": "active"
    },
    {
      "id": "agent_data_proc_002", 
      "name": "data_processor",
      "status": "active"
    },
    {
      "id": "agent_report_write_003",
      "name": "report_writer", 
      "status": "active"
    }
  ],
  "team_created": {
    "team_id": "team_research_001",
    "name": "research_analysis_team",
    "member_count": 3
  },
  "summary": {
    "total_requested": 3,
    "successfully_created": 3,
    "failed": 0,
    "total_cost": 0.00
  },
  "created_at": "2024-12-21T19:15:00Z"
}
```

### Health Check

**Request:**
```bash
curl -X GET https://api.prsm.ai/v1/health \
  -H "Authorization: Bearer $PRSM_API_KEY"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-21T19:20:00Z",
  "version": "1.2.0",
  "uptime": 2847392,
  "services": {
    "api_gateway": {
      "status": "healthy",
      "response_time": 23,
      "last_check": "2024-12-21T19:19:55Z"
    },
    "agent_service": {
      "status": "healthy",
      "active_agents": 15_432,
      "avg_response_time": 1.2,
      "last_check": "2024-12-21T19:19:58Z"
    },
    "p2p_network": {
      "status": "healthy",
      "connected_nodes": 5_439,
      "network_latency": 67,
      "last_check": "2024-12-21T19:19:59Z"
    },
    "workflow_engine": {
      "status": "healthy",
      "active_workflows": 234,
      "queued_tasks": 23,
      "last_check": "2024-12-21T19:19:57Z"
    },
    "database": {
      "status": "healthy",
      "connection_pool": "95% available",
      "query_time": 12,
      "last_check": "2024-12-21T19:19:56Z"
    }
  },
  "metrics": {
    "requests_per_second": 342.7,
    "success_rate": 99.2,
    "p95_response_time": 456,
    "error_rate": 0.8
  }
}
```

## 🚨 Error Handling Examples

### Common Error Responses

**400 Bad Request:**
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Invalid agent configuration provided",
    "details": {
      "validation_errors": [
        {
          "field": "model_name",
          "message": "Model 'gpt-5' is not available",
          "suggested_values": ["gpt-4", "gpt-3.5-turbo"]
        },
        {
          "field": "capabilities",
          "message": "At least one capability is required"
        }
      ]
    },
    "request_id": "req_abc123def456",
    "timestamp": "2024-12-21T19:25:00Z"
  }
}
```

**401 Unauthorized:**
```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid API key provided",
    "details": {
      "api_key_status": "invalid",
      "suggestions": [
        "Verify your API key is correct",
        "Check if your API key has expired",
        "Ensure you're using the correct authentication header"
      ]
    },
    "request_id": "req_def456ghi789",
    "timestamp": "2024-12-21T19:25:00Z"
  }
}
```

**429 Rate Limit Exceeded:**
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Too many requests. Rate limit of 1000 requests/hour exceeded",
    "details": {
      "limit": 1000,
      "current_usage": 1000,
      "reset_time": "2024-12-21T20:00:00Z",
      "retry_after": 2100
    },
    "request_id": "req_ghi789jkl012",
    "timestamp": "2024-12-21T19:25:00Z"
  }
}
```

**500 Internal Server Error:**
```json
{
  "error": {
    "code": "INTERNAL_SERVER_ERROR", 
    "message": "An unexpected error occurred while processing your request",
    "details": {
      "error_id": "err_internal_001",
      "support_contact": "support@prsm.network",
      "retry_safe": true
    },
    "request_id": "req_jkl012mno345",
    "timestamp": "2024-12-21T19:25:00Z"
  }
}
```

## 📚 Additional Resources

### Rate Limits
- **Free Tier**: 1,000 requests/hour
- **Pro Tier**: 10,000 requests/hour  
- **Enterprise**: 100,000 requests/hour
- **P2P Network**: Unlimited (subject to token balance)

### Response Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `429` - Rate Limited
- `500` - Internal Server Error
- `503` - Service Unavailable

### Base URLs
- **Production**: `https://api.prsm.ai`
- **Staging**: `https://staging-api.prsm.ai`
- **WebSocket**: `wss://ws.prsm.ai`

### Authentication Headers
```bash
# API Key
Authorization: Bearer your_api_key

# FTNS Token
X-FTNS-Token: your_ftns_token

# JWT Token (for user sessions)
Authorization: Bearer your_jwt_token
```

---

**Need more examples?** Check out our [interactive API explorer](https://docs.prsm.network/api-explorer) or join our [developer community](https://discord.gg/prsm) for live examples and support.