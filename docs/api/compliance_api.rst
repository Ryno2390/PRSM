Compliance API
==============

The Compliance API provides comprehensive SOC2/ISO27001 compliance management for the PRSM platform, enabling enterprise-grade security control management, risk assessment, audit tracking, and regulatory compliance reporting.

**Key Features:**

* Automated SOC2 Type II and ISO27001:2013 compliance assessment
* Risk management with continuous monitoring and treatment tracking
* Evidence collection with integrity verification and retention management
* Audit finding tracking and automated remediation workflows
* Comprehensive compliance reporting for auditors and management
* Enterprise-grade security controls and authorization

.. automodule:: prsm.api.compliance_api
   :members:
   :undoc-members:
   :show-inheritance:

API Endpoints
-------------

Compliance Assessment
~~~~~~~~~~~~~~~~~~~~~

.. http:post:: /compliance/assessments

   Conduct comprehensive compliance assessment for specified framework with automated control evaluation.

   **Request Body:**

   .. code-block:: json

      {
        "framework": "soc2_type_ii",
        "assessment_scope": ["AC-1", "AC-2", "AC-3"],
        "include_evidence": true,
        "include_risks": true
      }

   **Supported Frameworks:**
   
   * ``soc2_type_ii`` - SOC2 Type II Trust Service Criteria
   * ``iso27001`` - ISO27001:2013 Information Security Management

   **Response:**

   .. code-block:: json

      {
        "framework": "soc2_type_ii",
        "assessment_date": "2025-07-02T10:30:00Z",
        "overall_compliance_percentage": 85.7,
        "control_summary": {
          "implemented": 42,
          "partially_implemented": 8,
          "not_implemented": 3,
          "not_applicable": 2
        },
        "overdue_controls": [
          {
            "control_id": "AC-3",
            "name": "Access Enforcement",
            "days_overdue": 15,
            "last_tested": "2025-06-15T00:00:00Z"
          }
        ],
        "risk_summary": {
          "critical_risks": 0,
          "high_risks": 2,
          "medium_risks": 8,
          "low_risks": 15
        },
        "evidence_status": {
          "evidence_collected": 127,
          "evidence_required": 145,
          "coverage_percentage": 87.6
        },
        "recommendations": [
          "Complete overdue control testing for AC-3",
          "Collect missing evidence for IA-2 controls",
          "Update risk assessments for high-risk findings"
        ],
        "next_assessment_due": "2025-10-02T10:30:00Z"
      }

Control Management
~~~~~~~~~~~~~~~~~~

.. http:get:: /compliance/controls

   List compliance controls with implementation status, testing schedules, and evidence collection status.

   **Query Parameters:**
   
   * ``framework`` (optional) - Filter by compliance framework
   * ``status_filter`` (optional) - Filter by implementation status
   * ``overdue_only`` (optional, default: false) - Show only overdue controls

   **Response:**

   .. code-block:: json

      [
        {
          "control_id": "AC-1",
          "name": "Access Control Policy and Procedures",
          "description": "Develop, document, and disseminate access control policy",
          "framework": "soc2_type_ii",
          "control_type": "preventive",
          "control_family": "access_control",
          "implementation_status": "implemented",
          "last_tested": "2025-06-01T00:00:00Z",
          "next_test_due": "2025-09-01T00:00:00Z",
          "responsible_party": "Security Team",
          "evidence_count": 5,
          "automation_level": "high"
        }
      ]

.. http:get:: /compliance/controls/{control_id}

   Get detailed information about a specific compliance control including implementation guidance and evidence requirements.

   **Path Parameters:**
   
   * ``control_id`` - Unique identifier for the compliance control

   **Response:**

   .. code-block:: json

      {
        "control_specification": {
          "control_id": "AC-1",
          "name": "Access Control Policy and Procedures",
          "description": "Develop, document, and disseminate access control policy",
          "framework": "soc2_type_ii",
          "control_type": "preventive",
          "control_family": "access_control",
          "implementation_guidance": "Establish formal access control policies..."
        },
        "implementation_status": {
          "status": "implemented",
          "responsible_party": "Security Team",
          "implementation_notes": "Policy implemented and approved by board",
          "exceptions": []
        },
        "testing_information": {
          "testing_procedures": ["Review policy documentation", "Validate implementation"],
          "last_tested": "2025-06-01T00:00:00Z",
          "next_test_due": "2025-09-01T00:00:00Z",
          "testing_frequency": "quarterly"
        },
        "evidence_collection": {
          "evidence_requirements": ["policy_document", "approval_records"],
          "evidence_collected": 5,
          "latest_evidence": "2025-06-01T00:00:00Z",
          "evidence_gap": 0
        },
        "audit_findings": {
          "total_findings": 2,
          "open_findings": 0,
          "resolved_findings": 2
        },
        "compliance_metrics": {
          "control_effectiveness": "85%",
          "automation_level": "high",
          "maturity_level": "optimized"
        }
      }

Evidence Collection
~~~~~~~~~~~~~~~~~~~

.. http:post:: /compliance/evidence

   Collect and store compliance evidence with integrity verification and retention management.

   **Request Body:**

   .. code-block:: json

      {
        "control_id": "AC-1",
        "evidence_type": "policy_document",
        "evidence_data": {
          "document_name": "Access Control Policy v2.1",
          "document_hash": "sha256:abc123...",
          "approval_date": "2025-01-15",
          "approved_by": "CISO"
        },
        "collection_notes": "Annual policy review and update"
      }

   **Evidence Types:**
   
   * ``policy_document`` - Policy and procedure documentation
   * ``configuration_snapshot`` - System configuration evidence
   * ``log_analysis`` - Security log analysis and monitoring
   * ``penetration_test`` - Security testing results
   * ``vulnerability_scan`` - Vulnerability assessment reports
   * ``training_records`` - Security awareness training records

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "message": "Evidence collected successfully",
        "evidence_id": "evidence_abc123",
        "control_id": "AC-1",
        "evidence_type": "policy_document",
        "timestamp": "2025-07-02T10:30:00Z"
      }

.. http:get:: /compliance/evidence/{evidence_id}

   Get detailed information about collected evidence including integrity verification and retention status.

   **Path Parameters:**
   
   * ``evidence_id`` - Unique identifier for the evidence item

   **Response:**

   .. code-block:: json

      {
        "evidence_metadata": {
          "evidence_id": "evidence_abc123",
          "control_id": "AC-1",
          "control_name": "Access Control Policy and Procedures",
          "evidence_type": "policy_document",
          "collection_method": "automated",
          "collected_at": "2025-07-02T10:30:00Z",
          "collected_by": "user_123"
        },
        "integrity_verification": {
          "integrity_hash": "sha256:def456...",
          "hash_algorithm": "SHA-256",
          "verification_status": "verified",
          "last_verified": "2025-07-02T10:30:00Z"
        },
        "retention_management": {
          "retention_period": 2555,
          "retention_expires": "2032-07-02T10:30:00Z",
          "classification": "compliance_evidence",
          "disposition": "retain"
        },
        "evidence_data": {
          "document_name": "Access Control Policy v2.1",
          "document_hash": "sha256:abc123...",
          "approval_date": "2025-01-15",
          "approved_by": "CISO"
        },
        "audit_trail": {
          "creation_time": "2025-07-02T10:30:00Z",
          "access_count": 1,
          "last_accessed": "2025-07-02T10:30:00Z"
        }
      }

Risk Management
~~~~~~~~~~~~~~~

.. http:post:: /compliance/risk-assessments

   Perform comprehensive risk assessment with automated risk calculation and control effectiveness analysis.

   **Request Body:**

   .. code-block:: json

      {
        "risk_name": "Data Breach Risk",
        "description": "Risk of unauthorized access to customer data",
        "category": "information_security",
        "likelihood": 0.3,
        "impact": 0.8,
        "risk_owner": "Data Protection Officer",
        "mitigation_notes": "Implement additional access controls"
      }

   **Risk Categories:**
   
   * ``information_security`` - Information security risks
   * ``operational`` - Operational and process risks
   * ``compliance`` - Regulatory compliance risks
   * ``business_continuity`` - Business continuity risks
   * ``third_party`` - Third-party and vendor risks

   **Response:**

   .. code-block:: json

      {
        "risk_id": "risk_abc123",
        "risk_name": "Data Breach Risk",
        "description": "Risk of unauthorized access to customer data",
        "category": "information_security",
        "likelihood": 0.3,
        "impact": 0.8,
        "inherent_risk": "high",
        "residual_risk": "medium",
        "controls_applied": ["AC-1", "AC-2", "AC-3"],
        "mitigation_strategy": "Implement multi-factor authentication",
        "risk_owner": "Data Protection Officer",
        "assessment_date": "2025-07-02T10:30:00Z",
        "next_review_date": "2025-10-02T10:30:00Z"
      }

.. http:get:: /compliance/risk-assessments

   List risk assessments with filtering capabilities and risk level analysis.

   **Query Parameters:**
   
   * ``category`` (optional) - Filter by risk category
   * ``risk_level`` (optional) - Filter by residual risk level
   * ``risk_owner`` (optional) - Filter by risk owner

   **Response:**

   .. code-block:: json

      [
        {
          "risk_id": "risk_abc123",
          "risk_name": "Data Breach Risk",
          "description": "Risk of unauthorized access to customer data",
          "category": "information_security",
          "likelihood": 0.3,
          "impact": 0.8,
          "inherent_risk": "high",
          "residual_risk": "medium",
          "controls_applied": ["AC-1", "AC-2", "AC-3"],
          "mitigation_strategy": "Implement multi-factor authentication",
          "risk_owner": "Data Protection Officer",
          "assessment_date": "2025-07-02T10:30:00Z",
          "next_review_date": "2025-10-02T10:30:00Z"
        }
      ]

Audit Management
~~~~~~~~~~~~~~~~

.. http:post:: /compliance/audit-findings

   Create and track audit findings with automated remediation workflows and severity-based prioritization.

   **Request Body:**

   .. code-block:: json

      {
        "audit_id": "audit_2025_q2",
        "control_id": "AC-3",
        "finding_type": "control_deficiency",
        "severity": "high",
        "description": "Access enforcement not consistently applied",
        "recommendation": "Implement automated access enforcement mechanisms"
      }

   **Finding Types:**
   
   * ``control_deficiency`` - Control not operating effectively
   * ``design_weakness`` - Control design inadequate
   * ``implementation_gap`` - Control not fully implemented
   * ``evidence_gap`` - Insufficient evidence collected
   * ``policy_violation`` - Non-compliance with policies

   **Severity Levels:**
   
   * ``critical`` - Critical issues requiring immediate attention
   * ``high`` - High-priority issues with significant impact
   * ``medium`` - Medium-priority issues requiring remediation
   * ``low`` - Low-priority issues for tracking
   * ``informational`` - Informational findings for awareness

   **Response:**

   .. code-block:: json

      {
        "finding_id": "finding_abc123",
        "audit_id": "audit_2025_q2",
        "control_id": "AC-3",
        "finding_type": "control_deficiency",
        "severity": "high",
        "description": "Access enforcement not consistently applied",
        "recommendation": "Implement automated access enforcement mechanisms",
        "status": "open",
        "target_date": "2025-08-15T00:00:00Z",
        "created_at": "2025-07-02T10:30:00Z",
        "days_to_target": 44
      }

.. http:get:: /compliance/audit-findings

   List audit findings with filtering and remediation status tracking.

   **Query Parameters:**
   
   * ``audit_id`` (optional) - Filter by audit identifier
   * ``status_filter`` (optional) - Filter by remediation status
   * ``severity`` (optional) - Filter by finding severity
   * ``overdue_only`` (optional, default: false) - Show only overdue findings

   **Response:**

   .. code-block:: json

      [
        {
          "finding_id": "finding_abc123",
          "audit_id": "audit_2025_q2",
          "control_id": "AC-3",
          "finding_type": "control_deficiency",
          "severity": "high",
          "description": "Access enforcement not consistently applied",
          "recommendation": "Implement automated access enforcement mechanisms",
          "status": "open",
          "target_date": "2025-08-15T00:00:00Z",
          "created_at": "2025-07-02T10:30:00Z",
          "days_to_target": 44
        }
      ]

Compliance Reporting
~~~~~~~~~~~~~~~~~~~~

.. http:post:: /compliance/reports

   Generate comprehensive compliance reports for auditors and management with detailed analysis and recommendations.

   **Request Body:**

   .. code-block:: json

      {
        "framework": "soc2_type_ii",
        "report_type": "comprehensive",
        "include_evidence": true,
        "include_recommendations": true,
        "output_format": "json"
      }

   **Report Types:**
   
   * ``comprehensive`` - Complete compliance assessment report
   * ``executive_summary`` - High-level summary for executives
   * ``control_assessment`` - Detailed control implementation report
   * ``risk_assessment`` - Risk analysis and treatment report
   * ``audit_readiness`` - Audit preparation and gap analysis

   **Response:**

   .. code-block:: json

      {
        "report_metadata": {
          "framework": "soc2_type_ii",
          "report_type": "comprehensive",
          "generated_at": "2025-07-02T10:30:00Z",
          "generated_by": "user_123"
        },
        "executive_summary": {
          "overall_compliance_percentage": 85.7,
          "readiness_assessment": "audit_ready",
          "critical_gaps": 2,
          "recommendations_count": 8
        },
        "control_assessment": {
          "total_controls": 55,
          "implemented": 42,
          "partially_implemented": 8,
          "not_implemented": 3,
          "control_effectiveness": "85%"
        },
        "risk_assessment": {
          "total_risks": 25,
          "critical_risks": 0,
          "high_risks": 2,
          "medium_risks": 8,
          "low_risks": 15,
          "risk_treatment_coverage": "92%"
        },
        "evidence_status": {
          "evidence_collected": 127,
          "evidence_required": 145,
          "coverage_percentage": 87.6,
          "retention_compliance": "100%"
        },
        "audit_findings": {
          "total_findings": 12,
          "open_findings": 5,
          "resolved_findings": 7,
          "overdue_findings": 1
        },
        "recommendations": [
          "Complete overdue control testing for AC-3",
          "Collect missing evidence for IA-2 controls",
          "Update risk assessments for high-risk findings"
        ]
      }

System Health
~~~~~~~~~~~~~

.. http:get:: /compliance/health

   Health check for compliance system with framework readiness and system status.

   **Response:**

   .. code-block:: json

      {
        "status": "healthy",
        "timestamp": "2025-07-02T10:30:00Z",
        "frameworks_supported": 2,
        "controls_loaded": 155,
        "evidence_items": 127,
        "risk_assessments": 25,
        "audit_findings": 12,
        "system_readiness": {
          "soc2_controls": 75,
          "iso27001_controls": 80,
          "automation_level": "high",
          "audit_readiness": "production_ready"
        },
        "version": "1.0.0"
      }

Authentication & Authorization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All compliance endpoints require authentication. Evidence access, assessments, and reports require enterprise role or higher.

**Headers:**

.. code-block:: http

   Authorization: Bearer <jwt_token>
   Content-Type: application/json

**Permission Requirements:**

* **Compliance Assessments** - Enterprise role or higher
* **Evidence Collection** - Developer role or higher
* **Evidence Access** - Enterprise role or higher
* **Risk Assessments** - Developer role or higher
* **Audit Findings** - Developer role or higher
* **Compliance Reports** - Enterprise role or higher

Error Responses
--------------

Compliance API endpoints return standardized error responses:

.. code-block:: json

   {
     "error": "invalid_framework",
     "message": "Invalid framework. Must be one of: soc2_type_ii, iso27001",
     "code": 400,
     "timestamp": "2025-07-02T10:30:00Z"
   }

**Common Error Codes:**

* ``400`` - Bad Request (invalid parameters, invalid framework, invalid severity)
* ``401`` - Unauthorized (missing or invalid authentication token)
* ``403`` - Forbidden (insufficient permissions for enterprise features)
* ``404`` - Not Found (control not found, evidence not found)
* ``429`` - Rate Limited (too many assessment requests)
* ``500`` - Internal Server Error (compliance framework error)

**Compliance-Specific Errors:**

* ``invalid_framework`` - Specified compliance framework not supported
* ``control_not_found`` - Specified control ID does not exist
* ``evidence_not_found`` - Specified evidence ID does not exist
* ``invalid_severity`` - Invalid severity level for audit finding
* ``assessment_in_progress`` - Compliance assessment already in progress
* ``insufficient_evidence`` - Insufficient evidence for assessment
