"""Synthetic sample-corpus generator.

Used only when the auto-discovery step finds no existing corpus in the project.
It writes a small but realistic multi-domain corpus (policy / finance / SOP)
across several file formats so every downstream stage has meaningful input.
The documents deliberately contain answer-bearing sentences (approvals,
thresholds, owners, effective dates, SLAs, metrics, escalation steps).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Each entry: (relative_path, text_content). CSV/JSON handled separately.
# ---------------------------------------------------------------------------

_POLICY_PROCUREMENT = """# Procurement and Purchase Approval Policy 2025

## Scope and Applicability
This policy applies to all employees, contractors, and vendors engaged by the
company across all regions. It governs the approval of purchases, procurement
requests, and vendor onboarding. This policy is effective July 1, 2025.

## Approval Thresholds
Purchases up to $2,000 may be approved by the immediate Line Manager.
Purchases over $10,000 in India require Finance Director approval.
Purchases over $25,000 in the United States require Regional VP approval.
Purchases over $50,000 in any region require CFO approval.
All software purchases regardless of amount require IT Security review.

## Exceptions
Emergency purchases below $5,000 may be made without prior approval, except
that they must be reported to Finance within 48 hours. Purchases from
restricted vendors are prohibited unless an exception is granted by the
Chief Procurement Officer.

## Ownership
The Procurement Operations team owns this policy. Questions about approval
routing are handled by the Procurement Operations Manager.
"""

_POLICY_TRAVEL = """# Travel and Expense Policy

## Applicability
This policy applies to all employees who travel for business purposes.
It is effective March 15, 2025 and supersedes all prior travel guidelines.

## Booking Rules
Flights over 6 hours in duration may be booked in premium economy for
eligible senior staff. Hotel bookings must not exceed $300 per night in
the United States and must not exceed $180 per night in India.

## Reimbursement Thresholds
Expense claims under $75 do not require receipts. Expense claims over $75
require an itemized receipt. Claims over $2,500 require Finance Manager
approval before reimbursement.

## Exceptions
Client entertainment expenses over $500 require Regional VP approval, except
for pre-approved conference budgets. The Finance Operations team owns this
policy and is responsible for reimbursement processing.
"""

_POLICY_DATA = """# Data Classification and Access Policy

## Scope
This policy applies to all systems, employees, and third parties that access
company data. It became effective January 1, 2026.

## Classification Rules
Data classified as Restricted must be encrypted at rest and in transit.
Access to Restricted data requires Data Owner approval and a documented
business justification. Confidential data may be shared internally but must
not be sent to external parties without Legal approval.

## Access Approval
Requests to access Restricted datasets over 10,000 records require the
Chief Data Officer approval. Standard access requests are approved by the
Data Owner. The Information Security team owns this policy.

## Retention and Exceptions
Personal data must be retained for no longer than 24 months, except where a
legal hold applies. Exceptions to the retention rule require Legal and the
Data Protection Officer approval.
"""

_FINANCE_Q2 = """# Q2 2026 Financial Performance Summary

## Revenue Overview
Total revenue for Q2 2026 was $48.2 million, up 12 percent versus Q2 2025.
Subscription revenue was $34.5 million and services revenue was $13.7 million.
The revenue figure is sourced from the consolidated GL revenue table.

## Variance Analysis
Services revenue was 8 percent below forecast. The variance was driven by
delayed customer renewals in the India region. Subscription revenue exceeded
forecast by 4 percent due to strong enterprise expansion in the US.

## Margin and Cost
Gross margin for Q2 2026 was 71 percent, up 2 points versus the prior quarter.
Operating expenses were $22.1 million, in line with the approved budget.

## Forecast
The full-year 2026 revenue forecast is $205 million. Any forecast revision
over 5 percent requires CFO approval before it is communicated to the board.
"""

_FINANCE_BUDGET = """# 2026 Departmental Budget Guidelines

## Budget Ownership
Each department head owns their departmental budget. The FP&A team owns the
consolidated budget model. These guidelines are effective February 1, 2026.

## Spending Authority
Budget line reallocations under $20,000 may be approved by the Department Head.
Reallocations over $20,000 require FP&A Director approval. Reallocations over
$100,000 require CFO approval.

## Variance Reporting
Any budget variance over 10 percent must be explained in the monthly variance
report. The variance driver and corrective action must be documented.

## Source Tables
All budget figures are traced to the Workday budget source table. The metric
owner for headcount cost is the FP&A team.
"""

_SOP_INCIDENT = """# SOP: Security Incident Response

## Purpose and Owner
This standard operating procedure defines the steps for responding to a
security incident. The Security Operations Center owns this SOP. It is
effective April 1, 2026.

## Steps
Step 1: Detect and log the incident. The on-call analyst is responsible for
initial triage within 15 minutes of the alert.
Step 2: Classify the incident severity. The Incident Commander is responsible
for severity classification.
Step 3: Contain the incident. The Security Engineer is responsible for
containment actions.
Step 4: Notify stakeholders. Critical incidents must be escalated to the CISO
within 30 minutes.

## SLA and Escalation
Priority 1 incidents must be acknowledged within 15 minutes and resolved
within 4 hours. If an incident is not resolved within the SLA, it must be
escalated to the Incident Commander and then to the CISO.

## Inputs and Outputs
The input to this SOP is a security alert from the monitoring system. The
output is a resolved incident record and a post-incident review document.
"""

_SOP_ONBOARDING = """# SOP: Employee Onboarding

## Owner and Applicability
The People Operations team owns this SOP. It applies to all new full-time and
contract hires. It is effective June 1, 2025.

## Steps
Step 1: Create the employee record in the HRIS. The HR Coordinator is
responsible for this step and must complete it within 2 business days.
Step 2: Provision IT accounts and hardware. The IT Service Desk is responsible
and must complete provisioning before the start date.
Step 3: Assign onboarding training. The Learning team owns training assignment.
Step 4: Conduct a 30-day check-in. The hiring manager is responsible.

## SLA and Escalation
IT provisioning has an SLA of 3 business days. If provisioning is not complete
by the start date, the issue is escalated to the IT Service Desk Manager.

## Inputs and Outputs
The input is a signed offer letter. The output is a fully provisioned and
trained employee.
"""

_SOP_VENDOR = """# SOP: Vendor Onboarding and Risk Review

## Owner
The Vendor Management Office owns this SOP. It is effective September 1, 2025
and applies to all new vendor engagements.

## Steps
Step 1: Collect vendor documentation. The Procurement Analyst is responsible.
Step 2: Perform a risk assessment. The Risk team is responsible for scoring
vendor risk as low, medium, or high.
Step 3: Approve the vendor. Low risk vendors are approved by the Procurement
Manager. High risk vendors require Chief Risk Officer approval.
Step 4: Set up the vendor in the ERP system.

## SLA and Escalation
Vendor risk reviews must be completed within 5 business days. Overdue reviews
are escalated to the Vendor Management Office lead.
"""

_README = """# Company Knowledge Corpus (Sample)

This folder contains a small sample corpus used to exercise the Question-Space
Driven Knowledge Compilation pipeline. It mixes policy documents, finance
reports, and standard operating procedures across several file formats.

These documents are synthetic and were generated for demonstration purposes.
"""

_TEXT_DOCS: List[Tuple[str, str]] = [
    ("policies/procurement_policy_2025.md", _POLICY_PROCUREMENT),
    ("policies/travel_expense_policy.md", _POLICY_TRAVEL),
    ("policies/data_classification_policy.md", _POLICY_DATA),
    ("finance/q2_2026_financial_summary.md", _FINANCE_Q2),
    ("finance/2026_budget_guidelines.txt", _FINANCE_BUDGET),
    ("sops/sop_security_incident_response.md", _SOP_INCIDENT),
    ("sops/sop_employee_onboarding.md", _SOP_ONBOARDING),
    ("sops/sop_vendor_onboarding.txt", _SOP_VENDOR),
    ("README.md", _README),
]

# Structured approval matrix as CSV.
_APPROVAL_MATRIX_CSV = """region,transaction_type,amount_threshold,approver,effective_date
India,purchase,10000,Finance Director,2025-07-01
United States,purchase,25000,Regional VP,2025-07-01
Global,purchase,50000,CFO,2025-07-01
India,expense,2500,Finance Manager,2025-03-15
United States,entertainment,500,Regional VP,2025-03-15
Global,budget_reallocation,100000,CFO,2026-02-01
Global,data_access,10000,Chief Data Officer,2026-01-01
Global,vendor_high_risk,0,Chief Risk Officer,2025-09-01
"""

# Structured metric catalog as JSON.
_METRIC_CATALOG_JSON = {
    "title": "Finance Metric Catalog",
    "effective_date": "2026-02-01",
    "owner": "FP&A Team",
    "metrics": [
        {"metric": "Total Revenue", "period": "Q2 2026", "value": "$48.2M",
         "source_table": "consolidated_gl_revenue", "owner": "FP&A Team"},
        {"metric": "Subscription Revenue", "period": "Q2 2026", "value": "$34.5M",
         "source_table": "consolidated_gl_revenue", "owner": "FP&A Team"},
        {"metric": "Gross Margin", "period": "Q2 2026", "value": "71%",
         "source_table": "margin_summary", "owner": "FP&A Team"},
        {"metric": "Operating Expenses", "period": "Q2 2026", "value": "$22.1M",
         "source_table": "opex_ledger", "owner": "FP&A Team"},
    ],
}


def generate(target_dir: Path) -> List[Path]:
    """Write the sample corpus and return the list of files created."""
    created: List[Path] = []
    target_dir.mkdir(parents=True, exist_ok=True)

    for rel, content in _TEXT_DOCS:
        p = target_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        created.append(p)

    csv_path = target_dir / "finance" / "approval_matrix.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(_APPROVAL_MATRIX_CSV, encoding="utf-8")
    created.append(csv_path)

    json_path = target_dir / "finance" / "metric_catalog.json"
    json_path.write_text(json.dumps(_METRIC_CATALOG_JSON, indent=2),
                         encoding="utf-8")
    created.append(json_path)

    return created
