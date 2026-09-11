
from typing import TypedDict


class InvestmentState(TypedDict):
    """
    A typed dictionary representing the state of an investment.
    """
    # Input
    companies: list                    # List of companies to analyze

    # Research outputs (one per company)
    research_tesla: str                # Tesla research findings
    research_nvidia: str               # NVIDIA research findings

    # Analysis outputs
    financial_analysis: str            # Comparative financial analysis
    risk_assessment: str               # Risk assessment for all companies

    # Final output
    investment_memo: str               # The final investment memo

    # Metadata
    current_agent: str                 # Which agent is currently working

