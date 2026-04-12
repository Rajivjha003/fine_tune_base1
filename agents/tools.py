"""
Tools for the MerchFine agent.
These allow the planner agent to fetch real-time facts and perform calculations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.config import get_settings


def create_tools() -> list[Any]:
    """
    Create a list of tools for the LangGraph agent.
    Returns a list of Langchain 'Tool' objects.
    """
    from langchain_core.tools import tool

    settings = get_settings()

    @tool
    def lookup_sku_details(sku_id: str) -> str:
        """Look up basic details and hierarchy for a given SKU ID (returns JSON string)."""
        sku_path = settings.data_dir / "knowledge_base" / "sku_catalog.txt"
        if not sku_path.exists():
            return f"Error: SKU catalog not found."

        try:
            content = sku_path.read_text(encoding="utf-8")
            for line in content.splitlines():
                if f"SKU: {sku_id}" in line or f"{sku_id}" in line:
                    return line
            return f"Error: SKU {sku_id} not found in catalog."
        except Exception as e:
            return f"Error reading catalog: {e}"

    @tool
    def calculate_mio(current_stock: float, avg_monthly_sales: float) -> str:
        """
        Calculate current Months of Inventory Outstanding (MIO).
        Args:
            current_stock: The current units on hand.
            avg_monthly_sales: The average sales per month (units).
        """
        if avg_monthly_sales <= 0:
            return "Error: avg_monthly_sales must be > 0"
        mio = current_stock / avg_monthly_sales
        return f"MIO: {mio:.2f} months"

    @tool
    def get_seasonal_multiplier(season: str, category: str = "Tops") -> str:
        """
        Get the seasonal multiplier for a given season and category.
        Args:
            season: One of Q1, Q2, Q3, Q4, Summer, Winter, etc.
            category: Product category (e.g., Tops, Bottoms, Outerwear).
        """
        seasonal_path = settings.data_dir / "knowledge_base" / "seasonal_calendar.md"
        if not seasonal_path.exists():
            return "Error: Seasonal calendar not found."
            
        try:
            content = seasonal_path.read_text(encoding="utf-8")
            # Extremely naive text extraction for demonstration
            if season.lower() in content.lower():
                return f"Multiplier found in text: ~1.2-1.5 (check manual text for exact match for {category})"
            return f"No specific multiplier found for {season}."
        except Exception as e:
            return f"Error reading seasonal rules: {e}"

    return [lookup_sku_details, calculate_mio, get_seasonal_multiplier]
