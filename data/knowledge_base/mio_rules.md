# MerchMix MIO (Months of Inventory Outstanding) Rules

## Overview

MIO measures how many months of future sales demand the current inventory can cover. It is the primary metric for inventory health and replenishment planning.

**Formula:**
```
MIO = Current On-Hand Inventory / Average Monthly Sales Rate
```

## Target MIO Ranges by Category

| Category | Min MIO | Target MIO | Max MIO | Overstock Threshold |
|----------|---------|------------|---------|---------------------|
| Core Basics (Tees, Socks) | 1.5 | 2.5 | 4.0 | > 5.0 |
| Fashion / Seasonal | 0.5 | 1.5 | 2.5 | > 3.0 |
| Accessories | 1.0 | 2.0 | 3.5 | > 4.5 |
| Footwear | 2.0 | 3.0 | 4.5 | > 5.5 |
| Outerwear / Seasonal Heavy | 1.0 | 2.0 | 3.0 | > 4.0 |

## Reorder Decision Rules

### Rule 1: Standard Reorder Trigger
- **When:** MIO < Min MIO for the category
- **Action:** Generate reorder recommendation
- **Quantity:** Enough to bring MIO back to Target MIO level
- **Formula:** `Reorder Qty = (Target_MIO - Current_MIO) × Avg_Monthly_Sales`

### Rule 2: Seasonal Pre-Build
- **When:** 6-8 weeks before peak season start
- **Action:** Begin building inventory above normal Target MIO
- **Quantity:** Target MIO + 1.0 month buffer for peak period
- **Note:** Apply only to seasonal SKUs (seasonality != "year-round")

### Rule 3: Markdown / Liquidation Trigger
- **When:** MIO > Overstock Threshold AND sell-through rate < 50%
- **Action:** Flag for markdown consideration
- **Recommendation:** 15-30% markdown to accelerate sell-through

### Rule 4: Lead Time Buffer
- **When:** Generating any reorder recommendation
- **Action:** Add lead time coverage to reorder quantity
- **Formula:** `Safety_Buffer = Lead_Time_Days / 30 × Avg_Monthly_Sales × 0.5`
- **Rationale:** Cover half the lead time period as safety stock

### Rule 5: MOQ Alignment
- **When:** Calculated reorder quantity < MOQ
- **Action:** Round UP to nearest MOQ
- **Note:** If rounding up would bring MIO > Max MIO, flag for human review

## Sales Velocity Classification

| Velocity Class | Monthly Turnover | Replenishment Frequency |
|----------------|-----------------|------------------------|
| A (Fast) | > 400 units/month | Weekly or bi-weekly |
| B (Medium) | 150-400 units/month | Bi-weekly or monthly |
| C (Slow) | 50-150 units/month | Monthly or quarterly |
| D (Dead) | < 50 units/month | Flag for discontinuation review |

## Exceptions and Overrides

1. **New Product Launch:** First 90 days use forecasted demand (not historical) for MIO calculation
2. **Promotional Events:** Temporarily increase Target MIO by 50% for 2 weeks pre-promotion
3. **Supply Disruption:** If lead time increases > 50%, double the Safety Buffer
4. **End of Season:** At season end, Target MIO reduces to 0.5 months to minimize carryover
