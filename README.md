 
# SectorScope

**Stock screening tool powered by Python that finds the most liquid, technically strong stocks across all market industries**

## Work in Progress

Still being developed and is not a finished product. The main working script is **`industry_screener2.py`** - this is the latest and most complete version.

## Quick Start

### Setup


### Run the Scanner
```bash
python industry_screener2.py
```

## What It Does

- Screens **144 industries** automatically
- Finds stocks with **who are small market cap and/or above**, **have strong technical setups** (above SMA50/200), have historically made **new highs/gains on a weekly, monthly, quarterly bases** and are **liquid leaders**.
- Calculates **Average Dollar Volume** to identify liquid leaders
- Exports the results into a .csv file
- Uses Finviz data
- Takes 45minutes to an hour to run the complete scan

## Repository Structure

- **`industry_screener2.py`** ← **Main working script (use this one)**
- `requirements.txt` - Required Python packages
- Other `.py` files - Various development experiments (incomplete)

## Output

csv file with 20+ data points per stock, organized by industry with summary statistics showing the most tradeable names in each sector.

---

**Note:** This tool uses the Finviz API and requires internet connection. Do not change the rate limits or the connection will be shut off.

## Example 
![Screenshot_227](https://github.com/user-attachments/assets/3d86d20e-89cf-428c-98d6-b2223b54e000)
