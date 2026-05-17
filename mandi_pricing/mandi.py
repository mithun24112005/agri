import subprocess
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATA_GOV_API_KEY", "579b464db66ec23bdd0000013084e2cd1403424b53a4afb894401002")
RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"

def get_field(record, *keys):
    for key in keys:
        if key in record:
            return record[key]
    return ""

def get_mandi_prices(commodity, state=None, district=None, market=None, limit=10):

    base_url = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
    query = f"?api-key={API_KEY}&format=json&limit={limit}"

    if commodity: query += f"&filters%5Bcommodity%5D={commodity.capitalize()}"
    if state:     query += f"&filters%5Bstate.keyword%5D={state.title()}"
    if district:  query += f"&filters%5Bdistrict%5D={district.title()}"
    if market:    query += f"&filters%5Bmarket%5D={market.title()}"

    full_url = base_url + query

    curl_cmd = [
        "curl", "-s",
        "--ssl-no-revoke",
        "--globoff",
        "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "-H", "Accept: application/json",
        "-H", "Referer: https://data.gov.in/",
        full_url
    ]

    result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=20)

    if not result.stdout.strip():
        print("Empty response.", result.stderr)
        return

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("Parse error:", result.stdout[:300])
        return

    records = data.get("records", [])

    if not records:
        print(f"No records found for '{commodity}'. Total: {data.get('total', 0)}")
        return

    print(f"\nFound {len(records)} records for '{commodity.capitalize()}'")
    print("-" * 80)
    print(f"{'Market':<20} {'District':<16} {'State':<14} {'Date':<12} {'Min':>7} {'Max':>7} {'Modal':>7}")
    print("-" * 80)

    for r in records:
        market_val   = get_field(r, "market",       "Market")
        district_val = get_field(r, "district",     "District")
        state_val    = get_field(r, "state",        "State")
        date_val     = get_field(r, "arrival_date", "Arrival_Date")
        min_val      = get_field(r, "min_price",    "Min_Price",   "Min_x0020_Price")
        max_val      = get_field(r, "max_price",    "Max_Price",   "Max_x0020_Price")
        modal_val    = get_field(r, "modal_price",  "Modal_Price", "Modal_x0020_Price")

        print(f"{market_val:<20} {district_val:<16} {state_val:<14} {date_val:<12} "
              f"{str(min_val):>7} {str(max_val):>7} {str(modal_val):>7}")

    print("-" * 80)
    print("Prices in Rs/quintal")


if __name__ == "__main__":
    print("=== Mandi Price Checker ===")
    commodity = input("Enter commodity (e.g. Tomato, Wheat, Rice): ").strip()
    state     = input("Enter state (or press Enter to skip): ").strip() or None
    district  = input("Enter district (or press Enter to skip): ").strip() or None
    market    = input("Enter market/city (or press Enter to skip): ").strip() or None

    get_mandi_prices(commodity, state, district, market)

## Sample Run
'''
=== Mandi Price Checker ===
Enter commodity (e.g. Tomato, Wheat, Rice): Tomato
Enter state (or press Enter to skip): Karnataka
Enter district (or press Enter to skip): 
Enter market/city (or press Enter to skip): 

Found 10 records for 'Tomato'
-----------------------------------------------------------------
Market               District        State           Min    Max  Modal
-----------------------------------------------------------------
Yeshwanthpur         Bengaluru       Karnataka       800   2800   2000
Hubli                Dharwad         Karnataka       600   1800   1200
Mysuru               Mysuru          Karnataka       700   2000   1400
-----------------------------------------------------------------
Prices in ₹ per quintal
'''