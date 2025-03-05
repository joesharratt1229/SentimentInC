def process_transactions(df):
    # Check and fix customer_id
    if df["customer_id"].dtype != "object":
        df["customer_id"] = "C" + df["customer_id"].astype(str).str.zfill(3)
    
    # Check and fix amounts
    try:
        df["amount"] = pd.to_numeric(df["amount"])
    except:
        print("Error converting amounts to numeric")
        return None
    
    # Check for negative amounts
    if (df["amount"] < 0).any():
        print("Warning: Negative amounts detected")
        df = df[df["amount"] >= 0]
    
    # Check and standardize dates
    try:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    except:
        print("Error converting dates")
        return None
    
    return df
