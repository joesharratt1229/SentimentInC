import pandera as pa
from pandera import Column, DataFrameSchema, Check

# Define a single schema that all transaction data must conform to
transaction_schema = DataFrameSchema(
    {
        "customer_id": Column(
            pa.String, 
            checks=[Check(lambda x: x.str.startswith("C"), "ID must start with C")],
            nullable=False
        ),
        "amount": Column(
            pa.Float,
            checks=[
                Check(lambda x: x >= 0, "Amount must be non-negative"),
            ],
            coerce=True,  # Try to convert to float if possible
            nullable=False
        ),
        "transaction_date": Column(
            pa.DateTime,  # Will standardize different date formats
            coerce=True,
            nullable=False
        )
    }
)

# Define a processing pipeline using the schema
@pa.check_types
def process_transactions(df: pa.typing.DataFrame[transaction_schema]) -> pd.DataFrame:
    # At this point, we know the data is valid and properly formatted
    # We can focus on actual business logic
    return df.assign(
        month=df["transaction_date"].dt.month,
        year=df["transaction_date"].dt.year
    )

# Pre-process the API data to fix the customer_id format
transactions_api["customer_id"] = "C" + transactions_api["customer_id"].astype(str).str.zfill(3)

try:
    # Validate and process each dataset
    processed_db = process_transactions(transactions_db)
    processed_api = process_transactions(transactions_api)
    
    # Safe to combine now
    all_transactions = pd.concat([processed_db, processed_api])
    print("Successfully processed all transactions")
    
except pa.errors.SchemaError as e:
    print(f"Validation failed: {e}")
With Pandera:
