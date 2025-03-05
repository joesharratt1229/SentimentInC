data = {
    "customer_id": ["C001", "C002", "C003", "C004", "C005"],
    "name": ["John Smith", "Jane Doe", "Bob Johnson", "Alice Brown", None],
    "age": [32, 45, -5, 28, 39],
    "email": ["john@example.com", "invalid-email", "bob@example.com", "alice@example.com", "emma@example.com"],
    "signup_date": ["2022-01-15", "2022-02-20", "2022-03-10", "2022-04-05", "2022-05-12"],
    "purchase_count": [5, 12, 3, 0, 8]
}

df = pd.DataFrame(data)

data = {
    "customer_id": ["C001", "C002", "C003", "C004", "C005"],
    "name": ["John Smith", "Jane Doe", "Bob Johnson", "Alice Brown", None],
    "age": [32, 45, -5, 28, 39],
    "email": ["john@example.com", "invalid-email", "bob@example.com", "alice@example.com", "emma@example.com"]
}

df = pd.DataFrame(data)

import pandera as pa
from pandera import Column, DataFrameSchema, Check

# Define schema
customer_schema = DataFrameSchema(
    {
        "customer_id": Column(
            pa.String, 
            checks=[Check(lambda x: x.str.startswith("C"), "ID must start with C")],
            nullable=False
        ),
        "name": Column(
            pa.String,
            nullable=True
        ),
        "age": Column(
            pa.Int,
            checks=[
                Check(lambda x: x >= 0, "Age must be non-negative"),
                Check(lambda x: x < 120, "Age must be realistic")
            ],
            nullable=False
        ),
        "email": Column(
            pa.String,
            checks=[Check(lambda x: x.str.contains("@"), "Invalid email format")],
            nullable=False
        )
    }

from datetime import datetime

# Define DataFrame model class
class CustomerModel(pa.DataFrameModel):
    customer_id: Series[str] = pa.Field(
        checks=[pa.Check(lambda x: x.str.startswith("C"), "ID must start with C")],
        nullable=False
    )
    name: Series[str] = pa.Field(nullable=True)
    age: Series[int] = pa.Field(
        checks=[
            pa.Check(lambda x: x >= 0, "Age must be non-negative"),
            pa.Check(lambda x: x < 120, "Age must be realistic")
        ],
        nullable=False
    )
    email: Series[str] = pa.Field(
        checks=[pa.Check(lambda x: x.str.contains("@"), "Invalid email format")],
        nullable=False
    )

