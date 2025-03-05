import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import pytest

# Define your schema
customer_schema = DataFrameSchema(
    {
        "customer_id": Column(
            pa.String, 
            checks=[Check(lambda x: x.str.startswith("C"), "ID must start with C")],
            nullable=False
        ),
        "age": Column(
            pa.Int,
            checks=[Check(lambda x: x >= 18, "Customers must be 18 or older")],
            nullable=False
        )
    }
)

# Function to validate
def process_customer_data(df):
    """Process customer data, assuming it meets the schema requirements."""
    return customer_schema(df)

# Test cases
def test_valid_data_passes():
    """Test that valid data passes validation."""
    valid_df = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003"],
        "age": [25, 30, 42]
    })
    
    # This should not raise an exception
    result = process_customer_data(valid_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3

def test_invalid_customer_id_format():
    """Test that invalid customer ID format raises SchemaError."""
    invalid_df = pd.DataFrame({
        "customer_id": ["D001", "C002", "C003"],  # D001 is invalid
        "age": [25, 30, 42]
    })
    
    # Using pytest's context manager to check for exception
    with pytest.raises(pa.errors.SchemaError) as excinfo:
        process_customer_data(invalid_df)
    
    # Check that the error message mentions the specific issue
    assert "ID must start with C" in str(excinfo.value)

def test_underage_customers():
    """Test that underage customers raise SchemaError."""
    invalid_df = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003"],
        "age": [25, 17, 42]  # 17 is underage
    })
    
    with pytest.raises(pa.errors.SchemaError) as excinfo:
        process_customer_data(invalid_df)
    
    assert "Customers must be 18 or older" in str(excinfo.value)
