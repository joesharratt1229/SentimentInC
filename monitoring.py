from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator, Field
from typing import List

app = FastAPI()

# Pydantic model for request validation
class SalesItem(BaseModel):
    product_id: str = Field(..., description="Product identifier")
    quantity: int = Field(..., gt=0, description="Quantity ordered")
    price: float = Field(..., gt=0, description="Price per unit")
    customer_id: str = Field(..., description="Customer identifier")
    
    # Custom validator
    @validator('product_id')
    def validate_product_id(cls, value):
        if not value.startswith('P'):
            raise ValueError('Product ID must start with P')
        return value
    
    # Custom validator for multiple fields
    @validator('quantity', 'price')
    def validate_total_value(cls, value, values):
        if 'quantity' in values and 'price' in values:
            if values['quantity'] * values['price'] > 10000:
                raise ValueError('Total order value exceeds maximum limit')
        return value

# API endpoint with validation
@app.post("/sales/")
async def process_sales(items: List[SalesItem]):
    # Process validated sales data
    processed_items = []
    for item in items:
        processed_item = item.dict()
        processed_item["total"] = item.quantity * item.price
        processed_item["tax"] = processed_item["total"] * 0.08
        processed_items.append(processed_item)
    
    return {"status": "success", "processed_items": processed_items}
