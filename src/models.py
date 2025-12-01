from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class TestStep(BaseModel):
    id: str = Field(description="Unique identifier for this test step")
    description: str = Field(description="Description of what this step tests")
    endpoint: str = Field(description="The endpoint path being tested, e.g., /users")
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = Field(
        description="HTTP method"
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="List of step IDs that must complete before this one",
    )
    payload_description: Optional[str] = Field(
        None, description="Description of the payload structure if applicable"
    )
    expected_status: int = Field(description="Expected HTTP status code")


class TestPlan(BaseModel):
    summary: str = Field(description="High-level summary of the testing strategy")
    base_url: str = Field(description="The base URL of the API")
    steps: List[TestStep] = Field(description="List of test steps to execute")
