from enum import Enum
from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field


class TestCredentials(BaseModel):
    """
    Credentials to use for authentication during testing.
    Flexible key-value structure to support various auth schemes.
    """

    # Common OAuth2/password flow fields
    username: str | None = Field(
        None, description="Username or email for authentication"
    )
    password: str | None = Field(None, description="Password for authentication")

    # Additional fields for flexibility
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional credential fields (api_key, client_id, client_secret, etc.)",
    )

    def to_prompt_context(self) -> str:
        """Convert credentials to a context string for the LLM prompt."""
        parts = []
        if self.username:
            parts.append(f"- Username/Email: {self.username}")
        if self.password:
            parts.append(f"- Password: {self.password}")
        for key, value in self.extra.items():
            parts.append(f"- {key}: {value}")
        return "\n".join(parts) if parts else "No credentials provided"


class BodyFormat(str, Enum):
    """
    Enum representing the format of request body.
    Extensible for future formats like XML, GraphQL, etc.
    """

    NONE = "none"  # No request body (GET, DELETE without body)
    JSON = "json"  # application/json
    FORM_URLENCODED = (
        "form_urlencoded"  # application/x-www-form-urlencoded (OAuth2, login forms)
    )
    MULTIPART = "multipart"  # multipart/form-data (file uploads)
    RAW = "raw"  # Raw text/plain or other


class AuthRequirement(str, Enum):
    """
    Enum representing authentication requirements for an endpoint.
    """

    NONE = "none"  # No authentication required
    OPTIONAL = "optional"  # Authentication optional but may affect response
    REQUIRED = "required"  # Authentication required
    AUTH_PROVIDER = "auth_provider"  # This endpoint provides authentication tokens


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
    body_format: BodyFormat = Field(
        default=BodyFormat.JSON,
        description="Format of the request body. Use FORM_URLENCODED for OAuth2/login endpoints, MULTIPART for file uploads, JSON for standard API calls, NONE for GET/DELETE without body",
    )
    auth_requirement: AuthRequirement = Field(
        default=AuthRequirement.NONE,
        description="Authentication requirement for this endpoint. AUTH_PROVIDER for login/token endpoints, REQUIRED for protected routes, NONE for public routes",
    )
    auth_token_path: Optional[str] = Field(
        None,
        description="If this is an AUTH_PROVIDER step, the JSON path to extract the token from response (e.g., 'access_token' or 'data.token')",
    )


class TestPlan(BaseModel):
    summary: str = Field(description="High-level summary of the testing strategy")
    base_url: str = Field(description="The base URL of the API")
    steps: List[TestStep] = Field(description="List of test steps to execute")
