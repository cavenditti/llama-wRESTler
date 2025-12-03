"""
Dynamic Pydantic model generation and validation from OpenAPI specifications.

This module provides functionality to:
1. Parse OpenAPI/Swagger specifications
2. Generate Pydantic models programmatically from schema definitions
3. Validate requests and responses against those models
4. Provide per-type generators for deterministic test data
"""

from __future__ import annotations

import hashlib
import random
import string
from datetime import datetime, date, time, timedelta
from enum import Enum
from typing import Any, Callable, Type
from uuid import UUID

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo


# Type alias for OpenAPI schema dict
OpenAPISchema = dict[str, Any]


class ValidationResult(BaseModel):
    """Result of validating a request or response."""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    validated_data: Any = None


class SchemaType(str, Enum):
    """OpenAPI schema types."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class StringFormat(str, Enum):
    """Common OpenAPI string formats."""

    DATE = "date"
    DATE_TIME = "date-time"
    TIME = "time"
    EMAIL = "email"
    URI = "uri"
    URL = "url"
    UUID = "uuid"
    HOSTNAME = "hostname"
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    BYTE = "byte"  # base64
    BINARY = "binary"
    PASSWORD = "password"


class DeterministicGenerator:
    """
    Generates deterministic test data based on a seed.

    Uses a seeded random generator to produce consistent values
    for the same seed, ensuring reproducible test data.
    """

    # Sample data pools
    FIRST_NAMES = [
        "Alice",
        "Bob",
        "Charlie",
        "Diana",
        "Eve",
        "Frank",
        "Grace",
        "Henry",
        "Iris",
        "Jack",
        "Kate",
        "Leo",
        "Mia",
        "Noah",
        "Olivia",
        "Paul",
    ]
    LAST_NAMES = [
        "Smith",
        "Johnson",
        "Williams",
        "Brown",
        "Jones",
        "Garcia",
        "Miller",
        "Davis",
        "Rodriguez",
        "Martinez",
        "Hernandez",
        "Lopez",
        "Wilson",
        "Anderson",
    ]
    DOMAINS = ["example.com", "test.org", "sample.net", "demo.io", "mock.dev"]
    WORDS = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "chi",
        "psi",
        "omega",
    ]
    PET_NAMES = [
        "Buddy",
        "Max",
        "Charlie",
        "Cooper",
        "Rocky",
        "Bear",
        "Duke",
        "Tucker",
        "Bella",
        "Luna",
        "Lucy",
        "Daisy",
        "Molly",
        "Sadie",
        "Bailey",
        "Maggie",
    ]

    def __init__(self, seed: int | str | None = None):
        """
        Initialize the generator with a seed.

        Args:
            seed: Seed for reproducibility. Can be int, string (will be hashed), or None for random.
        """
        if seed is None:
            self._seed = random.randint(0, 2**32 - 1)
        elif isinstance(seed, str):
            self._seed = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)
        else:
            self._seed = seed

        self._rng = random.Random(self._seed)
        self._counter = 0

    @property
    def seed(self) -> int:
        """Get the current seed value."""
        return self._seed

    def _next_counter(self) -> int:
        """Get next counter value for unique IDs."""
        self._counter += 1
        return self._counter

    def reset(self) -> None:
        """Reset the generator to initial state."""
        self._rng = random.Random(self._seed)
        self._counter = 0

    # Primitive type generators
    def generate_string(
        self,
        min_length: int = 1,
        max_length: int = 50,
        pattern: str | None = None,
        enum_values: list[str] | None = None,
        format: str | None = None,
        field_name: str | None = None,
    ) -> str:
        """Generate a string value based on constraints."""
        # Handle enum
        if enum_values:
            return self._rng.choice(enum_values)

        # Handle format
        if format:
            return self._generate_formatted_string(format, field_name)

        # Try to infer from field name
        if field_name:
            lower_name = field_name.lower()
            if "email" in lower_name:
                return self.generate_email()
            if "name" in lower_name:
                if "first" in lower_name:
                    return self._rng.choice(self.FIRST_NAMES)
                if "last" in lower_name:
                    return self._rng.choice(self.LAST_NAMES)
                if "user" in lower_name:
                    return f"user_{self._next_counter()}"
                if "pet" in lower_name:
                    return self._rng.choice(self.PET_NAMES)
                return f"{self._rng.choice(self.FIRST_NAMES)} {self._rng.choice(self.LAST_NAMES)}"
            if "url" in lower_name or "uri" in lower_name:
                return self.generate_url()
            if "phone" in lower_name:
                return self.generate_phone()
            if "password" in lower_name:
                return self.generate_password()

        # Handle pattern (simplified regex support)
        if pattern:
            return self._generate_from_pattern(pattern, min_length, max_length)

        # Default: random string
        length = self._rng.randint(min_length, max_length)
        return "".join(
            self._rng.choices(string.ascii_letters + string.digits, k=length)
        )

    def _generate_formatted_string(
        self, format: str, field_name: str | None = None
    ) -> str:
        """Generate a string with specific format."""
        format_generators: dict[str, Callable[[], str]] = {
            "date": lambda: self.generate_date().isoformat(),
            "date-time": lambda: self.generate_datetime().isoformat(),
            "time": lambda: self.generate_time().isoformat(),
            "email": self.generate_email,
            "uri": self.generate_url,
            "url": self.generate_url,
            "uuid": lambda: str(self.generate_uuid()),
            "hostname": lambda: f"host{self._next_counter()}.example.com",
            "ipv4": self.generate_ipv4,
            "ipv6": self.generate_ipv6,
            "byte": lambda: "SGVsbG9Xb3JsZA==",  # base64 encoded "HelloWorld"
            "binary": lambda: "binary_data",
            "password": self.generate_password,
        }

        generator = format_generators.get(format)
        if generator:
            return generator()

        # Unknown format, generate plain string
        return self.generate_string(min_length=5, max_length=20, field_name=field_name)

    def _generate_from_pattern(self, pattern: str, min_len: int, max_len: int) -> str:
        """Generate a string matching a simplified regex pattern."""
        # Simple pattern handling for common cases
        result = ""
        i = 0
        while i < len(pattern):
            char = pattern[i]
            if char == "[":
                # Character class
                end = pattern.find("]", i)
                if end > i:
                    chars = pattern[i + 1 : end]
                    result += self._rng.choice(chars)
                    i = end + 1
                    continue
            elif char == "\\":
                if i + 1 < len(pattern):
                    next_char = pattern[i + 1]
                    if next_char == "d":
                        result += str(self._rng.randint(0, 9))
                    elif next_char == "w":
                        result += self._rng.choice(
                            string.ascii_letters + string.digits + "_"
                        )
                    else:
                        result += next_char
                    i += 2
                    continue
            elif char in "^$.*+?{}()|":
                i += 1
                continue
            else:
                result += char
            i += 1

        # Ensure minimum length
        while len(result) < min_len:
            result += self._rng.choice(string.ascii_lowercase)

        return result[:max_len]

    def generate_integer(
        self,
        minimum: int | None = None,
        maximum: int | None = None,
        exclusive_minimum: bool = False,
        exclusive_maximum: bool = False,
        multiple_of: int | None = None,
        format: str | None = None,
    ) -> int:
        """Generate an integer value based on constraints."""
        # Set defaults based on format
        if format == "int32":
            min_val = minimum if minimum is not None else -(2**31)
            max_val = maximum if maximum is not None else 2**31 - 1
        elif format == "int64":
            min_val = minimum if minimum is not None else -(2**63)
            max_val = maximum if maximum is not None else 2**63 - 1
        else:
            min_val = minimum if minimum is not None else 0
            max_val = maximum if maximum is not None else 1000

        if exclusive_minimum:
            min_val += 1
        if exclusive_maximum:
            max_val -= 1

        # Clamp to reasonable range
        min_val = max(min_val, -(10**9))
        max_val = min(max_val, 10**9)

        value = self._rng.randint(min_val, max_val)

        if multiple_of:
            value = (value // multiple_of) * multiple_of

        return value

    def generate_number(
        self,
        minimum: float | None = None,
        maximum: float | None = None,
        exclusive_minimum: bool = False,
        exclusive_maximum: bool = False,
        format: str | None = None,
    ) -> float:
        """Generate a floating point number based on constraints."""
        min_val = minimum if minimum is not None else 0.0
        max_val = maximum if maximum is not None else 1000.0

        if exclusive_minimum:
            min_val += 0.0001
        if exclusive_maximum:
            max_val -= 0.0001

        return round(self._rng.uniform(min_val, max_val), 4)

    def generate_boolean(self) -> bool:
        """Generate a boolean value."""
        return self._rng.choice([True, False])

    # Composite type generators
    def generate_array(
        self,
        item_generator: Callable[[], Any],
        min_items: int = 0,
        max_items: int = 5,
        unique_items: bool = False,
    ) -> list[Any]:
        """Generate an array with items from the generator."""
        count = self._rng.randint(min_items, max_items)
        items = []

        for _ in range(count):
            item = item_generator()
            if unique_items and item in items:
                continue
            items.append(item)

        return items

    # Special type generators
    def generate_email(self) -> str:
        """Generate a realistic email address."""
        first = self._rng.choice(self.FIRST_NAMES).lower()
        last = self._rng.choice(self.LAST_NAMES).lower()
        domain = self._rng.choice(self.DOMAINS)
        num = self._rng.randint(1, 999)
        return f"{first}.{last}{num}@{domain}"

    def generate_url(self) -> str:
        """Generate a URL."""
        domain = self._rng.choice(self.DOMAINS)
        path = "/".join(self._rng.choices(self.WORDS, k=self._rng.randint(1, 3)))
        return f"https://{domain}/{path}"

    def generate_phone(self) -> str:
        """Generate a phone number."""
        area = self._rng.randint(200, 999)
        exchange = self._rng.randint(200, 999)
        subscriber = self._rng.randint(1000, 9999)
        return f"+1-{area}-{exchange}-{subscriber}"

    def generate_password(self) -> str:
        """Generate a secure password."""
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        return "".join(self._rng.choices(chars, k=16))

    def generate_uuid(self) -> UUID:
        """Generate a deterministic UUID."""
        hex_chars = "".join(self._rng.choices("0123456789abcdef", k=32))
        return UUID(hex_chars)

    def generate_date(self, start_year: int = 2020, end_year: int = 2025) -> date:
        """Generate a date within range."""
        start = date(start_year, 1, 1)
        end = date(end_year, 12, 31)
        days = (end - start).days
        return start + timedelta(days=self._rng.randint(0, days))

    def generate_datetime(self) -> datetime:
        """Generate a datetime."""
        d = self.generate_date()
        t = self.generate_time()
        return datetime.combine(d, t)

    def generate_time(self) -> time:
        """Generate a time."""
        return time(
            hour=self._rng.randint(0, 23),
            minute=self._rng.randint(0, 59),
            second=self._rng.randint(0, 59),
        )

    def generate_ipv4(self) -> str:
        """Generate an IPv4 address."""
        return ".".join(str(self._rng.randint(1, 254)) for _ in range(4))

    def generate_ipv6(self) -> str:
        """Generate an IPv6 address."""
        groups = [f"{self._rng.randint(0, 65535):04x}" for _ in range(8)]
        return ":".join(groups)


class OpenAPISchemaParser:
    """
    Parses OpenAPI specifications and generates Pydantic models.

    Supports both Swagger 2.0 and OpenAPI 3.x specifications.
    """

    def __init__(self, spec: OpenAPISchema):
        """
        Initialize the parser with an OpenAPI specification.

        Args:
            spec: The parsed OpenAPI/Swagger specification as a dict
        """
        self.spec = spec
        self._model_cache: dict[str, Type[BaseModel]] = {}
        self._is_swagger2 = "swagger" in spec
        self._generator = DeterministicGenerator(seed=42)

        # Determine definitions location based on spec version
        if self._is_swagger2:
            self._definitions = spec.get("definitions", {})
        else:
            components = spec.get("components", {})
            self._definitions = components.get("schemas", {})

    def get_definitions(self) -> dict[str, OpenAPISchema]:
        """Get all schema definitions from the spec."""
        return self._definitions

    def resolve_ref(self, ref: str) -> OpenAPISchema:
        """
        Resolve a $ref pointer to its schema definition.

        Args:
            ref: The $ref string (e.g., "#/definitions/Pet" or "#/components/schemas/Pet")

        Returns:
            The resolved schema definition
        """
        # Parse the ref path
        if ref.startswith("#/"):
            parts = ref[2:].split("/")
            current = self.spec
            for part in parts:
                current = current[part]
            return current
        raise ValueError(f"External refs not supported: {ref}")

    def get_model_name_from_ref(self, ref: str) -> str:
        """Extract the model name from a $ref string."""
        return ref.split("/")[-1]

    def python_type_from_schema(
        self, schema: OpenAPISchema, field_name: str | None = None
    ) -> tuple[type, FieldInfo | None]:
        """
        Convert an OpenAPI schema to a Python type annotation.

        Args:
            schema: The OpenAPI schema definition
            field_name: Optional field name for context-aware generation

        Returns:
            Tuple of (type annotation, optional FieldInfo)
        """
        # Handle $ref
        if "$ref" in schema:
            model_name = self.get_model_name_from_ref(schema["$ref"])
            resolved = self.resolve_ref(schema["$ref"])
            return self.get_or_create_model(model_name, resolved), None

        schema_type = schema.get("type")
        schema_format = schema.get("format")

        # Build field constraints
        field_kwargs: dict[str, Any] = {}

        if "description" in schema:
            field_kwargs["description"] = schema["description"]
        if "default" in schema:
            field_kwargs["default"] = schema["default"]

        # String type
        if schema_type == "string":
            if "enum" in schema:
                enum_values = schema["enum"]
                # Create an enum type
                enum_name = f"{field_name or 'Value'}Enum"
                enum_type = Enum(enum_name, {v: v for v in enum_values})  # type: ignore
                return enum_type, Field(**field_kwargs) if field_kwargs else None

            if schema_format == "date":
                return date, Field(**field_kwargs) if field_kwargs else None
            if schema_format == "date-time":
                return datetime, Field(**field_kwargs) if field_kwargs else None
            if schema_format == "uuid":
                return UUID, Field(**field_kwargs) if field_kwargs else None

            # Add string constraints
            if "minLength" in schema:
                field_kwargs["min_length"] = schema["minLength"]
            if "maxLength" in schema:
                field_kwargs["max_length"] = schema["maxLength"]
            if "pattern" in schema:
                field_kwargs["pattern"] = schema["pattern"]

            return str, Field(**field_kwargs) if field_kwargs else None

        # Integer type
        if schema_type == "integer":
            if "minimum" in schema:
                field_kwargs["ge"] = schema["minimum"]
            if "maximum" in schema:
                field_kwargs["le"] = schema["maximum"]
            if "exclusiveMinimum" in schema:
                field_kwargs["gt"] = schema["exclusiveMinimum"]
            if "exclusiveMaximum" in schema:
                field_kwargs["lt"] = schema["exclusiveMaximum"]

            return int, Field(**field_kwargs) if field_kwargs else None

        # Number type
        if schema_type == "number":
            if "minimum" in schema:
                field_kwargs["ge"] = schema["minimum"]
            if "maximum" in schema:
                field_kwargs["le"] = schema["maximum"]

            return float, Field(**field_kwargs) if field_kwargs else None

        # Boolean type
        if schema_type == "boolean":
            return bool, Field(**field_kwargs) if field_kwargs else None

        # Array type
        if schema_type == "array":
            items_schema = schema.get("items", {})
            item_type, _ = self.python_type_from_schema(items_schema)

            if "minItems" in schema:
                field_kwargs["min_length"] = schema["minItems"]
            if "maxItems" in schema:
                field_kwargs["max_length"] = schema["maxItems"]

            return list[item_type], Field(**field_kwargs) if field_kwargs else None  # type: ignore

        # Object type
        if schema_type == "object":
            if "properties" in schema:
                # Create an inline model
                model_name = f"{field_name or 'Inline'}Model"
                return (
                    self.get_or_create_model(model_name, schema),
                    Field(**field_kwargs) if field_kwargs else None,
                )

            # Generic object with additionalProperties
            additional = schema.get("additionalProperties")
            if additional:
                if isinstance(additional, dict):
                    value_type, _ = self.python_type_from_schema(additional)
                    return dict[str, value_type], Field(
                        **field_kwargs
                    ) if field_kwargs else None  # type: ignore
                return dict[str, Any], Field(**field_kwargs) if field_kwargs else None

            return dict[str, Any], Field(**field_kwargs) if field_kwargs else None

        # AllOf, OneOf, AnyOf (simplified handling)
        if "allOf" in schema:
            # Merge all schemas
            merged: dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": [],
            }
            for sub_schema in schema["allOf"]:
                if "$ref" in sub_schema:
                    resolved = self.resolve_ref(sub_schema["$ref"])
                    sub_schema = resolved
                merged["properties"].update(sub_schema.get("properties", {}))
                merged["required"].extend(sub_schema.get("required", []))
            return self.python_type_from_schema(merged, field_name)

        if "oneOf" in schema or "anyOf" in schema:
            # For simplicity, use the first option
            options = schema.get("oneOf") or schema.get("anyOf") or []
            if options:
                return self.python_type_from_schema(options[0], field_name)

        # Default to Any
        return Any, None

    def get_or_create_model(self, name: str, schema: OpenAPISchema) -> Type[BaseModel]:
        """
        Get or create a Pydantic model for a schema definition.

        Args:
            name: The model name
            schema: The schema definition

        Returns:
            A Pydantic model class
        """
        if name in self._model_cache:
            return self._model_cache[name]

        # Avoid infinite recursion by adding placeholder
        placeholder_fields = {"__placeholder__": (bool, True)}
        placeholder = create_model(name, **placeholder_fields)  # type: ignore
        self._model_cache[name] = placeholder

        # Build fields
        fields: dict[str, tuple[type, Any]] = {}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for prop_name, prop_schema in properties.items():
            python_type, field_info = self.python_type_from_schema(
                prop_schema, prop_name
            )

            if prop_name not in required:
                python_type = python_type | None  # type: ignore
                if field_info is None:
                    field_info = Field(default=None)
                elif field_info.default is None:
                    field_info = Field(
                        default=None,
                        **{
                            k: v
                            for k, v in field_info._attributes.items()
                            if k not in ("default", "default_factory")
                        },
                    )

            fields[prop_name] = (python_type, field_info or ...)

        # Create the actual model
        model = create_model(name, **fields)  # type: ignore
        self._model_cache[name] = model

        return model

    def create_all_models(self) -> dict[str, Type[BaseModel]]:
        """
        Create Pydantic models for all definitions in the spec.

        Returns:
            Dict mapping definition names to Pydantic model classes
        """
        for name, schema in self._definitions.items():
            self.get_or_create_model(name, schema)

        return self._model_cache.copy()

    def get_request_body_schema(self, path: str, method: str) -> OpenAPISchema | None:
        """
        Get the request body schema for an endpoint.

        Args:
            path: The endpoint path
            method: The HTTP method (lowercase)

        Returns:
            The request body schema or None
        """
        paths = self.spec.get("paths", {})
        path_item = paths.get(path, {})
        operation = path_item.get(method.lower(), {})

        if self._is_swagger2:
            # Swagger 2.0: body parameter
            parameters = operation.get("parameters", [])
            for param in parameters:
                if param.get("in") == "body":
                    return param.get("schema")
        else:
            # OpenAPI 3.x: requestBody
            request_body = operation.get("requestBody", {})
            content = request_body.get("content", {})
            # Prefer JSON
            for content_type in ["application/json", "application/xml", "*/*"]:
                if content_type in content:
                    return content[content_type].get("schema")

        return None

    def get_response_schema(
        self, path: str, method: str, status_code: int | str
    ) -> OpenAPISchema | None:
        """
        Get the response schema for an endpoint and status code.

        Args:
            path: The endpoint path
            method: The HTTP method (lowercase)
            status_code: The HTTP status code

        Returns:
            The response schema or None
        """
        paths = self.spec.get("paths", {})
        path_item = paths.get(path, {})
        operation = path_item.get(method.lower(), {})
        responses = operation.get("responses", {})

        # Try specific status code, then default
        response = responses.get(str(status_code)) or responses.get("default")
        if not response:
            return None

        if self._is_swagger2:
            return response.get("schema")
        else:
            content = response.get("content", {})
            for content_type in ["application/json", "application/xml", "*/*"]:
                if content_type in content:
                    return content[content_type].get("schema")

        return None

    def get_parameters_schema(
        self, path: str, method: str
    ) -> dict[str, list[OpenAPISchema]]:
        """
        Get all parameters for an endpoint grouped by location.

        Args:
            path: The endpoint path
            method: The HTTP method (lowercase)

        Returns:
            Dict mapping parameter location to list of parameter schemas
        """
        paths = self.spec.get("paths", {})
        path_item = paths.get(path, {})
        operation = path_item.get(method.lower(), {})

        # Combine path-level and operation-level parameters
        all_params = path_item.get("parameters", []) + operation.get("parameters", [])

        result: dict[str, list[OpenAPISchema]] = {
            "path": [],
            "query": [],
            "header": [],
            "cookie": [],
            "formData": [],
        }

        for param in all_params:
            location = param.get("in", "query")
            if location in result:
                result[location].append(param)

        return result


class RequestResponseValidator:
    """
    Validates API requests and responses against OpenAPI specifications.
    """

    def __init__(self, parser: OpenAPISchemaParser):
        """
        Initialize with a schema parser.

        Args:
            parser: An initialized OpenAPISchemaParser
        """
        self.parser = parser
        self._models = parser.create_all_models()

    def validate_request_body(
        self, path: str, method: str, body: Any
    ) -> ValidationResult:
        """
        Validate a request body against the OpenAPI schema.

        Args:
            path: The endpoint path
            method: The HTTP method
            body: The request body to validate

        Returns:
            ValidationResult with validation status and any errors
        """
        schema = self.parser.get_request_body_schema(path, method)
        if schema is None:
            # No schema defined, accept anything (or nothing)
            return ValidationResult(valid=True, validated_data=body)

        return self._validate_against_schema(body, schema)

    def validate_response(
        self, path: str, method: str, status_code: int, body: Any
    ) -> ValidationResult:
        """
        Validate a response body against the OpenAPI schema.

        Args:
            path: The endpoint path
            method: The HTTP method
            status_code: The HTTP status code
            body: The response body to validate

        Returns:
            ValidationResult with validation status and any errors
        """
        schema = self.parser.get_response_schema(path, method, status_code)
        if schema is None:
            # No schema defined
            return ValidationResult(valid=True, validated_data=body)

        return self._validate_against_schema(body, schema)

    def _validate_against_schema(
        self, data: Any, schema: OpenAPISchema
    ) -> ValidationResult:
        """
        Validate data against a schema.

        Args:
            data: The data to validate
            schema: The OpenAPI schema

        Returns:
            ValidationResult
        """
        try:
            # Handle $ref
            if "$ref" in schema:
                model_name = self.parser.get_model_name_from_ref(schema["$ref"])
                if model_name in self._models:
                    model = self._models[model_name]
                    validated = model.model_validate(data)
                    return ValidationResult(valid=True, validated_data=validated)

            # Handle array
            if schema.get("type") == "array":
                if not isinstance(data, list):
                    return ValidationResult(
                        valid=False,
                        errors=[f"Expected array, got {type(data).__name__}"],
                    )

                items_schema = schema.get("items", {})
                validated_items = []
                errors = []

                for i, item in enumerate(data):
                    result = self._validate_against_schema(item, items_schema)
                    if result.valid:
                        validated_items.append(result.validated_data)
                    else:
                        errors.extend([f"Item {i}: {e}" for e in result.errors])

                if errors:
                    return ValidationResult(valid=False, errors=errors)
                return ValidationResult(valid=True, validated_data=validated_items)

            # Handle object inline
            if schema.get("type") == "object" and "properties" in schema:
                # Create temporary model
                temp_type, _ = self.parser.python_type_from_schema(schema)
                if isinstance(temp_type, type) and issubclass(temp_type, BaseModel):
                    validated = temp_type.model_validate(data)
                    return ValidationResult(valid=True, validated_data=validated)

            # Handle primitives
            schema_type = schema.get("type")
            if schema_type == "string":
                if not isinstance(data, str):
                    return ValidationResult(
                        valid=False,
                        errors=[f"Expected string, got {type(data).__name__}"],
                    )
                return ValidationResult(valid=True, validated_data=data)

            if schema_type == "integer":
                if not isinstance(data, int) or isinstance(data, bool):
                    return ValidationResult(
                        valid=False,
                        errors=[f"Expected integer, got {type(data).__name__}"],
                    )
                return ValidationResult(valid=True, validated_data=data)

            if schema_type == "number":
                if not isinstance(data, (int, float)) or isinstance(data, bool):
                    return ValidationResult(
                        valid=False,
                        errors=[f"Expected number, got {type(data).__name__}"],
                    )
                return ValidationResult(valid=True, validated_data=data)

            if schema_type == "boolean":
                if not isinstance(data, bool):
                    return ValidationResult(
                        valid=False,
                        errors=[f"Expected boolean, got {type(data).__name__}"],
                    )
                return ValidationResult(valid=True, validated_data=data)

            # Default: accept
            return ValidationResult(valid=True, validated_data=data)

        except Exception as e:
            return ValidationResult(valid=False, errors=[str(e)])


def generate_data_from_schema(
    schema: OpenAPISchema,
    parser: OpenAPISchemaParser,
    generator: DeterministicGenerator,
    field_name: str | None = None,
) -> Any:
    """
    Generate deterministic test data from an OpenAPI schema.

    Args:
        schema: The OpenAPI schema
        parser: The schema parser for resolving refs
        generator: The deterministic generator
        field_name: Optional field name for context-aware generation

    Returns:
        Generated test data matching the schema
    """
    # Handle $ref
    if "$ref" in schema:
        resolved = parser.resolve_ref(schema["$ref"])
        return generate_data_from_schema(resolved, parser, generator, field_name)

    schema_type = schema.get("type")
    schema_format = schema.get("format")

    # String
    if schema_type == "string":
        return generator.generate_string(
            min_length=schema.get("minLength", 1),
            max_length=schema.get("maxLength", 50),
            pattern=schema.get("pattern"),
            enum_values=schema.get("enum"),
            format=schema_format,
            field_name=field_name,
        )

    # Integer
    if schema_type == "integer":
        return generator.generate_integer(
            minimum=schema.get("minimum"),
            maximum=schema.get("maximum"),
            format=schema_format,
        )

    # Number
    if schema_type == "number":
        return generator.generate_number(
            minimum=schema.get("minimum"),
            maximum=schema.get("maximum"),
            format=schema_format,
        )

    # Boolean
    if schema_type == "boolean":
        return generator.generate_boolean()

    # Array
    if schema_type == "array":
        items_schema = schema.get("items", {})
        min_items = schema.get("minItems", 1)
        max_items = schema.get("maxItems", 3)

        return generator.generate_array(
            item_generator=lambda: generate_data_from_schema(
                items_schema, parser, generator
            ),
            min_items=min_items,
            max_items=max_items,
            unique_items=schema.get("uniqueItems", False),
        )

    # Object
    if schema_type == "object":
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        result = {}
        for prop_name, prop_schema in properties.items():
            # Always generate required fields, optionally generate others
            if prop_name in required or generator._rng.random() > 0.3:
                result[prop_name] = generate_data_from_schema(
                    prop_schema, parser, generator, prop_name
                )

        return result

    # AllOf
    if "allOf" in schema:
        result = {}
        for sub_schema in schema["allOf"]:
            sub_data = generate_data_from_schema(sub_schema, parser, generator)
            if isinstance(sub_data, dict):
                result.update(sub_data)
        return result

    # OneOf/AnyOf - pick first
    if "oneOf" in schema or "anyOf" in schema:
        options = schema.get("oneOf") or schema.get("anyOf") or []
        if options:
            return generate_data_from_schema(options[0], parser, generator)

    # Default
    return None


class APITestDataGenerator:
    """
    High-level interface for generating test data from OpenAPI specs.
    """

    def __init__(
        self,
        spec: OpenAPISchema,
        seed: int | str | None = None,
    ):
        """
        Initialize the test data generator.

        Args:
            spec: The OpenAPI specification
            seed: Optional seed for deterministic generation
        """
        self.parser = OpenAPISchemaParser(spec)
        self.generator = DeterministicGenerator(seed)
        self.validator = RequestResponseValidator(self.parser)

    def generate_request_body(self, path: str, method: str) -> Any:
        """
        Generate a valid request body for an endpoint.

        Args:
            path: The endpoint path
            method: The HTTP method

        Returns:
            Generated request body data
        """
        schema = self.parser.get_request_body_schema(path, method)
        if schema is None:
            return None

        return generate_data_from_schema(schema, self.parser, self.generator)

    def generate_path_params(self, path: str, method: str) -> dict[str, Any]:
        """
        Generate path parameters for an endpoint.

        Args:
            path: The endpoint path
            method: The HTTP method

        Returns:
            Dict of path parameter values
        """
        params = self.parser.get_parameters_schema(path, method)
        result = {}

        for param in params.get("path", []):
            param_name = param.get("name", "")
            param_schema = param.get("schema", param)  # Swagger 2.0 vs OpenAPI 3.x
            result[param_name] = generate_data_from_schema(
                param_schema, self.parser, self.generator, param_name
            )

        return result

    def generate_query_params(
        self, path: str, method: str, include_optional: bool = False
    ) -> dict[str, Any]:
        """
        Generate query parameters for an endpoint.

        Args:
            path: The endpoint path
            method: The HTTP method
            include_optional: Whether to include optional parameters

        Returns:
            Dict of query parameter values
        """
        params = self.parser.get_parameters_schema(path, method)
        result = {}

        for param in params.get("query", []):
            param_name = param.get("name", "")
            is_required = param.get("required", False)

            if is_required or include_optional:
                param_schema = param.get("schema", param)
                result[param_name] = generate_data_from_schema(
                    param_schema, self.parser, self.generator, param_name
                )

        return result

    def generate_headers(self, path: str, method: str) -> dict[str, str]:
        """
        Generate header parameters for an endpoint.

        Args:
            path: The endpoint path
            method: The HTTP method

        Returns:
            Dict of header values
        """
        params = self.parser.get_parameters_schema(path, method)
        result = {}

        for param in params.get("header", []):
            param_name = param.get("name", "")
            if param.get("required", False):
                param_schema = param.get("schema", param)
                value = generate_data_from_schema(
                    param_schema, self.parser, self.generator, param_name
                )
                result[param_name] = str(value)

        return result

    def validate_request(self, path: str, method: str, body: Any) -> ValidationResult:
        """Validate a request body."""
        return self.validator.validate_request_body(path, method, body)

    def validate_response(
        self, path: str, method: str, status_code: int, body: Any
    ) -> ValidationResult:
        """Validate a response body."""
        return self.validator.validate_response(path, method, status_code, body)
