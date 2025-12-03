"""Tests for the schema module - Pydantic model generation and validation."""

import pytest
from datetime import date, datetime
from uuid import UUID

from llama_wrestler.schema import (
    APITestDataGenerator,
    DeterministicGenerator,
    OpenAPISchemaParser,
    RequestResponseValidator,
    generate_data_from_schema,
)


# Sample OpenAPI specs for testing
PETSTORE_SPEC = {
    "swagger": "2.0",
    "info": {"title": "Petstore", "version": "1.0.0"},
    "basePath": "/v2",
    "paths": {
        "/pet": {
            "post": {
                "operationId": "addPet",
                "parameters": [
                    {
                        "in": "body",
                        "name": "body",
                        "required": True,
                        "schema": {"$ref": "#/definitions/Pet"},
                    }
                ],
                "responses": {
                    "200": {
                        "description": "successful operation",
                        "schema": {"$ref": "#/definitions/Pet"},
                    },
                    "405": {"description": "Invalid input"},
                },
            }
        },
        "/pet/{petId}": {
            "get": {
                "operationId": "getPetById",
                "parameters": [
                    {
                        "name": "petId",
                        "in": "path",
                        "required": True,
                        "type": "integer",
                        "format": "int64",
                    }
                ],
                "responses": {
                    "200": {
                        "description": "successful operation",
                        "schema": {"$ref": "#/definitions/Pet"},
                    },
                    "400": {"description": "Invalid ID supplied"},
                    "404": {"description": "Pet not found"},
                },
            }
        },
        "/pet/findByStatus": {
            "get": {
                "operationId": "findPetsByStatus",
                "parameters": [
                    {
                        "name": "status",
                        "in": "query",
                        "required": True,
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["available", "pending", "sold"],
                        },
                    }
                ],
                "responses": {
                    "200": {
                        "description": "successful operation",
                        "schema": {
                            "type": "array",
                            "items": {"$ref": "#/definitions/Pet"},
                        },
                    }
                },
            }
        },
    },
    "definitions": {
        "Pet": {
            "type": "object",
            "required": ["name", "photoUrls"],
            "properties": {
                "id": {"type": "integer", "format": "int64"},
                "name": {"type": "string", "example": "doggie"},
                "photoUrls": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "status": {
                    "type": "string",
                    "enum": ["available", "pending", "sold"],
                },
                "category": {"$ref": "#/definitions/Category"},
            },
        },
        "Category": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "format": "int64"},
                "name": {"type": "string"},
            },
        },
        "User": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "format": "int64"},
                "username": {"type": "string"},
                "email": {"type": "string", "format": "email"},
                "firstName": {"type": "string"},
                "lastName": {"type": "string"},
            },
        },
    },
}


OPENAPI3_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Test API", "version": "1.0.0"},
    "paths": {
        "/users": {
            "post": {
                "operationId": "createUser",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/User"}
                        }
                    },
                },
                "responses": {
                    "201": {
                        "description": "Created",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/User"}
                            }
                        },
                    }
                },
            }
        }
    },
    "components": {
        "schemas": {
            "User": {
                "type": "object",
                "required": ["email", "name"],
                "properties": {
                    "id": {"type": "integer"},
                    "email": {"type": "string", "format": "email"},
                    "name": {"type": "string", "minLength": 1, "maxLength": 100},
                    "age": {"type": "integer", "minimum": 0, "maximum": 150},
                    "isActive": {"type": "boolean"},
                },
            }
        }
    },
}


class TestDeterministicGenerator:
    """Tests for the DeterministicGenerator class."""

    def test_same_seed_produces_same_values(self):
        """Generator with same seed produces identical values."""
        gen1 = DeterministicGenerator(seed=42)
        gen2 = DeterministicGenerator(seed=42)

        for _ in range(10):
            assert gen1.generate_integer() == gen2.generate_integer()
            assert gen1.generate_string() == gen2.generate_string()
            assert gen1.generate_boolean() == gen2.generate_boolean()

    def test_different_seeds_produce_different_values(self):
        """Generator with different seeds produces different values."""
        gen1 = DeterministicGenerator(seed=42)
        gen2 = DeterministicGenerator(seed=43)

        # At least one value should differ
        values1 = [gen1.generate_integer() for _ in range(10)]
        values2 = [gen2.generate_integer() for _ in range(10)]
        assert values1 != values2

    def test_string_seed_works(self):
        """String seeds are hashed consistently."""
        gen1 = DeterministicGenerator(seed="my-test-seed")
        gen2 = DeterministicGenerator(seed="my-test-seed")

        assert gen1.generate_string() == gen2.generate_string()

    def test_reset_reproduces_values(self):
        """Reset returns generator to initial state."""
        gen = DeterministicGenerator(seed=42)
        first_values = [gen.generate_integer() for _ in range(5)]

        gen.reset()
        second_values = [gen.generate_integer() for _ in range(5)]

        assert first_values == second_values

    def test_generate_string_with_enum(self):
        """String generation respects enum constraints."""
        gen = DeterministicGenerator(seed=42)
        enum_values = ["red", "green", "blue"]

        for _ in range(10):
            value = gen.generate_string(enum_values=enum_values)
            assert value in enum_values

    def test_generate_string_with_length_constraints(self):
        """String generation respects length constraints."""
        gen = DeterministicGenerator(seed=42)

        for _ in range(10):
            value = gen.generate_string(min_length=5, max_length=10)
            assert 5 <= len(value) <= 10

    def test_generate_integer_with_constraints(self):
        """Integer generation respects min/max constraints."""
        gen = DeterministicGenerator(seed=42)

        for _ in range(10):
            value = gen.generate_integer(minimum=10, maximum=20)
            assert 10 <= value <= 20

    def test_generate_number_with_constraints(self):
        """Number generation respects min/max constraints."""
        gen = DeterministicGenerator(seed=42)

        for _ in range(10):
            value = gen.generate_number(minimum=1.0, maximum=2.0)
            assert 1.0 <= value <= 2.0

    def test_generate_email(self):
        """Email generation produces valid-looking emails."""
        gen = DeterministicGenerator(seed=42)
        email = gen.generate_email()

        assert "@" in email
        assert "." in email.split("@")[1]

    def test_generate_url(self):
        """URL generation produces valid-looking URLs."""
        gen = DeterministicGenerator(seed=42)
        url = gen.generate_url()

        assert url.startswith("https://")

    def test_generate_uuid(self):
        """UUID generation produces valid UUIDs."""
        gen = DeterministicGenerator(seed=42)
        uuid = gen.generate_uuid()

        assert isinstance(uuid, UUID)

    def test_generate_date(self):
        """Date generation produces valid dates."""
        gen = DeterministicGenerator(seed=42)
        d = gen.generate_date()

        assert isinstance(d, date)

    def test_generate_datetime(self):
        """Datetime generation produces valid datetimes."""
        gen = DeterministicGenerator(seed=42)
        dt = gen.generate_datetime()

        assert isinstance(dt, datetime)

    def test_generate_array(self):
        """Array generation respects item count constraints."""
        gen = DeterministicGenerator(seed=42)

        arr = gen.generate_array(
            item_generator=lambda: gen.generate_integer(),
            min_items=2,
            max_items=5,
        )

        assert isinstance(arr, list)
        assert 2 <= len(arr) <= 5

    def test_generate_formatted_string_date(self):
        """String generation with date format produces ISO date."""
        gen = DeterministicGenerator(seed=42)
        value = gen.generate_string(format="date")

        # Should be valid ISO date format
        date.fromisoformat(value)

    def test_generate_formatted_string_datetime(self):
        """String generation with date-time format produces ISO datetime."""
        gen = DeterministicGenerator(seed=42)
        value = gen.generate_string(format="date-time")

        # Should be valid ISO datetime format
        datetime.fromisoformat(value)

    def test_field_name_inference_email(self):
        """Email field names trigger email generation."""
        gen = DeterministicGenerator(seed=42)
        value = gen.generate_string(field_name="userEmail")

        assert "@" in value

    def test_field_name_inference_name(self):
        """Name field names trigger name generation."""
        gen = DeterministicGenerator(seed=42)
        value = gen.generate_string(field_name="firstName")

        # Should be a first name from the pool
        assert value in gen.FIRST_NAMES


class TestOpenAPISchemaParser:
    """Tests for the OpenAPISchemaParser class."""

    def test_parse_swagger2_definitions(self):
        """Parser correctly identifies Swagger 2.0 definitions."""
        parser = OpenAPISchemaParser(PETSTORE_SPEC)
        definitions = parser.get_definitions()

        assert "Pet" in definitions
        assert "Category" in definitions
        assert "User" in definitions

    def test_parse_openapi3_schemas(self):
        """Parser correctly identifies OpenAPI 3.0 schemas."""
        parser = OpenAPISchemaParser(OPENAPI3_SPEC)
        definitions = parser.get_definitions()

        assert "User" in definitions

    def test_resolve_ref_swagger2(self):
        """Resolves $ref pointers in Swagger 2.0 spec."""
        parser = OpenAPISchemaParser(PETSTORE_SPEC)
        resolved = parser.resolve_ref("#/definitions/Pet")

        assert resolved["type"] == "object"
        assert "name" in resolved["properties"]

    def test_resolve_ref_openapi3(self):
        """Resolves $ref pointers in OpenAPI 3.0 spec."""
        parser = OpenAPISchemaParser(OPENAPI3_SPEC)
        resolved = parser.resolve_ref("#/components/schemas/User")

        assert resolved["type"] == "object"
        assert "email" in resolved["properties"]

    def test_create_model_with_required_fields(self):
        """Created model enforces required fields."""
        parser = OpenAPISchemaParser(PETSTORE_SPEC)
        PetModel = parser.get_or_create_model("Pet", parser.get_definitions()["Pet"])

        # Should fail without required fields
        with pytest.raises(Exception):
            PetModel()

        # Should succeed with required fields
        pet = PetModel(name="Buddy", photoUrls=["http://example.com/photo.jpg"])
        assert pet.name == "Buddy"

    def test_create_model_with_optional_fields(self):
        """Created model allows optional fields."""
        parser = OpenAPISchemaParser(PETSTORE_SPEC)
        PetModel = parser.get_or_create_model("Pet", parser.get_definitions()["Pet"])

        pet = PetModel(name="Buddy", photoUrls=["http://example.com/photo.jpg"])
        assert pet.id is None
        assert pet.status is None

    def test_create_all_models(self):
        """Creates models for all definitions."""
        parser = OpenAPISchemaParser(PETSTORE_SPEC)
        models = parser.create_all_models()

        assert "Pet" in models
        assert "Category" in models
        assert "User" in models

    def test_get_request_body_schema_swagger2(self):
        """Gets request body schema from Swagger 2.0 spec."""
        parser = OpenAPISchemaParser(PETSTORE_SPEC)
        schema = parser.get_request_body_schema("/pet", "post")

        assert schema is not None
        assert "$ref" in schema

    def test_get_request_body_schema_openapi3(self):
        """Gets request body schema from OpenAPI 3.0 spec."""
        parser = OpenAPISchemaParser(OPENAPI3_SPEC)
        schema = parser.get_request_body_schema("/users", "post")

        assert schema is not None
        assert "$ref" in schema

    def test_get_response_schema(self):
        """Gets response schema for endpoint and status code."""
        parser = OpenAPISchemaParser(PETSTORE_SPEC)
        schema = parser.get_response_schema("/pet/{petId}", "get", 200)

        assert schema is not None
        assert "$ref" in schema

    def test_get_parameters_schema(self):
        """Gets parameters grouped by location."""
        parser = OpenAPISchemaParser(PETSTORE_SPEC)
        params = parser.get_parameters_schema("/pet/{petId}", "get")

        assert len(params["path"]) == 1
        assert params["path"][0]["name"] == "petId"

    def test_get_query_parameters(self):
        """Gets query parameters from endpoint."""
        parser = OpenAPISchemaParser(PETSTORE_SPEC)
        params = parser.get_parameters_schema("/pet/findByStatus", "get")

        assert len(params["query"]) == 1
        assert params["query"][0]["name"] == "status"


class TestRequestResponseValidator:
    """Tests for the RequestResponseValidator class."""

    def test_validate_valid_request_body(self):
        """Validates a correct request body."""
        parser = OpenAPISchemaParser(PETSTORE_SPEC)
        validator = RequestResponseValidator(parser)

        result = validator.validate_request_body(
            "/pet",
            "post",
            {"name": "Buddy", "photoUrls": ["http://example.com/photo.jpg"]},
        )

        assert result.valid
        assert not result.errors

    def test_validate_invalid_request_body(self):
        """Detects invalid request body."""
        parser = OpenAPISchemaParser(PETSTORE_SPEC)
        validator = RequestResponseValidator(parser)

        # Missing required field 'photoUrls'
        result = validator.validate_request_body("/pet", "post", {"name": "Buddy"})

        assert not result.valid
        assert len(result.errors) > 0

    def test_validate_valid_response(self):
        """Validates a correct response body."""
        parser = OpenAPISchemaParser(PETSTORE_SPEC)
        validator = RequestResponseValidator(parser)

        result = validator.validate_response(
            "/pet/{petId}",
            "get",
            200,
            {
                "id": 1,
                "name": "Buddy",
                "photoUrls": ["http://example.com/photo.jpg"],
                "status": "available",
            },
        )

        assert result.valid

    def test_validate_array_response(self):
        """Validates array response body."""
        parser = OpenAPISchemaParser(PETSTORE_SPEC)
        validator = RequestResponseValidator(parser)

        result = validator.validate_response(
            "/pet/findByStatus",
            "get",
            200,
            [
                {"name": "Buddy", "photoUrls": ["http://example.com/photo1.jpg"]},
                {"name": "Max", "photoUrls": ["http://example.com/photo2.jpg"]},
            ],
        )

        assert result.valid

    def test_validate_no_schema_defined(self):
        """Accepts any data when no schema is defined."""
        parser = OpenAPISchemaParser(PETSTORE_SPEC)
        validator = RequestResponseValidator(parser)

        # Non-existent endpoint
        result = validator.validate_request_body(
            "/nonexistent", "get", {"anything": "goes"}
        )

        assert result.valid


class TestGenerateDataFromSchema:
    """Tests for the generate_data_from_schema function."""

    def test_generate_object_data(self):
        """Generates valid object data from schema."""
        parser = OpenAPISchemaParser(PETSTORE_SPEC)
        generator = DeterministicGenerator(seed=42)

        schema = parser.get_definitions()["Pet"]
        data = generate_data_from_schema(schema, parser, generator)

        assert isinstance(data, dict)
        assert "name" in data  # Required field
        assert "photoUrls" in data  # Required field
        assert isinstance(data["photoUrls"], list)

    def test_generate_array_data(self):
        """Generates valid array data from schema."""
        parser = OpenAPISchemaParser(PETSTORE_SPEC)
        generator = DeterministicGenerator(seed=42)

        schema = {"type": "array", "items": {"type": "string"}, "minItems": 2}
        data = generate_data_from_schema(schema, parser, generator)

        assert isinstance(data, list)
        assert len(data) >= 2
        assert all(isinstance(item, str) for item in data)

    def test_generate_ref_data(self):
        """Generates data for $ref schemas."""
        parser = OpenAPISchemaParser(PETSTORE_SPEC)
        generator = DeterministicGenerator(seed=42)

        schema = {"$ref": "#/definitions/Category"}
        data = generate_data_from_schema(schema, parser, generator)

        assert isinstance(data, dict)

    def test_generate_enum_data(self):
        """Generates valid enum values."""
        parser = OpenAPISchemaParser(PETSTORE_SPEC)
        generator = DeterministicGenerator(seed=42)

        schema = {"type": "string", "enum": ["available", "pending", "sold"]}

        for _ in range(10):
            data = generate_data_from_schema(schema, parser, generator)
            assert data in ["available", "pending", "sold"]

    def test_generate_nested_object(self):
        """Generates data for nested objects."""
        parser = OpenAPISchemaParser(PETSTORE_SPEC)
        generator = DeterministicGenerator(seed=42)

        # Pet with nested Category
        schema = parser.get_definitions()["Pet"]
        data = generate_data_from_schema(schema, parser, generator)

        # Category might be generated (optional)
        if "category" in data:
            assert isinstance(data["category"], dict)


class TestAPITestDataGenerator:
    """Tests for the high-level APITestDataGenerator class."""

    def test_generate_request_body(self):
        """Generates valid request body for endpoint."""
        gen = APITestDataGenerator(PETSTORE_SPEC, seed=42)
        body = gen.generate_request_body("/pet", "post")

        assert body is not None
        assert "name" in body
        assert "photoUrls" in body

    def test_generate_path_params(self):
        """Generates path parameters for endpoint."""
        gen = APITestDataGenerator(PETSTORE_SPEC, seed=42)
        params = gen.generate_path_params("/pet/{petId}", "get")

        assert "petId" in params
        assert isinstance(params["petId"], int)

    def test_generate_query_params_required_only(self):
        """Generates only required query parameters by default."""
        gen = APITestDataGenerator(PETSTORE_SPEC, seed=42)
        params = gen.generate_query_params("/pet/findByStatus", "get")

        assert "status" in params

    def test_validate_request(self):
        """Validates request using internal validator."""
        gen = APITestDataGenerator(PETSTORE_SPEC, seed=42)
        body = gen.generate_request_body("/pet", "post")
        result = gen.validate_request("/pet", "post", body)

        assert result.valid

    def test_validate_response(self):
        """Validates response using internal validator."""
        gen = APITestDataGenerator(PETSTORE_SPEC, seed=42)
        body = gen.generate_request_body("/pet", "post")
        result = gen.validate_response("/pet", "post", 200, body)

        # Generated data should be valid
        assert result.valid

    def test_reproducible_generation(self):
        """Same seed produces identical data."""
        gen1 = APITestDataGenerator(PETSTORE_SPEC, seed=42)
        gen2 = APITestDataGenerator(PETSTORE_SPEC, seed=42)

        body1 = gen1.generate_request_body("/pet", "post")
        body2 = gen2.generate_request_body("/pet", "post")

        assert body1 == body2
