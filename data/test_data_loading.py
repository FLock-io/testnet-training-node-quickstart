import pytest
import json


@pytest.fixture
def load_data():
    with open("function_calling_demo.jsonl", "r", encoding="utf8") as f:
        data_list = f.readlines()
    return data_list


def test_data_loading(load_data):
    # Verify data was loaded correctly
    assert load_data is not None, "Data should be loaded"
    assert len(load_data) > 0, "Data list should not be empty"

    # Verify each line is a valid JSON with required fields
    for line in load_data:
        try:
            data = json.loads(line)
            assert isinstance(data, dict), (
                "Each line should be a JSON object (dictionary)"
            )

            # Check for "conversations" key
            assert "conversations" in data, (
                "'conversations' key is missing in the JSON object"
            )
            assert isinstance(data["conversations"], list), (
                "'conversations' should be a list"
            )

            # Check for 'user' and 'assistant' roles within "conversations"
            roles = {conv["role"] for conv in data["conversations"] if "role" in conv}
            assert "user" in roles, "Role 'user' is missing in conversations"
            assert "assistant" in roles, "Role 'assistant' is missing in conversations"

        except json.JSONDecodeError:
            pytest.fail("Each line in the data file should be valid JSON")
