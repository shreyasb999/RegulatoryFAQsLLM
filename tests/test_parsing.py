import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from scripts.ingest import TEST_EXPECTS, _validate_schema

@pytest.mark.parametrize("doc_key", TEST_EXPECTS.keys())
def test_schema(doc_key):
    """Ensure each parsed JSON file meets schema and chunk count requirements."""
    _validate_schema(doc_key)
