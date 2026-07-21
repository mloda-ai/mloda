"""Shared option-key specs for the input_data readers.

The readers are BaseInputData classes without PROPERTY_MAPPING; the wrapper feature groups declare these user keys."""

from mloda.provider import property_spec

DOCUMENT_SUFFIXES = "document_suffixes"
DATA_ACCESS_HANDLE = "data_access_handle"

DOCUMENT_SUFFIXES_SPEC = property_spec(
    "Structured file suffixes (e.g. '.json') ReadDocument may claim as documents; ReadFile auto-excludes them for that feature. Owned by the ReadFile and ReadDocument matchers.",
    default=None,
    context=False,
)
DATA_ACCESS_HANDLE_SPEC = property_spec(
    "Name of a DataAccessCollection handle pinning which registered file or credentials entry the reader uses; see docs/docs/in_depth/named-data-access-handles.md.",
    default=None,
)
