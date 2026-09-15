"""Shared SQLite column-affinity classification, used by both the relation and the reader."""


def sqlite_affinity_class(declared_type: str) -> str:
    # Precedence order per SQLite's rules: INT, then CHAR/CLOB/TEXT, then BLOB, then REAL/FLOA/DOUB, else NUMERIC.
    upper = declared_type.upper()
    if "INT" in upper:
        return "INTEGER"
    if "CHAR" in upper or "CLOB" in upper or "TEXT" in upper:
        return "TEXT"
    if "BLOB" in upper:
        return "BLOB"
    if "REAL" in upper or "FLOA" in upper or "DOUB" in upper:
        return "REAL"
    return "NUMERIC"
