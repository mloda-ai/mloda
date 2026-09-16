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
    # Diverges from SQLite's own rule 3 (no type specified -> BLOB affinity): both call
    # sites have always treated the undeclared case as their generic fallback
    # (pa.string() / None), so NUMERIC is returned here to preserve that behavior.
    return "NUMERIC"
