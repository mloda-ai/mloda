import difflib


def is_attribute_column_read(error: Exception, frame_types: tuple[type, ...], check_message: bool = True) -> bool:
    """True for an AttributeError from reading an unknown, non-method name on a frame (a `df.x` column read)."""
    if not isinstance(error, AttributeError) or not isinstance(error.obj, frame_types):
        return False
    name = error.name
    if not isinstance(name, str) or name.startswith("_"):
        return False
    if check_message and str(error) != f"'{type(error.obj).__name__}' object has no attribute '{name}'":
        return False
    attributes = dir(type(error.obj))
    if name in attributes:
        return False
    if len(name) < 6:
        return True
    public = [a for a in attributes if not a.startswith("_")]
    return not difflib.get_close_matches(name, public, n=1, cutoff=0.8)
