def get_class_from_string(path: str, namespace: dict = None):
    """
    Given a module/class path string (possibly with alias), return the actual class or object.
    """
    if namespace is None:
        namespace = globals()
    parts = path.split('.')
    # Resolve the first part: either alias in namespace or import module
    first = parts[0]
    if first in namespace:
        obj = namespace[first]
    else:
        obj = importlib.import_module(first)
    # Traverse the rest of the attributes
    for attr in parts[1:]:
        obj = getattr(obj, attr)
    return obj
