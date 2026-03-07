def parse_version(version_str):
    """Parse version string into a tuple of integers"""
    try:
        if "+" in version_str:
            version_str = version_str.split("+")[0]
        return tuple(map(int, version_str.split(".")))
    except:
        return (0, 0, 0)