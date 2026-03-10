from pathlib import Path


def load_h5_path() -> Path:
    """
    Utility function to load the current h5 file path from lyse.

    Returns
    -------
    Path
        The h5 file path of the current shot being analyzed in lyse.
    """
    import lyse

    # Is this script being run from within an interactive lyse session?
    if lyse.spinning_top:
        # If so, use the filepath of the current h5_path
        return Path(lyse.path)
    else:
        # If not, get the filepath of the last h5_path of the lyse DataFrame
        df = lyse.data()
        return Path(df.filepath.iloc[-1])
