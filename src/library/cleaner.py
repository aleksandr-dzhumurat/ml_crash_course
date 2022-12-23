import re

import pandas as pd

REGEX_REPLACEMENTS = [
    ("ё", "e"),
    # https://stackoverflow.com/questions/21932615/regular-expression-for-remove-link
    (r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\S\.-]*)", " "),  # links
    (r"\[(.+)\]\(.+\)", r"\1"),  # markdown links
    (r"[^a-zA-Zа-яА-Я0-9.!?\"': ]", " "),
    (r"\s+", " "),  # multiple spaces and new lines
]


def __clear_series(data: pd.Series) -> pd.Series:
    data = data.str.lower()
    for old, new in REGEX_REPLACEMENTS:
        data = data.str.replace(old, new, regex=True)
    data = data.str.strip()
    return data


def __clear_str(data: str) -> str:
    data = data.lower()
    for old, new in REGEX_REPLACEMENTS:
        data = re.sub(old, new, data)
    data = data.strip()
    return data


def clear(data: pd.Series | str) -> pd.Series | str:
    """Clean data.
    The function can be used to clean a single string or a series of strings.

    Usage examples:
    >>> list = [
    ...     "Hello, world! # 123?",
    ...     "https://www.google.com link to google",
    ...     "[Google](https://www.google.com) link in markdown",
    ...     "line and \\n new line",
    ...     " Умное высказывание. "
    ... ]
    >>> for x in list:
    ...     print(clear(x))
    hello world! 123?
    link to google
    google link in markdown
    line and new line
    умное высказывание.
    >>> data = pd.Series([
    ...     "Hello, world! # 123?",
    ...     "https://www.google.com link to google",
    ...     "[Google](https://www.google.com) link in markdown",
    ...     "line and \\n new line",
    ...     " Умное высказывание. "
    ... ])
    >>> data = clear(data)
    >>> for x in data.values:
    ...    print(x)
    hello world! 123?
    link to google
    google link in markdown
    line and new line
    умное высказывание.

    Function catches wrong type of data:
    >>> number = 1
    >>> clear(number)
    Traceback (most recent call last):
    TypeError: Expected pd.Series or str, got <class 'int'>
    """
    if isinstance(data, pd.Series):
        return __clear_series(data)
    elif isinstance(data, str):
        return __clear_str(data)
    else:
        raise TypeError("Expected pd.Series or str, got {}".format(type(data)))
