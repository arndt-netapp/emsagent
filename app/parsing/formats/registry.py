from typing import List

from app.parsing.base import EmsFormatParser
from app.parsing.formats.autosupport_format import AutosupportEmsFormatParser
from app.parsing.formats.ems_text_format import EmsTextFormatParser

# Tried in order, first match wins — but the two formats are mutually exclusive
# by construction (see the regex comments in each), so the order here is not
# load-bearing. `test_the_two_formats_do_not_overlap` keeps it that way.
PARSERS: List[EmsFormatParser] = [EmsTextFormatParser(), AutosupportEmsFormatParser()]


def detect_format(sample_lines: List[str]):
    for parser in PARSERS:
        if parser.detect(sample_lines):
            return parser
    return None
