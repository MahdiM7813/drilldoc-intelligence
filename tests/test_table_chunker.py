from __future__ import annotations

import unittest

from rag_system.chunking.table_chunker import _row_labels


class TableRowLabelTests(unittest.TestCase):
    def test_numeric_values_in_later_cells_are_not_row_labels(self):
        rows = [{"cells": [{"value": "Maximum inlet pressure"}, {"value": "2.4"}, {"value": "MPa"}]},
                {"cells": [{"value": "Test line length"}, {"value": "12.0"}, {"value": "metres"}]}]
        self.assertEqual(_row_labels(rows), [])

    def test_hierarchical_identifier_in_first_cell_is_preserved(self):
        self.assertEqual(_row_labels([{"cells": [{"value": "1.2"}, {"value": "Limit"}]}]), ["1.2"])


if __name__ == "__main__":
    unittest.main()
