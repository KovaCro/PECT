import unittest
from copy import deepcopy
import os
import pect

# tests/problem.tim
#
# events: 3
# rooms: 3
# features: 2
# students: 2
#
#             room
# size    | 2 | 1 | 2 |
#
#             event
# student | 1 | 1 | 0 |
#         | 0 | 1 | 1 |
#
#          feature
#         | 1 | 1 |
# room    | 1 | 0 |
#         | 0 | 1 |
#
#          feature
#         | 1 | 0 |
# event   | 0 | 0 |
#         | 0 | 1 |
#
#            timeslot (0..44)
#             0 on [17, 36]
# event        0 on [8, 9]
#          0 on [41, 42, 43, 44]
#
#             event
#         | 0 |-1 | 0 |
# event   | 1 | 0 | 0 |
#         | 0 | 0 | 0 |


class TestPECTMethods(unittest.TestCase):
    """ Test suite for non-trivial PECT methods """

    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline_problem = pect.parse_problem(os.getcwd() + r"\tests\problem.tim")
        return super().setUpClass()

    def test_is_valid(self):
        """Tests for pect.is_valid"""
        # trivial
        self.assertTrue(
            pect.is_valid(self.baseline_problem, [[-1, -1], [-1, -1], [-1, -1]])
        )
        # valid
        self.assertTrue(pect.is_valid(self.baseline_problem, [[1, 1], [0, 0], [2, 2]]))
        # invalid rule 1
        fails = []
        self.assertFalse(
            pect.is_valid(self.baseline_problem, [[0, 1], [0, 0], [2, 2]], fails)
        )
        self.assertEqual(fails, [{"rule": 1, "event": 1}])
        # invalid rule 2
        fails = []
        self.assertFalse(
            pect.is_valid(self.baseline_problem, [[-1, -1], [0, 1], [2, 2]], fails)
        )
        self.assertEqual(fails, [{"rule": 2, "event": 1}])
        # invalid rule 3
        fails = []
        self.assertFalse(
            pect.is_valid(self.baseline_problem, [[1, 0], [0, 0], [1, 0]], fails)
        )
        self.assertEqual(fails, [{"rule": 3, "event": 2}])
        # invalid rule 4
        fails = []
        self.assertFalse(
            pect.is_valid(self.baseline_problem, [[17, 1], [0, 0], [2, 2]], fails)
        )
        self.assertEqual(fails, [{"rule": 4, "event": 0}])
        # invalid rule 5
        fails = []
        self.assertFalse(
            pect.is_valid(self.baseline_problem, [[1, 1], [3, 0], [2, 2]], fails)
        )
        self.assertEqual(fails, [{"rule": 5, "event": 1}])
        # combination 1
        fails = []
        self.assertFalse(
            pect.is_valid(self.baseline_problem, [[0, 1], [0, 0], [2, 2]], fails)
        )
        self.assertEqual(fails, [{"rule": 1, "event": 1}])
        # combination 2
        fails = []
        self.assertFalse(
            pect.is_valid(self.baseline_problem, [[5, 1], [7, 0], [2, 1]], fails)
        )
        self.assertEqual(fails, [{"rule": 2, "event": 2}, {"rule": 5, "event": 1}])

    def test_is_feasible(self):
        """Tests for pect.is_feasible"""
        # contains uninserted event
        self.assertFalse(
            pect.is_feasible(self.baseline_problem, [[1, 1], [0, 0], [-1, -1]])
        )
        # valid and all events inserted
        self.assertTrue(
            pect.is_feasible(self.baseline_problem, [[1, 1], [0, 0], [2, 2]])
        )
        # fails validation
        self.assertFalse(
            pect.is_feasible(self.baseline_problem, [[1, 0], [0, 0], [1, 0]])
        )

    def test_make_valid(self):
        """Tests for pect.make_valid"""
        # fails rule 2 on event 2 and rule 5 on event 1
        self.assertEqual(
            pect.make_valid(self.baseline_problem, [[5, 1], [7, 0], [2, 1]]),
            [[5, 1], [-1, -1], [-1, -1]],
        )
        # feasible
        self.assertEqual(
            pect.make_valid(self.baseline_problem, [[1, 1], [0, 0], [2, 2]]),
            [[1, 1], [0, 0], [2, 2]],
        )
        # fails rule 2 on event 1
        self.assertEqual(
            pect.make_valid(self.baseline_problem, [[-1, -1], [0, 1], [2, 2]]),
            [[-1, -1], [-1, -1], [2, 2]],
        )

    def test_evaluate(self):
        """Tests for pect.evaluate"""
        self.assertEqual(
            pect.evaluate(self.baseline_problem, [[-1, -1], [-1, -1], [-1, -1]]), (4, 0)
        )
        self.assertEqual(
            pect.evaluate(self.baseline_problem, [[8, 1], [0, 0], [17, 2]]), (0, 4)
        )
        soft_3_test_problem = deepcopy(self.baseline_problem)
        soft_3_test_problem[5][0][2] = 1
        self.assertEqual(
            pect.evaluate(soft_3_test_problem, [[1, 1], [0, 0], [2, 2]]), (0, 1)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
