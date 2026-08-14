import unittest

from real_validation.widgets.theme import (
    PRIMARY, DANGER, ACCENT, QSS, PGG_OPTS, STATE_BADGE_COLORS)


class ThemeTest(unittest.TestCase):
    def test_palette_colors_present_in_qss(self):
        for color in (PRIMARY, DANGER, ACCENT):
            self.assertIn(color, QSS)

    def test_pyqtgraph_is_white_background(self):
        self.assertEqual(PGG_OPTS["background"], "#FFFFFF")
        self.assertEqual(PGG_OPTS["foreground"], "#334E68")

    def test_badge_colors_cover_all_session_states(self):
        from real_validation.session import SessionState
        state_values = {s.value for s in SessionState}
        self.assertTrue(state_values <= set(STATE_BADGE_COLORS))

    def test_configure_is_callable(self):
        from real_validation.widgets.theme import configure_pyqtgraph
        self.assertTrue(callable(configure_pyqtgraph))


if __name__ == "__main__":
    unittest.main()
