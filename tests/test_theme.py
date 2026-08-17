import unittest

from real_validation.gui.theme import (
    BG, CARD, BORDER, TAB_BG,
    TEXT_MAIN, TEXT_STRONG, TEXT_MUTED, INPUT_BG,
    PRIMARY, PRIMARY_HOVER, PRIMARY_PRESSED,
    DANGER, DANGER_HOVER, DANGER_PRESSED,
    ACCENT, ACCENT_HOVER, ACCENT_PRESSED,
    QSS, PGG_OPTS, STATE_BADGE_COLORS,
)

# 所有在 QSS 里被引用的调色板常量(OK_GREEN/WARN_ORANGE 仅用于状态色映射,不在 QSS)。
_QSS_PALETTE_CONSTANTS = (
    BG, CARD, BORDER, TAB_BG,
    TEXT_MAIN, TEXT_STRONG, TEXT_MUTED, INPUT_BG,
    PRIMARY, PRIMARY_HOVER, PRIMARY_PRESSED,
    DANGER, DANGER_HOVER, DANGER_PRESSED,
    ACCENT, ACCENT_HOVER, ACCENT_PRESSED,
)


class ThemeTest(unittest.TestCase):
    def test_palette_colors_present_in_qss(self):
        for color in _QSS_PALETTE_CONSTANTS:
            self.assertIn(color, QSS)

    def test_pyqtgraph_is_white_background(self):
        self.assertEqual(PGG_OPTS["background"], "#FFFFFF")
        self.assertEqual(PGG_OPTS["foreground"], "#334E68")

    def test_badge_colors_cover_all_session_states(self):
        from real_validation.core.session import SessionState
        state_values = {s.value for s in SessionState}
        self.assertTrue(state_values <= set(STATE_BADGE_COLORS))

    def test_configure_is_callable(self):
        from real_validation.gui.theme import configure_pyqtgraph
        configure_pyqtgraph()
        import pyqtgraph as pg
        self.assertEqual(pg.getConfigOption("background"), PGG_OPTS["background"])


if __name__ == "__main__":
    unittest.main()
