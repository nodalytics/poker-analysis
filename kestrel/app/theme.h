// The web app's colours (web/index.html :root) as a Qt style sheet, dark and light.
#pragma once
#include <QString>

namespace kestrel_app {

struct Palette {
    const char *bg, *panel, *panel2, *line, *text, *muted, *accent, *accent2, *win, *loss;
};

inline const Palette& palette(bool dark) {
    static const Palette d{"#0A0C11", "#12151C", "#181C25", "#232837", "#E6E8EE",
                           "#8A91A3", "#F97316", "#FDBA74", "#22C55E", "#EF4444"};
    static const Palette l{"#F5F6F8", "#FFFFFF", "#F0F2F6", "#E2E5EC", "#111827",
                           "#6B7280", "#F97316", "#C2410C", "#16A34A", "#DC2626"};
    return dark ? d : l;
}

inline QString styleSheet(bool dark) {
    const Palette& p = palette(dark);
    return QString(R"(
* { font-family: Inter, "Segoe UI", system-ui, sans-serif; font-size: 13px; color: %5; }
QMainWindow, QDialog, QWidget#page { background: %1; }
QWidget#header { background: %1; border-bottom: 1px solid %4; }
QLabel#title { font-size: 17px; font-weight: 600; }
QLabel#muted, QLabel[muted="true"] { color: %6; }
QLabel#h3 { color: %6; font-size: 11px; font-weight: 600; letter-spacing: 1px; }
QFrame#card { background: %2; border: 1px solid %4; border-radius: 12px; }
QTabWidget::pane { border: none; }
QTabBar::tab { background: transparent; color: %6; padding: 8px 16px; border: none; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: %5; border-bottom: 2px solid %7; }
QTabBar::tab:hover { color: %5; }
QPushButton { background: %3; border: 1px solid %4; border-radius: 8px; padding: 6px 14px; }
QPushButton:hover { border-color: %6; }
QPushButton:disabled { color: %6; }
QPushButton#primary { background: %7; border-color: %7; color: #FFFFFF; font-weight: 600; }
QPushButton#primary:hover { background: %8; border-color: %8; }
QLineEdit, QSpinBox, QComboBox, QPlainTextEdit, QTextBrowser, QListWidget, QTableWidget {
  background: %2; border: 1px solid %4; border-radius: 8px; padding: 5px; selection-background-color: %7; }
QComboBox QAbstractItemView { background: %2; border: 1px solid %4; }
QHeaderView::section { background: %3; color: %6; border: none; border-bottom: 1px solid %4; padding: 6px; }
QTableWidget { gridline-color: %4; }
QListWidget::item { padding: 8px; border-radius: 6px; }
QListWidget::item:selected { background: %3; color: %5; border: 1px solid %7; }
QCheckBox::indicator { width: 16px; height: 16px; }
QScrollBar:vertical { background: transparent; width: 10px; }
QScrollBar::handle:vertical { background: %4; border-radius: 5px; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QToolTip { background: %2; border: 1px solid %4; color: %5; }
)")
        .arg(p.bg, p.panel, p.panel2, p.line, p.text, p.muted, p.accent, p.accent2);
}

}  // namespace kestrel_app
