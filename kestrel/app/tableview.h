// A table drawn from Kestrel's events, as the web app draws it: the seats around an oval (practice and 9-max
// tables) or two seats (heads-up), the cards, stacks, bets and the pot, the board, and each seat's last action.
#pragma once
#include <QJsonObject>
#include <QWidget>

namespace kestrel_app {

class TableView : public QWidget {
    Q_OBJECT
  public:
    explicit TableView(QWidget* parent = nullptr);
    void setState(const QJsonObject& e, const QJsonObject& table, bool dark);
    void setCompact(bool c) { compact_ = c; update(); }

  signals:
    void clicked();

  protected:
    void paintEvent(QPaintEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override { emit clicked(); }

  private:
    QJsonObject e_, t_;
    bool dark_ = true, compact_ = false;
};

}  // namespace kestrel_app
