// The desktop app's study screens: Review (finished hands from the session logs), Spot study (Kestrel's advice for
// any spot you describe) and Trainer (pre-flop tournament spots: your move against the model's).
#pragma once
#include <QJsonArray>
#include <QJsonObject>
#include <QWidget>
#include <functional>
#include <random>

#include "client.h"
#include "tableview.h"

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPlainTextEdit;
class QSpinBox;
class QTextBrowser;
class QHBoxLayout;
class QTableWidget;

namespace kestrel_app {

QString sessionsDir();  // where kestreld writes session logs (settings sessions_dir, else the data folder)

// every player's prize equity under both models (Malmuth-Harville, Malmuth-Weitzman), as a table: done(html)
void equityTable(Client& c, const QJsonArray& players, const QJsonArray& payouts, const QString& me, bool dark,
                 std::function<void(const QString&)> done);

class ReviewPage : public QWidget {
    Q_OBJECT
  public:
    ReviewPage(Client& c, std::function<bool()> dark, QWidget* parent = nullptr);
    void reload();

  private:
    void openSession();
    void showHand();
    Client& client_;
    std::function<bool()> dark_;
    QString dir_;
    QListWidget *sessions_, *hands_;
    QLabel* summary_;
    QTextBrowser* detail_;
    TableView* drawn_;
    QJsonArray lines_;
};

class SpotPage : public QWidget {
    Q_OBJECT
  public:
    SpotPage(Client& c, std::function<bool()> dark, QWidget* parent = nullptr);
    void setModels(const QStringList& nineMax);

  private:
    void ask();
    Client& client_;
    std::function<bool()> dark_;
    QComboBox *game_, *model_, *solver_;
    QWidget *hu_, *table_;
    // heads-up
    QLineEdit *huAction_, *huHole_, *huBoard_;
    QComboBox* huPos_;
    // a 2-9 handed table
    QLineEdit *stacks_, *hole_, *board_, *payouts_;
    QSpinBox *button_, *me_, *sb_, *bb_, *ante_;
    QPlainTextEdit* actions_;
    QTextBrowser* answer_;
    TableView* drawn_;
};

class TrainerPage : public QWidget {
    Q_OBJECT
  public:
    TrainerPage(Client& c, std::function<bool()> dark, QWidget* parent = nullptr);
    void setModels(const QStringList& nineMax);

  private:
    void deal();
    void answer(int k);
    Client& client_;
    std::function<bool()> dark_;
    std::mt19937 rng_{std::random_device{}()};
    QComboBox* model_;
    QLabel *prompt_, *score_;
    TableView* drawn_;
    QWidget* choices_;
    QTextBrowser* verdict_;
    QJsonObject advice_, spot_;
    QString equity_;  // your equity under both models, shown with the verdict
    int asked_ = 0, agreed_ = 0;
    double probSum_ = 0;
    bool open_ = false;
};

// the opponents Kestrel has met, by platform and username, with their statistics and the hands they showed
class OpponentsPage : public QWidget {
    Q_OBJECT
  public:
    OpponentsPage(Client& c, QWidget* parent = nullptr);
    void reload();

  private:
    void showPlatform();
    Client& client_;
    QComboBox* platform_;
    class QTableWidget* table_;
    QTextBrowser* shown_;
    QJsonObject players_;
};

}  // namespace kestrel_app
