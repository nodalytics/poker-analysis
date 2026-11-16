// Kestrel's desktop app: the tables as they play, the models and connectors, and the settings (admin), all through
// the same local API as the web app.
#pragma once
#include <QJsonArray>
#include <QJsonObject>
#include <QMainWindow>
#include <QMap>

#include "client.h"
#include "study.h"
#include "tableview.h"

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPlainTextEdit;
class QPushButton;
class QSpinBox;
class QTableWidget;
class QTabWidget;
class QTextBrowser;
class QStackedWidget;
class QGridLayout;

namespace kestrel_app {

class Window : public QMainWindow {
    Q_OBJECT
  public:
    explicit Window(QWidget* parent = nullptr);
    void showTab(const QString& name);  // tables, models, connectors, settings or log

  private:
    QWidget* header();
    QWidget* tablesPage();
    QWidget* modelsPage();
    QWidget* connectorsPage();
    QWidget* settingsPage();
    QWidget* logPage();

    void applyTheme();
    void setOnline(bool online);
    void onEvent(const QJsonObject& e);
    void refreshTables();
    void showTable();
    void refreshModels();
    void refreshConnectors();
    void loadSettings();
    void saveSettings();
    void showGrid(bool grid);
    void selectTable(int id);
    void act(int slot);
    void newTable();
    void newPractice();
    void runKestreld(const QStringList& args, std::function<void(int, QString)> done);
    void log(const QString& line);

    Client client_;
    bool dark_ = true;
    QJsonArray engines_models_, engines_solvers_, connectors_;
    QMap<int, QJsonObject> tables_;  // id -> table summary, updated by events
    QMap<int, QJsonObject> state_;   // id -> its latest event

    QTabWidget* tabs_ = nullptr;
    ReviewPage* review_ = nullptr;
    SpotPage* spot_ = nullptr;
    TrainerPage* trainer_ = nullptr;
    OpponentsPage* opponents_ = nullptr;
    QLabel *status_ = nullptr, *dot_ = nullptr;
    QPushButton* serverBtn_ = nullptr;
    QListWidget* tableList_ = nullptr;
    QTextBrowser* tableView_ = nullptr;  // the decision and the session
    QTextBrowser* handLog_ = nullptr;
    TableView* drawn_ = nullptr;
    QWidget* moves_ = nullptr;
    QStackedWidget* views_ = nullptr;
    QGridLayout* grid_ = nullptr;
    QMap<int, TableView*> gridViews_;  // the grid's tables, updated live
    QCheckBox* advice_ = nullptr;
    QTableWidget *modelTable_ = nullptr, *solverTable_ = nullptr, *connTable_ = nullptr;
    QPlainTextEdit* log_ = nullptr;

    // settings (admin)
    QSpinBox *port_ = nullptr, *wsPort_ = nullptr, *tcpPort_ = nullptr, *resolveIters_ = nullptr, *searchSamples_ = nullptr, *tsThreads_ = nullptr;
    QCheckBox* lan_ = nullptr;
    QComboBox* defaultModel_ = nullptr;
    QLineEdit *tsBin_ = nullptr, *tsRes_ = nullptr, *sessionsDir_ = nullptr, *packDir_ = nullptr, *dbPath_ = nullptr;
    QLabel *settingsNote_ = nullptr, *lanLinks_ = nullptr;
};

}  // namespace kestrel_app
