#include "window.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QJsonArray>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QSpinBox>
#include <QSplitter>
#include <QShortcut>
#include <QStackedWidget>
#include <QStyleHints>
#include <QGridLayout>
#include <QTabWidget>
#include <QTableWidget>
#include <QTextBrowser>
#include <QUrl>
#include <QVBoxLayout>

#include "theme.h"

namespace kestrel_app {

namespace {

QFrame* card(QWidget* inner, const QString& title = {}) {
    auto f = new QFrame;
    f->setObjectName("card");
    auto l = new QVBoxLayout(f);
    l->setContentsMargins(16, 14, 16, 16);
    l->setSpacing(10);
    if (!title.isEmpty()) {
        auto h = new QLabel(title.toUpper());
        h->setObjectName("h3");
        l->addWidget(h);
    }
    l->addWidget(inner);
    return f;
}

QWidget* page(QLayout* l) {
    auto w = new QWidget;
    w->setObjectName("page");
    w->setLayout(l);
    l->setContentsMargins(20, 18, 20, 18);
    return w;
}

// message boxes always show plain text: output from kestreld, connector files or the server is never read as HTML
int box(QWidget* parent, QMessageBox::Icon icon, const QString& text, QMessageBox::StandardButtons buttons = QMessageBox::Ok) {
    QMessageBox m(icon, "Kestrel", text, buttons, parent);
    m.setTextFormat(Qt::PlainText);
    return m.exec();
}
void info(QWidget* w, const QString& t) { box(w, QMessageBox::Information, t); }
void warn(QWidget* w, const QString& t) { box(w, QMessageBox::Warning, t); }
bool ask(QWidget* w, const QString& t) {
    return box(w, QMessageBox::Question, t, QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes;
}

QStringList names(const QJsonArray& a, std::function<bool(const QJsonObject&)> keep) {
    QStringList out;
    for (const auto& v : a)
        if (keep(v.toObject())) out << v.toObject().value("name").toString();
    return out;
}

const char* kStreetNames[] = {"Pre-flop", "Flop", "Turn", "River"};

QString sign(double x) { return (x >= 0 ? "+" : "") + QString::number(x, 'f', 1); }

QString esc(const QString& s) { return s.toHtmlEscaped(); }

QTableWidgetItem* item(const QString& s) {
    auto i = new QTableWidgetItem(s);
    i->setFlags(i->flags() & ~Qt::ItemIsEditable);
    return i;
}

}  // namespace

Window::Window(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("Kestrel");
    setWindowIcon(QIcon(":/logo-mark-256.png"));
    resize(1280, 820);
    QSettings ui;
    dark_ = ui.value("dark", false).toBool();  // light unless the user picked dark (as the web app)

    auto central = new QWidget;
    auto v = new QVBoxLayout(central);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(0);
    v->addWidget(header());
    auto tabs = tabs_ = new QTabWidget;
    tabs->setDocumentMode(true);
    tabs->addTab(tablesPage(), "Tables");
    auto isDark = [this] { return dark_; };
    tabs->addTab(review_ = new ReviewPage(client_, isDark), "Review");
    tabs->addTab(spot_ = new SpotPage(client_, isDark), "Spot study");
    tabs->addTab(trainer_ = new TrainerPage(client_, isDark), "Trainer");
    tabs->addTab(opponents_ = new OpponentsPage(client_), "Opponents");
    tabs->addTab(modelsPage(), "Models");
    tabs->addTab(connectorsPage(), "Connectors");
    tabs->addTab(settingsPage(), "Settings");
    tabs->addTab(logPage(), "Log");
    v->addWidget(tabs, 1);
    setCentralWidget(central);
    applyTheme();

    connect(&client_, &Client::onlineChanged, this, &Window::setOnline);
    connect(&client_, &Client::event, this, &Window::onEvent);
    connect(&client_, &Client::logLine, this, &Window::log);
    setOnline(false);
    QTimer::singleShot(600, this, [this] {
        if (!client_.online()) client_.ensureServer();  // no server yet: start one
    });
}

void Window::showTab(const QString& name) {
    for (int k = 0; k < tabs_->count(); ++k)
        if (tabs_->tabText(k).compare(name, Qt::CaseInsensitive) == 0) tabs_->setCurrentIndex(k);
}

QWidget* Window::header() {
    auto w = new QWidget;
    w->setObjectName("header");
    auto h = new QHBoxLayout(w);
    h->setContentsMargins(18, 10, 18, 10);
    auto logo = new QLabel;
    logo->setPixmap(QPixmap(":/logo-mark-256.png").scaled(30, 30, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    auto title = new QLabel("Kestrel");
    title->setObjectName("title");
    dot_ = new QLabel("●");
    status_ = new QLabel;
    status_->setTextFormat(Qt::PlainText);
    status_->setObjectName("muted");
    serverBtn_ = new QPushButton;
    serverBtn_->setFocusPolicy(Qt::NoFocus);  // never pressed by a stray key
    connect(serverBtn_, &QPushButton::clicked, this, [this] {
        if (!client_.online()) return client_.ensureServer();
        int playing = 0;
        for (const auto& t : tables_) playing += t.value("status").toString() == "playing";
        if (playing && !ask(this, QString("%1 table(s) are playing. Stopping Kestrel stops them. Stop?").arg(playing))) return;
        client_.stopServer();
    });
    auto web = new QPushButton("Open web app");
    connect(web, &QPushButton::clicked, this, [this] { QDesktopServices::openUrl(QUrl(client_.base() + "/")); });
    auto theme = new QPushButton("◐");
    theme->setToolTip("Light or dark");
    connect(theme, &QPushButton::clicked, this, [this] {
        dark_ = !dark_;
        QSettings().setValue("dark", dark_);
        applyTheme();
    });
    h->addWidget(logo);
    h->addWidget(title);
    h->addSpacing(16);
    h->addWidget(dot_);
    h->addWidget(status_);
    h->addStretch();
    h->addWidget(serverBtn_);
    h->addWidget(web);
    h->addWidget(theme);
    return w;
}

void Window::applyTheme() {
    qApp->setStyleSheet(kestrel_app::styleSheet(dark_));
    setOnline(client_.online());
}

void Window::setOnline(bool online) {
    if (!dot_) return;
    const Palette& p = kestrel_app::palette(dark_);
    dot_->setStyleSheet(QString("color: %1; font-size: 14px;").arg(online ? p.win : p.loss));
    status_->setText(online ? QString("Kestrel is running on %1%2").arg(client_.base(), client_.owned() ? " (started by this app)" : "")
                            : "Kestrel is not running");
    serverBtn_->setText(online ? "Stop Kestrel" : "Start Kestrel");
    if (online) {
        refreshTables();
        refreshModels();
        refreshConnectors();
        loadSettings();
        opponents_->reload();
    }
}

void Window::log(const QString& line) {
    if (log_) log_->appendPlainText(line);
}

// ---- Tables: every table, live ----

QWidget* Window::tablesPage() {
    tableList_ = new QListWidget;
    tableList_->setMinimumWidth(260);
    connect(tableList_, &QListWidget::currentRowChanged, this, [this] { showTable(); });
    tableView_ = new QTextBrowser;
    handLog_ = new QTextBrowser;
    drawn_ = new TableView;

    auto newT = new QPushButton("New table");
    newT->setObjectName("primary");
    connect(newT, &QPushButton::clicked, this, &Window::newTable);
    auto newP = new QPushButton("Practice tournament");
    connect(newP, &QPushButton::clicked, this, &Window::newPractice);
    auto stop = new QPushButton("Stop table");
    connect(stop, &QPushButton::clicked, this, [this] {
        auto it = tableList_->currentItem();
        if (!it) return;
        int id = it->data(Qt::UserRole).toInt();
        client_.del(QString("/v1/sessions/%1").arg(id), [this](const QJsonDocument&, int, const QString& err) {
            if (!err.isEmpty()) warn(this, err);
            refreshTables();
        });
    });
    auto screenBtn = new QPushButton("Screen"), gridBtn = new QPushButton("Grid");
    connect(screenBtn, &QPushButton::clicked, this, [this] { showGrid(false); });
    connect(gridBtn, &QPushButton::clicked, this, [this] { showGrid(true); });

    // your move at a practice table where you play a seat
    moves_ = new QWidget;
    advice_ = new QCheckBox("Kestrel's advice");
    connect(advice_, &QCheckBox::toggled, this, [this] { showTable(); });

    auto left = new QWidget;
    auto lv = new QVBoxLayout(left);
    lv->setContentsMargins(0, 0, 0, 0);
    auto buttons = new QHBoxLayout;
    buttons->addWidget(newT);
    buttons->addWidget(newP);
    lv->addLayout(buttons);
    lv->addWidget(card(tableList_, "Tables"), 1);

    // one table: drawn, with this hand's log and Kestrel's decision beside it
    auto screen = new QWidget;
    auto sv = new QVBoxLayout(screen);
    sv->setContentsMargins(0, 0, 0, 0);
    sv->setSpacing(12);
    auto felt = card(drawn_);
    sv->addWidget(felt, 3);
    sv->addWidget(moves_);
    auto lower = new QHBoxLayout;
    lower->addWidget(card(handLog_, "This hand"), 1);
    lower->addWidget(card(tableView_, "Kestrel's decision"), 1);
    sv->addLayout(lower, 2);

    // every table at once
    auto gridHost = new QWidget;
    grid_ = new QGridLayout(gridHost);
    grid_->setSpacing(12);
    auto gridScroll = new QScrollArea;
    gridScroll->setWidget(gridHost);
    gridScroll->setWidgetResizable(true);
    gridScroll->setFrameShape(QFrame::NoFrame);

    views_ = new QStackedWidget;
    views_->addWidget(screen);
    views_->addWidget(gridScroll);

    auto right = new QWidget;
    auto rv = new QVBoxLayout(right);
    rv->setContentsMargins(0, 0, 0, 0);
    auto rb = new QHBoxLayout;
    rb->addWidget(screenBtn);
    rb->addWidget(gridBtn);
    rb->addStretch();
    rb->addWidget(stop);
    rv->addLayout(rb);
    rv->addWidget(views_, 1);

    auto split = new QSplitter;
    split->addWidget(left);
    split->addWidget(right);
    split->setStretchFactor(1, 1);
    split->setSizes({280, 1000});
    auto l = new QVBoxLayout;
    l->addWidget(split);

    // the web app's keys: 1-9 a table, g the grid, n a new table, t the theme
    for (int k = 1; k <= 9; ++k)
        connect(new QShortcut(QKeySequence(QString::number(k)), this), &QShortcut::activated, this, [this, k] {
            if (k <= tableList_->count()) tableList_->setCurrentRow(k - 1), showGrid(false);
        });
    connect(new QShortcut(QKeySequence("G"), this), &QShortcut::activated, this, [this] { showGrid(views_->currentIndex() == 0); });
    connect(new QShortcut(QKeySequence("N"), this), &QShortcut::activated, this, &Window::newTable);
    connect(new QShortcut(QKeySequence("T"), this), &QShortcut::activated, this, [this] {
        dark_ = !dark_;
        QSettings().setValue("dark", dark_);
        applyTheme();
    });
    return page(l);
}

void Window::showGrid(bool grid) {
    views_->setCurrentIndex(grid ? 1 : 0);
    if (!grid) return showTable();
    gridViews_.clear();
    while (auto it = grid_->takeAt(0)) {
        if (it->widget()) it->widget()->deleteLater();
        delete it;
    }
    int k = 0;
    const int cols = tables_.size() > 4 ? 3 : 2;
    for (auto it = tables_.cbegin(); it != tables_.cend(); ++it, ++k) {
        const int id = it.key();
        auto tv = new TableView;
        tv->setCompact(true);
        tv->setMinimumSize(380, 240);
        tv->setCursor(Qt::PointingHandCursor);
        tv->setState(state_.value(id), it.value(), dark_);
        gridViews_[id] = tv;
        connect(tv, &TableView::clicked, this, [this, id] { selectTable(id), showGrid(false); });
        auto title = new QLabel;
        title->setTextFormat(Qt::PlainText);
        title->setText(QString("#%1 · %2 · %3 hands").arg(id).arg(
            it.value().value("kind").toString() == "practice" ? "practice " + it.value().value("model").toString()
                                                              : it.value().value("connector").toString())
                                    .arg(it.value().value("hands").toInt()));
        auto box = new QWidget;
        auto bl = new QVBoxLayout(box);
        bl->setContentsMargins(0, 0, 0, 0);
        bl->addWidget(title);
        bl->addWidget(card(tv), 1);
        grid_->addWidget(box, k / cols, k % cols);
    }
}

void Window::selectTable(int id) {
    for (int r = 0; r < tableList_->count(); ++r)
        if (tableList_->item(r)->data(Qt::UserRole).toInt() == id) tableList_->setCurrentRow(r);
}

void Window::act(int slot) {
    auto it = tableList_->currentItem();
    if (!it) return;
    const int id = it->data(Qt::UserRole).toInt();
    state_[id]["type"] = "practice_waiting";  // the buttons go until the next turn
    showTable();
    client_.post(QString("/v1/practice/%1/act").arg(id), {{"slot", slot}}, [this](const QJsonDocument&, int, const QString& err) {
        if (!err.isEmpty()) warn(this, err);
    });
}

void Window::refreshTables() {
    client_.get("/v1/sessions", [this](const QJsonDocument& d, int, const QString& err) {
        if (!err.isEmpty()) return;
        for (const auto& v : d.object().value("sessions").toArray()) {
            QJsonObject t = v.toObject();
            const int id = t.value("id").toInt();
            tables_[id] = t;
            if (!state_.contains(id))  // a table that started before this window: its current state, as the web app loads it
                client_.get(QString("/v1/sessions/%1").arg(id), [this, id](const QJsonDocument& sd, int, const QString& e2) {
                    if (!e2.isEmpty() || state_.contains(id)) return;
                    state_[id] = sd.object().value("state").toObject();
                    if (tableList_->currentItem() && tableList_->currentItem()->data(Qt::UserRole).toInt() == id) showTable();
                });
        }
        const int keep = tableList_->currentItem() ? tableList_->currentItem()->data(Qt::UserRole).toInt() : -1;
        tableList_->clear();
        for (auto it = tables_.cbegin(); it != tables_.cend(); ++it) {
            const QJsonObject& t = it.value();
            const bool practice = t.value("kind").toString() == "practice";
            QString name = practice ? QString("Practice · %1 · %2-max").arg(t.value("model").toString()).arg(t.value("players").toInt(9))
                                    : QString("%1 · %2").arg(t.value("connector").toString(), t.value("model").toString());
            QString line = QString("#%1  %2\n%3 · %4 hands%5")
                               .arg(it.key())
                               .arg(name, t.value("status").toString())
                               .arg(t.value("hands").toInt())
                               .arg(practice ? "" : " · " + sign(t.value("bb100").toDouble()) + " bb/100");
            auto row = new QListWidgetItem(line);
            row->setData(Qt::UserRole, it.key());
            tableList_->addItem(row);
            if (it.key() == keep) tableList_->setCurrentItem(row);
        }
        if (!tableList_->currentItem() && tableList_->count()) tableList_->setCurrentRow(tableList_->count() - 1);
        showTable();
    });
}

void Window::onEvent(const QJsonObject& e) {
    if (!e.contains("table")) return;
    const int id = e.value("table").toInt();
    const QString type = e.value("type").toString();
    if (type == "table") return refreshTables();
    state_[id] = e;
    if (views_->currentIndex() == 1) {
        if (auto* tv = gridViews_.value(id)) tv->setState(e, tables_.value(id), dark_);
        else showGrid(true);  // a new table
    }
    if (type == "hand" || type == "practice_hand" || type == "tournament") {
        QJsonObject& t = tables_[id];
        if (e.contains("hands")) t["hands"] = e.value("hands");
        if (e.contains("bb100")) t["bb100"] = e.value("bb100");
        refreshTables();  // the list's counts
        return;
    }
    if (tableList_->currentItem() && tableList_->currentItem()->data(Qt::UserRole).toInt() == id) showTable();
}

// the selected table: drawn, its hand so far, and its latest decision with Kestrel's probabilities
void Window::showTable() {
    auto it = tableList_->currentItem();
    const Palette& p = kestrel_app::palette(dark_);
    delete moves_->layout();
    for (auto* w : moves_->findChildren<QWidget*>(Qt::FindDirectChildrenOnly)) if (w != advice_) w->deleteLater();
    moves_->hide();
    if (!it) {
        drawn_->setState({}, {}, dark_);
        handLog_->setHtml("");
        tableView_->setHtml("<p style='color:gray'>No table yet. Start one with <b>New table</b> or a practice tournament.</p>");
        return;
    }
    const int id = it->data(Qt::UserRole).toInt();
    const QJsonObject t = tables_.value(id), e = state_.value(id);
    drawn_->setState(e, t, dark_);

    // the hand so far
    QString log;
    for (const auto& v : e.value("log").toArray()) {
        QJsonObject x = v.toObject();
        log += QString("<div>%1 · %2 %3 <b>%4</b></div>")
                   .arg(kStreetNames[std::clamp(x.value("street").toInt(), 0, 3)], esc(x.value("pos").toString()),
                        esc(x.value("name").toString()), esc(x.value("action").toString()));
    }
    if (log.isEmpty() && e.contains("action")) log = "<div style='font-family:monospace'>" + esc(e.value("action").toString()) + "</div>";
    handLog_->setHtml(log.isEmpty() ? "<p style='color:gray'>–</p>" : log);

    // your move (practice, your seat)
    if (e.value("type").toString() == "practice_your_turn") {
        auto ml = new QHBoxLayout(moves_);
        ml->setContentsMargins(0, 0, 0, 0);
        auto lbl = new QLabel;
        lbl->setTextFormat(Qt::PlainText);
        lbl->setText(QString("Your move · %1 · pot %2 bb").arg(e.value("pos").toString()).arg(e.value("pot_bb").toDouble(), 0, 'f', 1));
        ml->addWidget(lbl);
        double best = 0;
        for (const auto& v : e.value("choices").toArray()) best = std::max(best, v.toObject().value("advice").toDouble());
        for (const auto& v : e.value("choices").toArray()) {
            QJsonObject c = v.toObject();
            QString text = c.value("label").toString();
            if (advice_->isChecked()) text += QString("  %1%").arg(int(c.value("advice").toDouble() * 100 + 0.5));
            auto b = new QPushButton(text);
            if (c.value("slot").toInt() != 0) b->setObjectName("primary");
            if (advice_->isChecked() && c.value("advice").toDouble() == best) b->setStyleSheet(QString("border: 2px solid %1;").arg(p.accent2));
            const int slot = c.value("slot").toInt();
            connect(b, &QPushButton::clicked, this, [this, slot] { act(slot); });
            ml->addWidget(b);
        }
        ml->addStretch();
        ml->addWidget(advice_);
        moves_->show();
    }

    // the session, and the last decision
    QString h = QString("<p style='margin:0'><b>#%1 %2</b> <span style='color:%3'>· %4 · %5 + %6 · %7 hands</span></p>")
                    .arg(id)
                    .arg(esc(t.value("connector").toString(t.value("kind").toString())), p.muted,
                         esc(t.value("status").toString()), esc(t.value("model").toString()),
                         esc(t.value("solver").toString("none")))
                    .arg(t.value("hands").toInt());
    if (t.contains("bb100") && t.value("kind").toString() != "practice")
        h += QString("<p style='margin:4px 0'><b style='font-size:18px;color:%1'>%2 bb/100</b></p>")
                 .arg(t.value("bb100").toDouble() >= 0 ? p.win : p.loss, sign(t.value("bb100").toDouble()));
    const QJsonArray opts = e.value("options").toArray();
    if (!opts.isEmpty() && !e.value("hidden").toBool()) {
        h += QString("<p style='color:%1;margin:8px 0 2px'>%2%3</p>")
                 .arg(p.muted, esc(e.value("name").toString(e.value("pos").toString())),
                      e.value("pos").isString() && e.contains("name") ? " · " + esc(e.value("pos").toString()) : "");
        h += "<table cellspacing=5 width='100%'>";
        for (const auto& v : opts) {
            const QJsonObject o = v.toObject();
            const double pr = o.value("prob").toDouble();
            const QString label = o.value("label").toString(o.value("kind").toString(o.value("action").toString()));
            const bool chosen = o.value("chosen").toBool() || (e.contains("chose") && o.value("action").toString() == e.value("chose").toString());
            h += QString("<tr><td width='34%'>%1</td><td><table cellspacing=0 cellpadding=0 width='100%'><tr>"
                         "<td width='%2%' style='background:%3;height:8px'></td><td style='background:%4'></td></tr></table></td>"
                         "<td width='12%' align='right' style='font-family:monospace'>%5%</td></tr>")
                     .arg(chosen ? "<b>" + esc(label) + "</b>" : esc(label))
                     .arg(std::max(1, int(pr * 100)))
                     .arg(chosen ? p.accent : p.accent2, p.line)
                     .arg(int(pr * 100 + 0.5));
        }
        h += "</table>";
        if (e.contains("value_bb")) h += QString("<p style='color:%1'>value %2 bb</p>").arg(p.muted, sign(e.value("value_bb").toDouble()));
    } else if (e.value("hidden").toBool()) {
        h += "<p style='color:gray'>You play this table: Kestrel's probabilities stay hidden.</p>";
    }
    tableView_->setHtml(h);
}


void Window::newTable() {
    QDialog d(this);
    d.setWindowTitle("New table");
    auto f = new QFormLayout(&d);
    auto conn = new QComboBox, model = new QComboBox, solver = new QComboBox;
    auto hands = new QSpinBox, tours = new QSpinBox;
    hands->setRange(1, 1000000), hands->setValue(200);
    tours->setRange(1, 10000), tours->setValue(10);
    for (const auto& v : connectors_) conn->addItem(v.toObject().value("name").toString());
    auto fill = [&, this] {
        int players = 2;
        for (const auto& v : connectors_)
            if (v.toObject().value("name").toString() == conn->currentText()) players = v.toObject().value("players").toInt(2);
        model->clear(), solver->clear();
        model->addItems(names(engines_models_, [&](const QJsonObject& m) {
            return m.value("available").toBool() && m.value("players").toInt() == (players > 2 ? 9 : 2);
        }));
        solver->addItems(names(engines_solvers_, [&](const QJsonObject& s) {  // 9-max solvers say players 9
            return s.value("available").toBool() &&
                   (s.value("name").toString() == "none" || s.value("players").toInt(2) == (players > 2 ? 9 : 2));
        }));
        tours->setEnabled(players > 2);
    };
    connect(conn, &QComboBox::currentTextChanged, &d, fill);
    fill();
    f->addRow("Connector", conn);
    f->addRow("Model", model);
    f->addRow("Solver", solver);
    f->addRow("Hands", hands);
    f->addRow("Tournaments", tours);
    auto bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    f->addRow(bb);
    connect(bb, &QDialogButtonBox::accepted, &d, &QDialog::accept);
    connect(bb, &QDialogButtonBox::rejected, &d, &QDialog::reject);
    if (d.exec() != QDialog::Accepted) return;
    QJsonObject body{{"connector", conn->currentText()}, {"model", model->currentText()}, {"solver", solver->currentText()},
                     {"hands", hands->value()}, {"tournaments", tours->value()}};
    client_.post("/v1/sessions", body, [this](const QJsonDocument&, int, const QString& err) {
        if (!err.isEmpty()) warn(this, err);
        refreshTables();
    });
}

void Window::newPractice() {
    QDialog d(this);
    d.setWindowTitle("Practice tournament");
    auto f = new QFormLayout(&d);
    auto model = new QComboBox;
    model->addItems(names(engines_models_, [](const QJsonObject& m) { return m.value("practice").toBool(); }));
    auto players = new QSpinBox, hands = new QSpinBox, pace = new QSpinBox;
    players->setRange(2, 9), players->setValue(9);
    hands->setRange(1, 100000), hands->setValue(300);
    pace->setRange(0, 5000), pace->setValue(1800), pace->setSingleStep(500), pace->setSuffix(" ms a move");
    auto avg = new QSpinBox;
    avg->setRange(5, 60), avg->setValue(20), avg->setSuffix(" bb");
    auto human = new QCheckBox("Play a seat yourself (in the web app)");
    f->addRow("Model", model);
    f->addRow("Players", players);
    f->addRow("Average stack", avg);
    f->addRow("Hands", hands);
    f->addRow("Pace", pace);
    f->addRow("", human);
    auto bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    f->addRow(bb);
    connect(bb, &QDialogButtonBox::accepted, &d, &QDialog::accept);
    connect(bb, &QDialogButtonBox::rejected, &d, &QDialog::reject);
    if (d.exec() != QDialog::Accepted) return;
    QJsonObject body{{"model", model->currentText()}, {"players", players->value()}, {"avg_bb", avg->value()},
                     {"hands", hands->value()}, {"pace_ms", pace->value()}, {"human", human->isChecked()}};
    client_.post("/v1/practice", body, [this, h = human->isChecked()](const QJsonDocument&, int, const QString& err) {
        if (!err.isEmpty()) warn(this, err);
        else if (h) QDesktopServices::openUrl(QUrl(client_.base() + "/"));
        refreshTables();
    });
}

// ---- Models ----

QWidget* Window::modelsPage() {
    modelTable_ = new QTableWidget(0, 4);
    modelTable_->setHorizontalHeaderLabels({"Model", "Game", "Players", "About"});
    modelTable_->horizontalHeader()->setStretchLastSection(true);
    modelTable_->verticalHeader()->hide();
    solverTable_ = new QTableWidget(0, 3);
    solverTable_->setHorizontalHeaderLabels({"Solver", "Available", "About"});
    solverTable_->horizontalHeader()->setStretchLastSection(true);
    solverTable_->verticalHeader()->hide();
    auto install = new QPushButton("Install a model package…");
    install->setObjectName("primary");
    connect(install, &QPushButton::clicked, this, [this] {
        QString f = QFileDialog::getOpenFileName(this, "Model package", QDir::homePath(), "Model packages (*.tar.gz)");
        if (f.isEmpty()) return;
        if (!QFileInfo::exists(f + ".sha256") &&
            !ask(this, "There is no .sha256 beside the package to check it against. Install anyway?"))
            return;
        runKestreld({"models", "install", f}, [this](int code, const QString& out) {
            if (code == 0) info(this, out.trimmed());
            else warn(this, out.trimmed());
            refreshModels();
        });
    });
    auto folder = new QPushButton("Open the models folder");
    connect(folder, &QPushButton::clicked, this, [] {
        QDir().mkpath(modelsDir());
        QDesktopServices::openUrl(QUrl::fromLocalFile(modelsDir()));
    });
    auto b = new QHBoxLayout;
    b->addWidget(install);
    b->addWidget(folder);
    b->addStretch();
    auto l = new QVBoxLayout;
    l->setSpacing(14);
    l->addLayout(b);
    l->addWidget(card(modelTable_, "Models"), 2);
    l->addWidget(card(solverTable_, "Solvers"), 1);
    return page(l);
}

void Window::refreshModels() {
    client_.get("/v1/engines", [this](const QJsonDocument& d, int, const QString& err) {
        if (!err.isEmpty()) return;
        engines_models_ = d.object().value("models").toArray();
        engines_solvers_ = d.object().value("solvers").toArray();
        modelTable_->setRowCount(0);
        for (const auto& v : engines_models_) {
            QJsonObject m = v.toObject();
            int r = modelTable_->rowCount();
            modelTable_->insertRow(r);
            modelTable_->setItem(r, 0, item(m.value("name").toString()));
            modelTable_->setItem(r, 1, item(m.value("game").toString()));
            modelTable_->setItem(r, 2, item(QString::number(m.value("players").toInt())));
            modelTable_->setItem(r, 3, item(m.value("about").toString()));
        }
        solverTable_->setRowCount(0);
        for (const auto& v : engines_solvers_) {
            QJsonObject s = v.toObject();
            int r = solverTable_->rowCount();
            solverTable_->insertRow(r);
            solverTable_->setItem(r, 0, item(s.value("name").toString()));
            solverTable_->setItem(r, 1, item(s.value("available").toBool() ? "yes" : "no"));
            solverTable_->setItem(r, 2, item(s.value("about").toString()));
        }
        modelTable_->resizeColumnsToContents();
        modelTable_->horizontalHeader()->setSectionResizeMode(modelTable_->columnCount() - 1, QHeaderView::Stretch);
        solverTable_->resizeColumnsToContents();
        solverTable_->horizontalHeader()->setSectionResizeMode(solverTable_->columnCount() - 1, QHeaderView::Stretch);
        loadSettings();  // the default-model choices
        const QStringList nine = names(engines_models_, [](const QJsonObject& m) { return m.value("players").toInt() == 9; });
        spot_->setModels(nine);
        trainer_->setModels(nine);
    });
}

// ---- Connectors ----

QWidget* Window::connectorsPage() {
    connTable_ = new QTableWidget(0, 5);
    connTable_->setHorizontalHeaderLabels({"Connector", "Codec", "Players", "Talks to", "File"});
    connTable_->horizontalHeader()->setStretchLastSection(true);
    connTable_->verticalHeader()->hide();
    connTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    auto add = new QPushButton("Add a connector file…");
    add->setObjectName("primary");
    connect(add, &QPushButton::clicked, this, [this] {
        QString f = QFileDialog::getOpenFileName(this, "Connector", QDir::homePath(), "Connectors (*.yaml *.yml *.json)");
        if (f.isEmpty()) return;
        // checked before it is copied: a broken file never reaches the folder
        runKestreld({"check", f}, [this, f](int code, const QString& out) {
            if (code != 0) return void(warn(this, "This connector has a problem:\n\n" + out.trimmed()));
            if (!ask(this, out.trimmed() + "\n\nAdd it?")) return;
            QDir().mkpath(connectorsDir());
            QString to = QDir(connectorsDir()).filePath(QFileInfo(f).fileName());
            if (QFileInfo::exists(to) &&
                !ask(this, QFileInfo(to).fileName() + " is already there. Replace it?"))
                return;
            QFile::remove(to);
            if (!QFile::copy(f, to)) warn(this, "Could not copy it to " + to);
            refreshConnectors();
        });
    });
    auto check = new QPushButton("Check the selected one");
    connect(check, &QPushButton::clicked, this, [this] {
        int r = connTable_->currentRow();
        if (r < 0) return;
        runKestreld({"check", connTable_->item(r, 4)->text()}, [this](int code, const QString& out) {
            if (code == 0) info(this, out.trimmed());
            else warn(this, out.trimmed());
        });
    });
    auto folder = new QPushButton("Open your connectors folder");
    connect(folder, &QPushButton::clicked, this, [] {
        QDir().mkpath(connectorsDir());
        QDesktopServices::openUrl(QUrl::fromLocalFile(connectorsDir()));
    });
    auto reload = new QPushButton("Reload");
    connect(reload, &QPushButton::clicked, this, &Window::refreshConnectors);
    auto b = new QHBoxLayout;
    b->addWidget(add);
    b->addWidget(check);
    b->addWidget(folder);
    b->addStretch();
    b->addWidget(reload);
    auto l = new QVBoxLayout;
    l->setSpacing(14);
    l->addLayout(b);
    l->addWidget(card(connTable_, "Connectors"), 1);
    auto note = new QLabel("A connector names everything it talks to and the KESTREL_* variables it reads; check a file "
                           "before adding it. Each one is described in docs/design.md (Connector files).");
    note->setObjectName("muted");
    note->setWordWrap(true);
    l->addWidget(note);
    return page(l);
}

void Window::refreshConnectors() {
    client_.get("/v1/connectors", [this](const QJsonDocument& d, int, const QString& err) {
        if (!err.isEmpty()) return;
        connectors_ = d.object().value("connectors").toArray();
        connTable_->setRowCount(0);
        for (const auto& v : connectors_) {
            QJsonObject c = v.toObject();
            int r = connTable_->rowCount();
            connTable_->insertRow(r);
            QStringList hosts;
            for (const auto& x : c.value("hosts").toArray()) hosts << x.toString();
            connTable_->setItem(r, 0, item(c.value("name").toString()));
            connTable_->setItem(r, 1, item(c.value("codec").toString()));
            connTable_->setItem(r, 2, item(QString::number(c.value("players").toInt())));
            connTable_->setItem(r, 3, item(hosts.join(", ")));
            connTable_->setItem(r, 4, item(c.value("path").toString()));
        }
        connTable_->resizeColumnsToContents();
        connTable_->horizontalHeader()->setSectionResizeMode(connTable_->columnCount() - 1, QHeaderView::Stretch);
    });
}

// ---- Settings (admin): settings.json, applied by restarting the server ----

QWidget* Window::settingsPage() {
    auto spin = [](int lo, int hi) {
        auto s = new QSpinBox;
        s->setRange(lo, hi);
        return s;
    };
    port_ = spin(1024, 65535), wsPort_ = spin(0, 65535), tcpPort_ = spin(0, 65535);
    resolveIters_ = spin(50, 20000), tsThreads_ = spin(1, 256), searchSamples_ = spin(1024, 4096);
    lan_ = new QCheckBox("Reach Kestrel from other devices on the network (they need the access link)");
    resolveFloor_ = new QCheckBox("Defence floor on re-solved river spots");
    resolveFloor_->setToolTip("On (recommended): facing a big river bet, a re-solved decision never folds a hand in the "
                              "top P/(P+B) of its range (pot P, bet B). Off: the re-solve alone.");
    defaultModel_ = new QComboBox;
    tsBin_ = new QLineEdit, tsRes_ = new QLineEdit, sessionsDir_ = new QLineEdit, packDir_ = new QLineEdit, dbPath_ = new QLineEdit;
    for (auto* e : {tsBin_, tsRes_, sessionsDir_, packDir_, dbPath_}) e->setPlaceholderText("default");
    auto browse = [this](QLineEdit* e, bool dir) {
        auto row = new QWidget;
        auto h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        auto b = new QPushButton("Browse…");
        connect(b, &QPushButton::clicked, this, [this, e, dir] {
            QString p = dir ? QFileDialog::getExistingDirectory(this, "Folder", e->text().isEmpty() ? QDir::homePath() : e->text())
                            : QFileDialog::getOpenFileName(this, "Program", e->text().isEmpty() ? QDir::homePath() : e->text());
            if (!p.isEmpty()) e->setText(p);
        });
        h->addWidget(e, 1);
        h->addWidget(b);
        return row;
    };
    auto server = new QWidget;
    auto fs = new QFormLayout(server);
    fs->addRow("Port (web app and REST)", port_);
    fs->addRow("WebSocket port (0: off)", wsPort_);
    fs->addRow("TCP port (0: off)", tcpPort_);
    fs->addRow("", lan_);
    lanLinks_ = new QLabel;
    lanLinks_->setTextFormat(Qt::PlainText);
    lanLinks_->setWordWrap(true);
    lanLinks_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    fs->addRow("Access links", lanLinks_);
    auto rotate = new QPushButton("New access token");
    rotate->setToolTip("Every device has to open the new link again");
    connect(rotate, &QPushButton::clicked, this, [this] {
        if (!ask(this, "Make a new access token? Devices on the network will need the new link."))
            return;
        QFile::remove(configDir() + "/token");
        client_.restartServer();
    });
    fs->addRow("", rotate);

    auto play = new QWidget;
    auto fp = new QFormLayout(play);
    fp->addRow("Default 9-max model", defaultModel_);
    fp->addRow("River re-solve iterations", resolveIters_);
    fp->addRow("", resolveFloor_);
    fp->addRow("9-max search deals per decision", searchSamples_);
    fp->addRow("TexasSolver program", browse(tsBin_, false));
    fp->addRow("TexasSolver resources", browse(tsRes_, true));
    fp->addRow("TexasSolver threads", tsThreads_);

    auto files = new QWidget;
    auto ff = new QFormLayout(files);
    ff->addRow("Sessions folder", browse(sessionsDir_, true));
    ff->addRow("Database (kestrel.db)", browse(dbPath_, false));
    ff->addRow("Heads-up data pack", browse(packDir_, true));

    settingsNote_ = new QLabel;
    settingsNote_->setObjectName("muted");
    settingsNote_->setWordWrap(true);
    auto save = new QPushButton("Save and restart Kestrel");
    save->setObjectName("primary");
    connect(save, &QPushButton::clicked, this, &Window::saveSettings);
    auto revert = new QPushButton("Revert");
    connect(revert, &QPushButton::clicked, this, &Window::loadSettings);
    auto b = new QHBoxLayout;
    b->addWidget(save);
    b->addWidget(revert);
    b->addStretch();

    auto inner = new QWidget;
    auto l = new QVBoxLayout(inner);
    l->setSpacing(14);
    l->addWidget(card(server, "Server and network"));
    l->addWidget(card(play, "Play"));
    l->addWidget(card(files, "Folders"));
    l->addLayout(b);
    l->addWidget(settingsNote_);
    l->addStretch();
    auto scroll = new QScrollArea;
    scroll->setWidget(inner);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    auto outer = new QVBoxLayout;
    outer->addWidget(scroll);
    loadSettings();
    return page(outer);
}

void Window::loadSettings() {
    if (!port_) return;
    const QJsonObject s = readSettings();
    port_->setValue(s.value("port").toInt(8790));
    wsPort_->setValue(s.value("ws_port").toInt(8791));
    tcpPort_->setValue(s.value("tcp_port").toInt(8792));
    lan_->setChecked(s.value("lan").toBool(true));
    resolveIters_->setValue(s.value("resolve_iters").toInt(500));
    searchSamples_->setValue(s.value("search_samples").toInt(1024));
    resolveFloor_->setChecked(s.value("resolve_floor").toBool(true));
    tsThreads_->setValue(s.value("texassolver_threads").toInt(8));
    tsBin_->setText(s.value("texassolver_bin").toString());
    tsRes_->setText(s.value("texassolver_resources").toString());
    sessionsDir_->setText(s.value("sessions_dir").toString());
    dbPath_->setText(s.value("db").toString());
    packDir_->setText(s.value("pack").toString());
    const QString want = s.value("default_model").toString("nm25");
    defaultModel_->clear();
    QStringList nine = names(engines_models_, [](const QJsonObject& m) { return m.value("players").toInt() == 9; });
    if (!nine.contains(want)) nine.prepend(want);
    defaultModel_->addItems(nine);
    defaultModel_->setCurrentText(want);
    settingsNote_->setText("Saved in " + settingsPath() + ". Changes apply when Kestrel restarts.");
    if (client_.online())
        client_.get("/v1/network", [this](const QJsonDocument& d, int, const QString&) {
            QStringList links;
            for (const auto& v : d.object().value("links").toArray()) links << v.toString();
            lanLinks_->setText(links.isEmpty() ? "this computer only" : links.join("\n"));
        });
}

void Window::saveSettings() {
    QJsonObject s = readSettings();  // keys this page does not show are kept
    s["port"] = port_->value();
    s["ws_port"] = wsPort_->value();
    s["tcp_port"] = tcpPort_->value();
    s["lan"] = lan_->isChecked();
    s["default_model"] = defaultModel_->currentText();
    s["resolve_iters"] = resolveIters_->value();
    s["search_samples"] = searchSamples_->value();
    s["resolve_floor"] = resolveFloor_->isChecked();
    s["texassolver_threads"] = tsThreads_->value();
    auto put = [&](const char* k, QLineEdit* e) {
        if (e->text().trimmed().isEmpty()) s.remove(k);
        else s[k] = e->text().trimmed();
    };
    put("texassolver_bin", tsBin_), put("texassolver_resources", tsRes_), put("sessions_dir", sessionsDir_), put("pack", packDir_), put("db", dbPath_);
    if (lan_->isChecked() && !readSettings().value("lan").toBool(true) &&
        !ask(this, "Other devices on your network will be able to reach Kestrel with the access link. Continue?"))
        return;
    QString err;
    if (!writeSettings(s, &err)) return void(warn(this, "Could not save: " + err));
    log("settings saved: " + settingsPath());
    if (client_.online()) {
        int playing = 0;
        for (const auto& t : tables_) playing += t.value("status").toString() == "playing";
        if (playing && !ask(this, QString("%1 table(s) are playing. Restarting stops them. Restart now?").arg(playing)))
            return;
        client_.restartServer();
    } else {
        client_.ensureServer();
    }
}

// ---- Log ----

QWidget* Window::logPage() {
    log_ = new QPlainTextEdit;
    log_->setReadOnly(true);
    log_->setMaximumBlockCount(5000);
    log_->setStyleSheet("font-family: 'JetBrains Mono', monospace; font-size: 12px;");
    auto l = new QVBoxLayout;
    l->addWidget(card(log_, "Kestrel's output"));
    return page(l);
}

// kestreld for the commands with no REST route (models install, check): its output when it ends
void Window::runKestreld(const QStringList& args, std::function<void(int, QString)> done) {
    auto p = new QProcess(this);
    p->setProcessChannelMode(QProcess::MergedChannels);
    connect(p, &QProcess::finished, this, [p, done](int code, QProcess::ExitStatus) {
        done(code, QString::fromUtf8(p->readAll()));
        p->deleteLater();
    });
    connect(p, &QProcess::errorOccurred, this, [p, done](QProcess::ProcessError) {
        if (p->state() == QProcess::NotRunning) done(-1, "could not run kestreld: " + p->errorString());
    });
    log("kestreld " + args.join(' '));
    p->start(kestreldPath(), args);
}

}  // namespace kestrel_app
