#include "study.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QFileInfo>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFormLayout>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QSpinBox>
#include <QSplitter>
#include <QStackedWidget>
#include <QHeaderView>
#include <QTableWidget>
#include <QTextBrowser>
#include <QUrl>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>
#include <memory>

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
    if (qobject_cast<QLabel*>(inner)) l->addStretch(1);  // a text card keeps its text at the top
    return f;
}

QLabel* plain(const QString& text = {}) {
    auto l = new QLabel(text);
    l->setTextFormat(Qt::PlainText);
    l->setWordWrap(true);
    return l;
}

QString esc(const QString& s) { return s.toHtmlEscaped(); }
QString sign(double x) { return (x >= 0 ? "+" : "") + QString::number(x, 'f', 1); }

QStringList cardsOf(const QString& text) {
    QString t = text;
    t.remove(' ').remove(',');
    QStringList out;
    for (int k = 0; k + 1 < t.size(); k += 2) out << t.mid(k, 2);
    return out;
}

QJsonArray toArray(const QStringList& l) {
    QJsonArray a;
    for (const auto& x : l) a.append(x);
    return a;
}

// Kestrel's options as probability bars; chosen (by you) and the model's favourite are marked
QString bars(const QJsonArray& opts, const Palette& p, int chosen = -1) {
    double best = 0;
    for (const auto& v : opts) best = std::max(best, v.toObject().value("prob").toDouble());
    QString h = "<table cellspacing=5 width='100%'>";
    for (int k = 0; k < opts.size(); ++k) {
        const QJsonObject o = opts[k].toObject();
        const double pr = o.value("prob").toDouble();
        QString label = o.value("label").toString(o.value("action").toString());
        if (label.isEmpty()) label = o.value("action").toString();
        if (o.contains("to") && o.value("action").toString() == "raise" && !o.contains("label"))
            label += " to " + QString::number(o.value("to").toDouble(), 'f', 0);
        const bool top = pr == best;
        QString name = esc(label);
        if (k == chosen) name = "<b>" + name + " ← you</b>";
        else if (top) name = "<b>" + name + "</b>";
        h += QString("<tr><td width='36%'>%1</td><td><table cellspacing=0 cellpadding=0 width='100%'><tr>"
                     "<td width='%2%' style='background:%3;height:8px'></td><td style='background:%4'></td></tr></table></td>"
                     "<td width='12%' align='right' style='font-family:monospace'>%5%</td></tr>")
                 .arg(name)
                 .arg(std::max(1, int(pr * 100)))
                 .arg(top ? p.accent : p.accent2, p.line)
                 .arg(QString::number(pr * 100, 'f', pr < 0.1 && pr > 0 ? 1 : 0));
    }
    return h + "</table>";
}

}  // namespace

QString sessionsDir() {
    QString s = readSettings().value("sessions_dir").toString();
    if (!s.isEmpty()) return s;
    QString x = qEnvironmentVariable("XDG_DATA_HOME");
    QString home = qEnvironmentVariable("HOME", qEnvironmentVariable("USERPROFILE", "."));
    return (x.isEmpty() ? home + "/.local/share" : x) + "/kestrel/sessions";
}

void equityTable(Client& c, const QJsonArray& players, const QJsonArray& payouts, const QString& me, bool dark,
                 std::function<void(const QString&)> done) {
    QJsonArray stacks;
    for (const auto& v : players) stacks.append(v.toObject().value("stack").toDouble());
    auto mh = std::make_shared<QJsonArray>(), mw = std::make_shared<QJsonArray>();
    auto finish = [=] {
        if (mh->isEmpty() || mw->isEmpty()) return;
        const Palette& p = palette(dark);
        QString h = QString("<table cellspacing=4><tr style='color:%1'><td>player</td><td align='right'>chips</td>"
                            "<td align='right'>ICM (Malmuth-Harville)</td><td align='right'>Malmuth-Weitzman</td></tr>")
                        .arg(p.muted);
        for (int k = 0; k < players.size(); ++k) {
            const QString id = players[k].toObject().value("id").toString();
            const double a = (*mh)[k].toDouble() * 100, b = (*mw)[k].toDouble() * 100;
            h += QString("<tr><td>%1</td><td align='right'>%2</td><td align='right'>%3%</td><td align='right'>%4%</td></tr>")
                     .arg(id == me ? "<b>" + esc(id) + " (you)</b>" : esc(id))
                     .arg(stacks[k].toDouble(), 0, 'f', 0)
                     .arg(a, 0, 'f', 1)
                     .arg(b, 0, 'f', 1);
        }
        done(h + "</table>");
    };
    for (auto [model, out] : {std::pair{QString("mh"), mh}, std::pair{QString("mw"), mw}})
        c.post("/v1/icm", {{"stacks", stacks}, {"payouts", payouts}, {"model", model}},
               [out, finish](const QJsonDocument& d, int, const QString& err) {
                   if (!err.isEmpty()) return;
                   *out = d.object().value("equity").toArray();
                   finish();
               });
}

// ---- Review: the hands of finished and running sessions, from their logs ----

ReviewPage::ReviewPage(Client& c, std::function<bool()> dark, QWidget* parent) : QWidget(parent), client_(c), dark_(dark) {
    setObjectName("page");
    dir_ = sessionsDir();
    sessions_ = new QListWidget;
    hands_ = new QListWidget;
    summary_ = plain();
    detail_ = new QTextBrowser;
    drawn_ = new TableView;
    drawn_->setMinimumHeight(260);
    connect(sessions_, &QListWidget::currentRowChanged, this, [this] { openSession(); });
    connect(hands_, &QListWidget::currentRowChanged, this, [this] { showHand(); });

    auto pick = new QPushButton("Other folder…");
    connect(pick, &QPushButton::clicked, this, [this] {
        QString d = QFileDialog::getExistingDirectory(this, "Sessions folder", dir_);
        if (!d.isEmpty()) dir_ = d, reload();
    });
    auto refresh = new QPushButton("Reload");
    connect(refresh, &QPushButton::clicked, this, &ReviewPage::reload);
    auto luck = new QPushButton("Luck-adjusted result");
    luck->setToolTip("AIVAT-style: removes the luck of the cards from a heads-up session's result");
    connect(luck, &QPushButton::clicked, this, [this] {
        auto it = sessions_->currentItem();
        if (!it) return;
        auto p = new QProcess(this);
        p->setProcessChannelMode(QProcess::MergedChannels);
        connect(p, &QProcess::finished, this, [this, p](int, QProcess::ExitStatus) {
            detail_->setPlainText(QString::fromUtf8(p->readAll()));
            p->deleteLater();
        });
        p->start(kestreldPath(), {"luck", it->data(Qt::UserRole).toString()});
    });

    auto top = new QHBoxLayout;
    top->addWidget(plain("Sessions in " + dir_), 1);
    top->addWidget(luck);
    top->addWidget(pick);
    top->addWidget(refresh);
    auto right = new QWidget;
    auto rv = new QVBoxLayout(right);
    rv->setContentsMargins(0, 0, 0, 0);
    rv->addWidget(card(drawn_), 3);
    rv->addWidget(card(detail_, "This hand"), 2);
    auto split = new QSplitter;
    split->addWidget(card(sessions_, "Sessions"));
    auto mid = new QWidget;
    auto mv = new QVBoxLayout(mid);
    mv->setContentsMargins(0, 0, 0, 0);
    mv->addWidget(summary_);
    mv->addWidget(card(hands_, "Hands"), 1);
    split->addWidget(mid);
    split->addWidget(right);
    split->setSizes({260, 360, 700});
    auto l = new QVBoxLayout(this);
    l->setContentsMargins(20, 18, 20, 18);
    l->addLayout(top);
    l->addWidget(split, 1);
    reload();
}

void ReviewPage::reload() {
    sessions_->clear();
    QDir d(dir_);
    auto entries = d.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Time);
    for (const auto& e : entries) {
        if (!QFile::exists(e.filePath() + "/hands.jsonl")) continue;
        auto it = new QListWidgetItem(e.fileName() + "\n" + e.lastModified().toString("yyyy-MM-dd HH:mm"));
        it->setData(Qt::UserRole, e.filePath());
        sessions_->addItem(it);
    }
    if (sessions_->count()) sessions_->setCurrentRow(0);
    else summary_->setText("No session logs in " + dir_ + " yet.");
}

void ReviewPage::openSession() {
    hands_->clear();
    lines_ = {};
    auto it = sessions_->currentItem();
    if (!it) return;
    QFile f(it->data(Qt::UserRole).toString() + "/hands.jsonl");
    if (!f.open(QIODevice::ReadOnly)) return;
    double total = 0, sq = 0;
    int n = 0, practice = 0;
    while (!f.atEnd()) {
        QJsonObject h = QJsonDocument::fromJson(f.readLine()).object();
        if (h.isEmpty()) continue;
        lines_.append(h);
        QString hole, label;
        double r = 0;
        bool scored = h.contains("result_bb");
        if (h.contains("results")) {  // a practice hand: your seat's result, or the winner's
            ++practice;
            QJsonObject mine;
            for (const auto& v : h.value("results").toArray())
                if (v.toObject().value("you").toBool()) mine = v.toObject();
            if (!mine.isEmpty()) {
                r = mine.value("result_bb").toDouble(), scored = true;
                for (const auto& c : mine.value("hole").toArray()) hole += c.toString();
            }
            label = QString::number(h.value("log").toArray().size()) + " actions";
        } else {
            r = h.value("result_bb").toDouble();
            for (const auto& c : h.value("hole").toArray()) hole += c.toString() + " ";
            label = h.value("action").toString();
        }
        if (scored) total += r, sq += r * r, ++n;
        auto row = new QListWidgetItem(QString("#%1  %2  %3%4")
                                           .arg(lines_.size())
                                           .arg(hole.trimmed(), label.left(40))
                                           .arg(scored ? "   " + sign(r) + " bb" : QString()));
        hands_->addItem(row);
    }
    if (n) {
        double m = total / n, sd = std::sqrt(std::max(0.0, sq / n - m * m)), half = 1.96 * sd / std::sqrt(double(n));
        summary_->setText(QString("%1 hands · %2 bb · %3 bb/100 [%4, %5]")
                              .arg(n)
                              .arg(sign(total), sign(m * 100), sign((m - half) * 100), sign((m + half) * 100)));
    } else {
        summary_->setText(QString("%1 hands%2").arg(lines_.size()).arg(practice ? " (practice, Kestrel in every seat)" : ""));
    }
    if (hands_->count()) hands_->setCurrentRow(hands_->count() - 1);
}

void ReviewPage::showHand() {
    const int k = hands_->currentRow();
    if (k < 0 || k >= lines_.size()) return;
    QJsonObject h = lines_[k].toObject();
    const Palette& p = kestrel_app::palette(dark_());
    QString d;
    if (h.contains("results")) {  // practice: draw it as the table showed it at the end
        h["type"] = "practice_hand";
        h["showdown"] = h.value("board").toArray().size() == 5;
        for (const auto& v : h.value("log").toArray()) {
            QJsonObject x = v.toObject();
            d += QString("<div>%1 · %2 %3 <b>%4</b></div>")
                     .arg(x.value("street").toInt() == 0 ? "Pre-flop" : x.value("street").toInt() == 1 ? "Flop" : x.value("street").toInt() == 2 ? "Turn" : "River",
                          esc(x.value("pos").toString()), esc(x.value("name").toString()), esc(x.value("action").toString()));
        }
    } else {
        h["type"] = "hand";
        d = QString("<p style='font-family:monospace'>%1</p>").arg(esc(h.value("action").toString()));
        if (h.contains("decisions")) {
            for (const auto& v : h.value("decisions").toArray())
                d += QString("<div>street %1: %2</div>").arg(v.toObject().value("street").toInt()).arg(esc(v.toObject().value("chose").toString()));
        }
        d += QString("<p>result <b style='color:%1'>%2 bb</b></p>")
                 .arg(h.value("result_bb").toDouble() >= 0 ? p.win : p.loss, sign(h.value("result_bb").toDouble()));
        if (h.contains("model")) d += QString("<p style='color:%1'>%2 + %3</p>").arg(p.muted, esc(h.value("model").toString()), esc(h.value("solver").toString("none")));
    }
    detail_->setHtml(d);
    drawn_->setState(h, {}, dark_());
}

// ---- Spot study: describe a spot, see Kestrel's advice ----

SpotPage::SpotPage(Client& c, std::function<bool()> dark, QWidget* parent) : QWidget(parent), client_(c), dark_(dark) {
    setObjectName("page");
    game_ = new QComboBox;
    game_->addItems({"2-9 handed table (9-max model)", "Heads-up (blueprint)"});
    model_ = new QComboBox;
    solver_ = new QComboBox;
    solver_->addItems({"none", "kestrel-river", "texassolver"});

    // a table
    table_ = new QWidget;
    auto ft = new QFormLayout(table_);
    stacks_ = new QLineEdit("1500, 900, 2200, 1800, 1200, 2600");
    stacks_->setToolTip("Chips at the start of the hand, in seat order (clockwise); players are P1, P2, ...");
    auto spin = [](int lo, int hi, int v) {
        auto s = new QSpinBox;
        s->setRange(lo, hi), s->setValue(v);
        return s;
    };
    button_ = spin(1, 9, 1), me_ = spin(1, 9, 2), sb_ = spin(1, 1000000, 25), bb_ = spin(1, 1000000, 50), ante_ = spin(0, 1000000, 50);
    hole_ = new QLineEdit("Ah Kd");
    board_ = new QLineEdit;
    board_->setPlaceholderText("e.g. 7d Ts 2c");
    payouts_ = new QLineEdit("0.5, 0.3, 0.2");
    payouts_->setToolTip("Prize shares of the places still to be paid, first place first; empty: chips");
    actions_ = new QPlainTextEdit("P4 0 raise 125\nP5 0 fold\nP6 0 fold\nP1 0 fold");
    actions_->setToolTip("One action a line: player street(0-3) kind [to], kinds fold check call bet raise allin; "
                         "'to' is the street total after the action");
    actions_->setMaximumHeight(110);
    ft->addRow("Stacks (chips)", stacks_);
    ft->addRow("Button (seat)", button_);
    ft->addRow("You (seat)", me_);
    ft->addRow("Small / big blind", [&] {
        auto w = new QWidget;
        auto h = new QHBoxLayout(w);
        h->setContentsMargins(0, 0, 0, 0);
        h->addWidget(sb_), h->addWidget(bb_), h->addWidget(plain("big-blind ante")), h->addWidget(ante_);
        return w;
    }());
    ft->addRow("Your cards", hole_);
    ft->addRow("Board", board_);
    ft->addRow("Actions so far", actions_);
    ft->addRow("Payouts", payouts_);

    // heads-up
    hu_ = new QWidget;
    auto fh = new QFormLayout(hu_);
    huAction_ = new QLineEdit("b200c/");
    huAction_->setToolTip("ACPC action string: k check, c call, f fold, b<street total> bet or raise, streets split by /");
    huPos_ = new QComboBox;
    huPos_->addItems({"0: big blind", "1: small blind (button)"});
    huHole_ = new QLineEdit("Ah Kd");
    huBoard_ = new QLineEdit("7d Ts 2c");
    fh->addRow("Action string", huAction_);
    fh->addRow("Position", huPos_);
    fh->addRow("Your cards", huHole_);
    fh->addRow("Board", huBoard_);
    fh->addRow("Solver", solver_);

    auto forms = new QStackedWidget;
    forms->addWidget(table_);
    forms->addWidget(hu_);
    connect(game_, &QComboBox::currentIndexChanged, forms, &QStackedWidget::setCurrentIndex);

    auto askBtn = new QPushButton("Ask Kestrel");
    askBtn->setObjectName("primary");
    connect(askBtn, &QPushButton::clicked, this, &SpotPage::ask);
    answer_ = new QTextBrowser;
    drawn_ = new TableView;
    drawn_->setMinimumHeight(260);

    auto form = new QWidget;
    auto fl = new QVBoxLayout(form);
    fl->setContentsMargins(0, 0, 0, 0);
    auto head = new QFormLayout;
    head->addRow("Game", game_);
    head->addRow("Model", model_);
    fl->addLayout(head);
    fl->addWidget(forms);
    fl->addWidget(askBtn);
    fl->addStretch();
    auto right = new QWidget;
    auto rv = new QVBoxLayout(right);
    rv->setContentsMargins(0, 0, 0, 0);
    rv->addWidget(card(drawn_), 3);
    rv->addWidget(card(answer_, "Kestrel's advice"), 2);
    auto split = new QSplitter;
    split->addWidget(card(form, "The spot"));
    split->addWidget(right);
    split->setSizes({480, 800});
    auto l = new QVBoxLayout(this);
    l->setContentsMargins(20, 18, 20, 18);
    l->addWidget(split);
}

void SpotPage::setModels(const QStringList& nineMax) {
    const QString keep = model_->currentText();
    model_->clear();
    model_->addItems(nineMax);
    if (nineMax.contains(keep)) model_->setCurrentText(keep);
}

void SpotPage::ask() {
    QJsonObject body;
    if (game_->currentIndex() == 1) {
        body = {{"game", "headsup"}, {"action", huAction_->text().trimmed()}, {"pos", huPos_->currentIndex()},
                {"hole", toArray(cardsOf(huHole_->text()))}, {"board", toArray(cardsOf(huBoard_->text()))},
                {"solver", solver_->currentText()}};
    } else {
        QJsonArray players;
        const QStringList st = stacks_->text().split(',', Qt::SkipEmptyParts);
        for (int k = 0; k < st.size(); ++k) players.append(QJsonObject{{"id", QString("P%1").arg(k + 1)}, {"stack", st[k].trimmed().toDouble()}});
        QJsonArray acts;
        for (const QString& line : actions_->toPlainText().split('\n', Qt::SkipEmptyParts)) {
            const QStringList w = line.simplified().split(' ');
            if (w.size() < 3) continue;
            QJsonObject a{{"player", w[0]}, {"street", w[1].toInt()}, {"kind", w[2]}};
            if (w.size() > 3) a["to"] = w[3].toDouble();
            acts.append(a);
        }
        QJsonArray pay;
        for (const QString& x : payouts_->text().split(',', Qt::SkipEmptyParts)) pay.append(x.trimmed().toDouble());
        body = {{"game", "ninemax"}, {"model", model_->currentText()}, {"players", players},
                {"button", QString("P%1").arg(button_->value())}, {"me", QString("P%1").arg(me_->value())},
                {"sb", sb_->value()}, {"bb", bb_->value()}, {"ante", ante_->value()},
                {"hole", toArray(cardsOf(hole_->text()))}, {"board", toArray(cardsOf(board_->text()))},
                {"actions", acts}, {"payouts", pay}, {"similar", 8}};
    }
    answer_->setHtml("<p style='color:gray'>Asking…</p>");
    client_.post("/v1/decide", body, [this, body](const QJsonDocument& d, int, const QString& err) {
        const Palette& p = kestrel_app::palette(dark_());
        QJsonObject r = d.object();
        if (!err.isEmpty() || r.contains("error") && !r.value("error").toString().isEmpty()) {
            answer_->setHtml("<p>Kestrel could not read this spot: <b>" + esc(err.isEmpty() ? r.value("error").toString() : err) + "</b></p>");
            return;
        }
        QString h = bars(r.value("options").toArray(), p);
        if (r.contains("value_bb")) h += QString("<p style='color:%1'>value %2 bb · %3</p>").arg(p.muted, sign(r.value("value_bb").toDouble()), esc(r.value("model").toString()));
        if (r.contains("on_tree")) h += QString("<p style='color:%1'>%2%3</p>").arg(p.muted, r.value("on_tree").toBool() ? "on the blueprint's tree" : "off the tree", r.value("resolved").toBool() ? " · re-solved" : "");
        if (r.value("translated").toInt()) h += QString("<p style='color:%1'>%2 bet(s) at sizes the model does not use</p>").arg(p.muted).arg(r.value("translated").toInt());
        // the stored spots the network sees as most like this one (cosine similarity of its embeddings, kestrel.db)
        if (const QJsonArray sim = r.value("similar").toArray(); !sim.isEmpty()) {
            h += "<p style='margin-top:10px'><b>Similar spots Kestrel has played</b></p><table cellspacing='6'>"
                 "<tr><th align='left'>cosine</th><th align='left'>pos</th><th align='left'>cards</th><th align='left'>board</th>"
                 "<th align='left'>played</th><th align='right'>result</th></tr>";
            for (const auto& v : sim) {
                const QJsonObject o = v.toObject(), hd = o.value("hand").toObject();
                QStringList hole, board;
                for (const auto& x : hd.value("hole").toArray()) hole << x.toString();
                for (const auto& x : hd.value("board").toArray()) board << x.toString();
                const QJsonValue res = o.value("result_bb");
                h += "<tr><td>" + QString::number(o.value("similarity").toDouble(), 'f', 3) + "</td><td>" + esc(o.value("pos").toString()) +
                     "</td><td style='font-family:monospace'>" + esc(hole.join(' ')) + "</td><td style='font-family:monospace'>" +
                     esc(board.join(' ')) + "</td><td>" + esc(o.value("chose").toString()) + "</td><td align='right'>" +
                     (res.isDouble() ? sign(res.toDouble()) + " bb" : QString("–")) + "</td></tr>";
            }
            h += "</table>";
        }
        answer_->setHtml(h);
        drawn_->setState(r.value("view").toObject(), {}, dark_());
        if (body.value("game").toString() == "ninemax" && !body.value("payouts").toArray().isEmpty())
            equityTable(client_, body.value("players").toArray(), body.value("payouts").toArray(), body.value("me").toString(),
                        dark_(), [this, h](const QString& t) {
                            answer_->setHtml(h + "<p style='margin-top:10px'><b>Prize equity at the start of the hand</b></p>" + t +
                                             "<p style='color:gray'>The model reads Malmuth-Harville; Malmuth-Weitzman is shown "
                                             "for comparison (llm-research/docs/2026-11-15-tournament-equity-models.md).</p>");
                        });
    });
}

// ---- Trainer: pre-flop tournament spots, your move against the model's ----

TrainerPage::TrainerPage(Client& c, std::function<bool()> dark, QWidget* parent) : QWidget(parent), client_(c), dark_(dark) {
    setObjectName("page");
    model_ = new QComboBox;
    prompt_ = plain();
    prompt_->setStyleSheet("font-size: 15px;");
    score_ = plain();
    score_->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    drawn_ = new TableView;
    drawn_->setMinimumHeight(300);
    choices_ = new QWidget;
    new QHBoxLayout(choices_);
    verdict_ = new QTextBrowser;
    auto next = new QPushButton("Next spot");
    next->setObjectName("primary");
    connect(next, &QPushButton::clicked, this, &TrainerPage::deal);
    auto reset = new QPushButton("Reset score");
    connect(reset, &QPushButton::clicked, this, [this] {
        asked_ = agreed_ = 0, probSum_ = 0;
        score_->setText("No spots yet.");
    });
    auto top = new QHBoxLayout;
    top->addWidget(plain("Pre-flop tournament spots, 6 to 9 players at 8-30bb with ICM payouts: play your move, then see the model's."), 1);
    top->addWidget(new QLabel("Model"));
    top->addWidget(model_);
    top->addWidget(reset);
    top->addWidget(next);
    auto l = new QVBoxLayout(this);
    l->setContentsMargins(20, 18, 20, 18);
    l->setSpacing(12);
    l->addLayout(top);
    l->addWidget(card(drawn_), 3);
    l->addWidget(prompt_);
    l->addWidget(choices_);
    auto bottom = new QHBoxLayout;
    bottom->addWidget(card(verdict_, "The model"), 3);
    bottom->addWidget(card(score_, "Your score"), 1);
    l->addLayout(bottom, 2);
    score_->setText("No spots yet.");
}

void TrainerPage::setModels(const QStringList& nineMax) {
    const QString keep = model_->currentText();
    model_->clear();
    model_->addItems(nineMax);
    if (nineMax.contains(keep)) model_->setCurrentText(keep);
    if (!asked_ && !open_ && !nineMax.isEmpty()) deal();  // the first spot, as soon as there is a model
}

// a random spot: everyone folds to you (first in), or one player opens and you answer
void TrainerPage::deal() {
    static const double prizes[9] = {0.30, 0.20, 0.145, 0.105, 0.08, 0.065, 0.05, 0.035, 0.02};
    std::uniform_int_distribution<int> nd(6, 9);
    const int n = nd(rng_);
    const int b = std::uniform_int_distribution<int>(0, n - 1)(rng_);
    QJsonArray players, pay;
    double sum = 0;
    for (int k = 0; k < n; ++k) {
        double bbs = 8 * std::pow(30.0 / 8, std::uniform_real_distribution<double>(0, 1)(rng_));
        players.append(QJsonObject{{"id", QString("P%1").arg(k + 1)}, {"stack", std::round(bbs * 100)}});
        sum += prizes[k];
    }
    for (int k = 0; k < n; ++k) pay.append(prizes[k] / sum);
    // pre-flop order: under the gun first, the big blind last
    std::vector<int> order;
    for (int k = 0; k < n; ++k) order.push_back((b + 3 + k) % n);
    std::vector<int> cards(52);
    for (int k = 0; k < 52; ++k) cards[k] = k;
    std::shuffle(cards.begin(), cards.end(), rng_);
    auto name = [](int c) { return QString("23456789TJQKA"[c / 4]) + QString("shdc"[c % 4]); };
    QJsonArray acts;
    int hero;
    if (std::uniform_real_distribution<double>(0, 1)(rng_) < 0.6) {
        const int k = std::uniform_int_distribution<int>(0, n - 2)(rng_);  // not the big blind: it would win the blinds
        for (int j = 0; j < k; ++j) acts.append(QJsonObject{{"player", QString("P%1").arg(order[j] + 1)}, {"street", 0}, {"kind", "fold"}});
        hero = order[k];
    } else {
        const int j = std::uniform_int_distribution<int>(0, n - 2)(rng_);
        const int k = std::uniform_int_distribution<int>(j + 1, n - 1)(rng_);
        for (int t = 0; t < k; ++t) {
            QJsonObject a{{"player", QString("P%1").arg(order[t] + 1)}, {"street", 0}};
            if (t == j) a["kind"] = "raise", a["to"] = std::uniform_int_distribution<int>(0, 1)(rng_) ? 200 : 250;
            else a["kind"] = "fold";
            acts.append(a);
        }
        hero = order[k];
    }
    QJsonObject body{{"game", "ninemax"}, {"model", model_->currentText()}, {"players", players},
                     {"button", QString("P%1").arg(b + 1)}, {"me", QString("P%1").arg(hero + 1)},
                     {"sb", 50}, {"bb", 100}, {"ante", 100}, {"hole", QJsonArray{name(cards[0]), name(cards[1])}},
                     {"actions", acts}, {"payouts", pay}};
    open_ = false;
    verdict_->setHtml("");
    spot_ = body;
    equity_.clear();
    equityTable(client_, players, pay, body.value("me").toString(), dark_(), [this](const QString& t) { equity_ = t; });
    client_.post("/v1/decide", body, [this](const QJsonDocument& d, int, const QString& err) {
        advice_ = d.object();
        if (!err.isEmpty() || advice_.value("options").toArray().isEmpty()) {
            prompt_->setText("Could not deal a spot: " + (err.isEmpty() ? advice_.value("error").toString() : err));
            return;
        }
        const QJsonObject view = advice_.value("view").toObject();
        drawn_->setState(view, {}, dark_());
        double stack = 0;
        for (const auto& v : view.value("seats").toArray())
            if (v.toObject().value("you").toBool()) stack = v.toObject().value("stack_bb").toDouble() + v.toObject().value("in_bb").toDouble();
        prompt_->setText(QString("You are %1 with %2 bb, %3 players left. Your move?")
                             .arg(view.value("pos").toString())
                             .arg(stack, 0, 'f', 1)
                             .arg(view.value("players_left").toInt()));
        auto l = static_cast<QHBoxLayout*>(choices_->layout());
        while (auto it = l->takeAt(0)) {
            if (it->widget()) it->widget()->deleteLater();
            delete it;
        }
        const QJsonArray opts = advice_.value("options").toArray();
        for (int k = 0; k < opts.size(); ++k) {
            auto b = new QPushButton(opts[k].toObject().value("label").toString());
            if (opts[k].toObject().value("action").toString() != "fold") b->setObjectName("primary");
            connect(b, &QPushButton::clicked, this, [this, k] { answer(k); });
            l->addWidget(b);
        }
        l->addStretch();
        open_ = true;
    });
}

void TrainerPage::answer(int k) {
    if (!open_) return;
    open_ = false;
    const QJsonArray opts = advice_.value("options").toArray();
    double best = 0;
    for (const auto& v : opts) best = std::max(best, v.toObject().value("prob").toDouble());
    const double mine = opts[k].toObject().value("prob").toDouble();
    const bool agree = mine >= 0.5 * best;  // a move the model plays often enough to count as its own
    ++asked_, agreed_ += agree, probSum_ += mine;
    const Palette& p = kestrel_app::palette(dark_());
    verdict_->setHtml(QString("<p style='font-size:15px;color:%1'><b>%2</b></p>")
                          .arg(agree ? p.win : p.loss,
                               agree ? "The model plays this too." : QString("The model plays this %1% of the time.").arg(int(mine * 100 + 0.5))) +
                      bars(opts, p, k) +
                      (equity_.isEmpty() ? QString() : "<p style='margin-top:10px'><b>Prize equity at the start of the hand</b></p>" + equity_));
    score_->setText(QString("%1 spots\n%2 agreed (%3%)\naverage model probability of your moves: %4%")
                        .arg(asked_)
                        .arg(agreed_)
                        .arg(int(100.0 * agreed_ / asked_ + 0.5))
                        .arg(int(100 * probSum_ / asked_ + 0.5)));
}

// ---- Opponents ----

OpponentsPage::OpponentsPage(Client& c, QWidget* parent) : QWidget(parent), client_(c) {
    setObjectName("page");
    platform_ = new QComboBox;
    platform_->setMinimumWidth(180);
    connect(platform_, &QComboBox::currentTextChanged, this, [this] { showPlatform(); });
    table_ = new QTableWidget(0, 9);
    table_->setHorizontalHeaderLabels({"Player", "Hands", "VPIP %", "Pre-flop raise %", "3-bet %", "Fold to c-bet %", "Aggression", "Showdown %", "Type"});
    table_->verticalHeader()->hide();
    table_->setSelectionBehavior(QAbstractItemView::SelectRows);
    table_->setSortingEnabled(true);
    table_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    shown_ = new QTextBrowser;
    connect(table_, &QTableWidget::currentCellChanged, this, [this](int row) {
        if (row < 0 || !table_->item(row, 0)) return;
        const QString name = table_->item(row, 0)->text();
        const QJsonObject pr = players_.value(name).toObject().value("profile").toObject();
        QString h = "<p><b>" + esc(name) + "</b>: closest type <b>" + esc(pr.value("type").toString()) + "</b> (cosine " +
                    QString::number(pr.value("similarity").toDouble(), 'f', 2) + "), then " + esc(pr.value("runner_up").toString()) +
                    " (" + QString::number(pr.value("runner_up_similarity").toDouble(), 'f', 2) + ")" +
                    (pr.value("settled").toBool() ? QString() : QString(" &mdash; fewer than 30 hands, not settled")) + "</p>";
        QString shownHtml = "<p><b>The last hands shown</b></p>";
        for (const auto& v : players_.value(name).toObject().value("shown").toArray()) {
            QStringList c, b;
            for (const auto& x : v.toObject().value("cards").toArray()) c << x.toString();
            for (const auto& x : v.toObject().value("board").toArray()) b << x.toString();
            shownHtml += "<div style='font-family:monospace'>" + esc(c.join(' ')) + " &nbsp;|&nbsp; " + esc(b.join(' ')) + "</div>";
        }
        shown_->setHtml(h + shownHtml);
        // the players met before who play most like this one (cosine similarity of their profiles)
        client_.get("/v1/opponents?platform=" + QUrl::toPercentEncoding(platform_->currentText()) + "&player=" + QUrl::toPercentEncoding(name),
                    [this, h, shownHtml](const QJsonDocument& d, int, const QString& err) {
                        if (!err.isEmpty()) return;
                        QStringList like;
                        for (const auto& v : d.object().value("like").toArray()) {
                            const QJsonObject o = v.toObject();
                            like << esc(o.value("player").toString()) + " " + QString::number(o.value("similarity").toDouble(), 'f', 2);
                        }
                        shown_->setHtml(h + "<p>Plays most like: " + (like.isEmpty() ? QString("no one yet (30 hands needed)") : like.join(", ")) +
                                        "</p>" + shownHtml);
                    });
    });
    auto reloadBtn = new QPushButton("Reload");
    connect(reloadBtn, &QPushButton::clicked, this, &OpponentsPage::reload);
    auto top = new QHBoxLayout;
    top->addWidget(new QLabel("Platform"));
    top->addWidget(platform_);
    top->addWidget(plain("Every opponent Kestrel has met, kept across sessions. Statistics only: Kestrel does not yet change its play on them."), 1);
    top->addWidget(reloadBtn);
    auto split = new QSplitter(Qt::Vertical);
    split->addWidget(card(table_, "Opponents"));
    split->addWidget(card(shown_, "Hands shown"));
    split->setSizes({500, 200});
    auto l = new QVBoxLayout(this);
    l->setContentsMargins(20, 18, 20, 18);
    l->addLayout(top);
    l->addWidget(split, 1);
}

void OpponentsPage::reload() {
    client_.get("/v1/opponents", [this](const QJsonDocument& d, int, const QString& err) {
        if (!err.isEmpty()) return;
        const QString keep = platform_->currentText();
        platform_->blockSignals(true);
        platform_->clear();
        for (const auto& v : d.object().value("platforms").toArray()) platform_->addItem(v.toObject().value("platform").toString());
        if (!keep.isEmpty()) platform_->setCurrentText(keep);
        platform_->blockSignals(false);
        showPlatform();
    });
}

void OpponentsPage::showPlatform() {
    const QString plat = platform_->currentText();
    if (plat.isEmpty()) return;
    client_.get("/v1/opponents?platform=" + QUrl::toPercentEncoding(plat), [this](const QJsonDocument& d, int, const QString& err) {
        if (!err.isEmpty()) return;
        players_ = d.object().value("players").toObject();
        table_->setSortingEnabled(false);
        table_->setRowCount(0);
        auto num = [](double x) {
            auto i = new QTableWidgetItem;
            i->setData(Qt::DisplayRole, x < 0 ? QVariant("–") : QVariant(x));  // -1: no chance yet
            i->setFlags(i->flags() & ~Qt::ItemIsEditable);
            return i;
        };
        for (auto it = players_.begin(); it != players_.end(); ++it) {
            const QJsonObject s = it.value().toObject();
            const int r = table_->rowCount();
            table_->insertRow(r);
            auto name = new QTableWidgetItem(it.key());
            name->setFlags(name->flags() & ~Qt::ItemIsEditable);
            table_->setItem(r, 0, name);
            table_->setItem(r, 1, num(s.value("hands").toDouble()));
            int c = 2;
            for (const char* k : {"vpip", "pfr", "threebet", "fold_to_cbet", "aggression", "went_to_showdown"})
                table_->setItem(r, c++, num(s.value(k).toDouble()));
            const QJsonObject pr = s.value("profile").toObject();
            auto type = new QTableWidgetItem(pr.value("settled").toBool() ? pr.value("type").toString() + " " +
                                                                                 QString::number(pr.value("similarity").toDouble(), 'f', 2)
                                                                           : "(" + pr.value("type").toString() + ")");
            type->setFlags(type->flags() & ~Qt::ItemIsEditable);
            table_->setItem(r, c, type);
        }
        table_->setSortingEnabled(true);
        table_->sortItems(1, Qt::DescendingOrder);
    });
}

}  // namespace kestrel_app
