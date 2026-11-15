#include "tableview.h"

#include <QJsonArray>
#include <QPainter>
#include <QPainterPath>
#include <QtMath>
#include <algorithm>

#include "theme.h"

namespace kestrel_app {

namespace {

const char* kStreets[] = {"Pre-flop", "Flop", "Turn", "River"};

// four-colour cards, as the web app: spades slate, hearts red, diamonds blue, clubs green
QColor suitColor(QChar s, bool dark) {
    switch (s.toLower().unicode()) {
        case 'h': return QColor(dark ? "#EF4444" : "#DC2626");
        case 'd': return QColor(dark ? "#3B82F6" : "#2563EB");
        case 'c': return QColor(dark ? "#22C55E" : "#16A34A");
        default: return QColor("#0F172A");
    }
}

QString suitGlyph(QChar s) {
    switch (s.toLower().unicode()) {
        case 'h': return QStringLiteral("♥");
        case 'd': return QStringLiteral("♦");
        case 'c': return QStringLiteral("♣");
        default: return QStringLiteral("♠");
    }
}

bool isCard(const QString& c) {
    return c.size() == 2 && QStringLiteral("23456789TJQKA").contains(c[0].toUpper()) &&
           QStringLiteral("shdc").contains(c[1].toLower());
}

void drawCard(QPainter& p, const QRectF& r, const QString& c, bool dark) {
    p.setPen(Qt::NoPen);
    if (!isCard(c)) {  // a face-down card
        p.setBrush(QColor(dark ? "#1F2433" : "#CBD5E1"));
        p.drawRoundedRect(r, r.width() * 0.12, r.width() * 0.12);
        p.setBrush(QColor("#F97316"));
        p.setOpacity(0.35);
        p.drawRoundedRect(r.adjusted(r.width() * 0.18, r.height() * 0.14, -r.width() * 0.18, -r.height() * 0.14), 3, 3);
        p.setOpacity(1);
        return;
    }
    p.setBrush(QColor("#F8FAFC"));
    p.drawRoundedRect(r, r.width() * 0.12, r.width() * 0.12);
    QColor col = suitColor(c[1], dark);
    p.setPen(col);
    QFont f = p.font();
    f.setBold(true);
    f.setPixelSize(int(r.height() * 0.36));
    p.setFont(f);
    QString rank = c[0].toUpper() == 'T' ? QStringLiteral("10") : QString(c[0].toUpper());
    p.drawText(r.adjusted(0, r.height() * 0.06, 0, -r.height() * 0.45), Qt::AlignHCenter | Qt::AlignTop, rank);
    f.setPixelSize(int(r.height() * 0.38));
    f.setBold(false);
    p.setFont(f);
    p.drawText(r.adjusted(0, r.height() * 0.45, 0, -r.height() * 0.04), Qt::AlignHCenter | Qt::AlignBottom, suitGlyph(c[1]));
}

QString bbText(double x) {
    return std::abs(x - std::round(x)) < 1e-9 ? QString::number(x, 'f', 0) : QString::number(x, 'f', 1);
}

struct Seat {
    int seat = 0;
    QString name, pos, said, result, hud;
    double stack = -1, icm = -1, in = 0, bounty = 0;
    QStringList hole;
    bool folded = false, you = false, acting = false, showHole = false, win = false;
};

}  // namespace

TableView::TableView(QWidget* parent) : QWidget(parent) {
    setMinimumSize(240, 160);
    setAttribute(Qt::WA_OpaquePaintEvent, false);
}

void TableView::setState(const QJsonObject& e, const QJsonObject& table, bool dark) {
    e_ = e, t_ = table, dark_ = dark;
    update();
}

void TableView::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    const Palette& pal = kestrel_app::palette(dark_);
    const QRectF R = rect().adjusted(4, 4, -4, -4);
    const double u = std::min(R.width() / 900.0, R.height() / 560.0);  // scale everything with the widget

    // the felt
    QRectF oval(R.left() + R.width() * 0.12, R.top() + R.height() * 0.16, R.width() * 0.76, R.height() * 0.66);
    QColor felt(dark_ ? "#0F1A17" : "#EAF3EF"), feltLine(dark_ ? "#1C2B26" : "#D3E4DC");
    p.setPen(QPen(feltLine, std::max(1.0, 3 * u)));
    p.setBrush(felt);
    p.drawRoundedRect(oval, oval.height() / 2, oval.height() / 2);

    const QString type = e_.value("type").toString();
    const bool done = type == "practice_hand" || type == "hand";
    QFont base = font();

    // the label: street, hand, level
    {
        QFont f = base;
        f.setPixelSize(std::max(9, int(12 * u + 2)));
        f.setLetterSpacing(QFont::AbsoluteSpacing, 1);
        p.setFont(f);
        p.setPen(QColor(pal.muted));
        int st = std::clamp(e_.value("street").toInt(), 0, 3);
        QString label = done ? (e_.value("showdown").toBool() ? "SHOWDOWN" : "HAND OVER") : QString(kStreets[st]).toUpper();
        if (e_.contains("hand")) label += " · HAND " + e_.value("hand").toVariant().toString();
        if (e_.contains("level")) label += " · LEVEL " + QString::number(e_.value("level").toInt());
        if (e_.isEmpty()) label = "WAITING FOR THE FIRST HAND";
        p.drawText(R.adjusted(8 * u, 4 * u, 0, 0), Qt::AlignLeft | Qt::AlignTop, label);
    }

    // the seats
    std::vector<Seat> seats;
    QMap<QString, QString> last;  // a seat's last action this hand, from the log
    for (const auto& v : e_.value("log").toArray()) {
        QJsonObject x = v.toObject();
        last[x.value("name").toString()] = x.value("action").toString();
    }
    QMap<QString, QJsonObject> results;
    for (const auto& v : e_.value("results").toArray()) results[v.toObject().value("name").toString()] = v.toObject();
    const int actor = done ? -1 : e_.value("actor").toInt(-1);
    if (e_.contains("seats") || e_.contains("results")) {
        const QJsonArray list = done && e_.contains("results") ? e_.value("results").toArray() : e_.value("seats").toArray();
        for (const auto& v : list) {
            QJsonObject s = v.toObject();
            Seat x;
            x.seat = s.value("seat").toInt();
            x.name = s.value("name").toString();
            x.pos = s.value("pos").toString();
            x.stack = s.contains("stack_bb") && !s.value("stack_bb").isNull() ? s.value("stack_bb").toDouble() : -1;
            x.icm = s.contains("icm") && !s.value("icm").isNull() ? s.value("icm").toDouble() : -1;
            x.in = s.value("in_bb").toDouble();
            x.bounty = s.value("bounty").toDouble();
            x.hud = s.value("hud").toString();
            x.folded = s.value("folded").toBool();
            x.you = s.value("you").toBool();
            x.acting = x.seat == actor;
            for (const auto& c : s.value("hole").toArray()) x.hole << c.toString();
            x.said = last.value(x.name);
            if (done && results.contains(x.name)) {
                double r = results[x.name].value("result_bb").toDouble();
                x.result = (r >= 0 ? "+" : "") + bbText(r) + " bb", x.win = r > 0;
            }
            seats.push_back(x);
        }
    } else if (!e_.isEmpty()) {  // heads-up: the platform's two seats (pos 1: we are the button)
        const double big = e_.value("big").toDouble(100);
        Seat them, me;
        them.seat = 0, me.seat = 1;
        them.name = "Opponent", me.name = "Kestrel";
        const int pos = e_.value("pos").toInt();
        me.pos = pos == 1 ? "BTN" : "BB", them.pos = pos == 1 ? "BB" : "BTN";
        if (e_.contains("my_stack")) me.stack = e_.value("my_stack").toDouble() / big;
        if (e_.contains("their_stack")) them.stack = e_.value("their_stack").toDouble() / big;
        me.in = e_.value("my_bet").toDouble() / big, them.in = e_.value("their_bet").toDouble() / big;
        for (const auto& c : e_.value("hole").toArray()) me.hole << c.toString();
        for (const auto& c : e_.value("their_hole").toArray()) them.hole << c.toString();
        me.you = true;
        if (done) {
            double r = e_.value("result_bb").toDouble();
            me.result = (r >= 0 ? "+" : "") + bbText(r) + " bb", me.win = r > 0;
        }
        seats = {them, me};
    }
    std::vector<int> order;
    for (auto& s : seats) order.push_back(s.seat);
    std::sort(order.begin(), order.end());
    const int n = int(order.size());
    for (auto& s : seats) {
        int k = n ? (int(std::find(order.begin(), order.end(), s.seat) - order.begin()) + 1) % n : 0;
        double a = M_PI / 2 + k * 2 * M_PI / std::max(n, 1);
        if (n == 2) a = s.you ? M_PI / 2 : -M_PI / 2;  // heads-up: us at the bottom
        QPointF c(R.left() + R.width() * (0.5 + 0.41 * std::cos(a)), R.top() + R.height() * (0.52 + 0.38 * std::sin(a)));
        const double w = 150 * u + 30, h = (compact_ ? 70 : s.hud.isEmpty() ? 104 : 116) * u + 26;
        QRectF box(c.x() - w / 2, c.y() - h / 2, w, h);
        // inside the frame, below the label
        const double top = R.top() + 22 * u + 8;
        if (box.bottom() > R.bottom()) box.moveBottom(R.bottom());
        if (box.top() < top) box.moveTop(top);
        if (box.left() < R.left()) box.moveLeft(R.left());
        if (box.right() > R.right()) box.moveRight(R.right());
        p.setOpacity(s.folded && !done ? 0.45 : 1.0);
        p.setPen(QPen(QColor(s.acting ? pal.accent : s.you ? pal.accent2 : pal.line), s.acting ? 2 : 1));
        p.setBrush(QColor(pal.panel));
        p.drawRoundedRect(box, 10 * u + 3, 10 * u + 3);
        QFont f = base;
        f.setPixelSize(std::max(9, int(13 * u + 3)));
        f.setBold(true);
        p.setFont(f);
        p.setPen(QColor(pal.text));
        QRectF line = box.adjusted(6, 4 * u + 2, -6, 0);
        p.drawText(line, Qt::AlignHCenter | Qt::AlignTop, s.name);
        f.setBold(false);
        f.setPixelSize(std::max(8, int(11 * u + 2)));
        p.setFont(f);
        p.setPen(QColor(pal.muted));
        QString meta = s.pos;
        if (s.stack >= 0) meta += " · " + bbText(s.stack) + " bb";
        if (s.icm >= 0 && !compact_) meta += " · ICM " + QString::number(100 * s.icm, 'f', 1) + "%";
        if (s.bounty > 0) meta += " · bounty " + QString::number(100 * s.bounty, 'f', 1) + "%";
        p.drawText(line.adjusted(0, 17 * u + 4, 0, 0), Qt::AlignHCenter | Qt::AlignTop, meta);
        if (!s.hud.isEmpty() && !compact_) {  // VPIP / pre-flop raise / 3-bet, % · hands seen
            p.setPen(QColor(pal.accent2));
            p.drawText(line.adjusted(0, 31 * u + 6, 0, 0), Qt::AlignHCenter | Qt::AlignTop, s.hud);
            p.setPen(QColor(pal.muted));
        }
        if (!compact_) {
            const double ch = 38 * u + 8, cw = ch * 0.7;
            QRectF c1(box.center().x() - cw - 2, box.top() + (s.hud.isEmpty() ? 40 : 52) * u + 12, cw, ch), c2 = c1.translated(cw + 4, 0);
            if (!s.hole.isEmpty()) {
                drawCard(p, c1, s.hole.value(0), dark_), drawCard(p, c2, s.hole.value(1), dark_);
            } else if (!s.folded) {
                drawCard(p, c1, {}, dark_), drawCard(p, c2, {}, dark_);
            }
        }
        QString said = !s.result.isEmpty() ? s.result : !s.said.isEmpty() ? s.said : s.in > 0 ? "in " + bbText(s.in) : QString();
        p.setPen(QColor(!s.result.isEmpty() ? (s.win ? pal.win : pal.loss) : pal.accent2));
        p.drawText(box.adjusted(4, 0, -4, -3 * u - 1), Qt::AlignHCenter | Qt::AlignBottom, said);
        p.setOpacity(1);
    }

    // the board and the pot
    const double ch = (compact_ ? 34 : 64) * u + 10, cw = ch * 0.7, gap = 6 * u + 2;
    QStringList board;
    for (const auto& c : e_.value("board").toArray()) board << c.toString();
    double x0 = oval.center().x() - (5 * cw + 4 * gap) / 2, y0 = oval.center().y() - ch / 2 - 8 * u;
    for (int k = 0; k < 5; ++k) {
        QRectF r(x0 + k * (cw + gap), y0, cw, ch);
        if (k < board.size()) {
            drawCard(p, r, board[k], dark_);
        } else {
            p.setPen(QPen(feltLine, 1, Qt::DashLine));
            p.setBrush(Qt::NoBrush);
            p.drawRoundedRect(r, cw * 0.12, cw * 0.12);
        }
    }
    double pot = e_.contains("pot_bb") ? e_.value("pot_bb").toDouble()
                 : e_.contains("pot")  ? e_.value("pot").toDouble() / e_.value("big").toDouble(100)
                                       : -1;
    if (pot >= 0 && !done) {
        QFont f = base;
        f.setPixelSize(std::max(9, int(14 * u + 3)));
        p.setFont(f);
        p.setPen(QColor(pal.muted));
        p.drawText(QRectF(oval.left(), y0 + ch + 6 * u, oval.width(), 30 * u + 12), Qt::AlignHCenter | Qt::AlignTop,
                   "pot " + bbText(pot) + " bb");
    }
}

}  // namespace kestrel_app
