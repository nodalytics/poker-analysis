#include "client.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QSaveFile>
#include <QStandardPaths>

namespace kestrel_app {

static QString home() {
    QString h = qEnvironmentVariable("HOME");
    if (h.isEmpty()) h = qEnvironmentVariable("USERPROFILE");
    return h.isEmpty() ? QStringLiteral(".") : h;
}

QString configDir() {
    QString x = qEnvironmentVariable("XDG_CONFIG_HOME");
    return x.isEmpty() ? home() + "/.config/kestrel" : x + "/kestrel";
}
QString settingsPath() { return configDir() + "/settings.json"; }
QString connectorsDir() { return configDir() + "/connectors"; }
QString modelsDir() {
    QString x = qEnvironmentVariable("XDG_DATA_HOME");
    return x.isEmpty() ? home() + "/.local/share/kestrel/models" : x + "/kestrel/models";
}

QString kestreldPath() {
#ifdef Q_OS_WIN
    const QString exe = QStringLiteral("kestreld.exe");
#else
    const QString exe = QStringLiteral("kestreld");
#endif
    QString beside = QDir(QCoreApplication::applicationDirPath()).filePath(exe);
    if (QFileInfo::exists(beside)) return beside;
    QString onPath = QStandardPaths::findExecutable(exe);
    return onPath.isEmpty() ? beside : onPath;
}

QJsonObject readSettings() {
    QFile f(settingsPath());
    if (!f.open(QIODevice::ReadOnly)) return {};
    return QJsonDocument::fromJson(f.readAll()).object();
}

bool writeSettings(const QJsonObject& s, QString* error) {
    QDir().mkpath(configDir());
    QSaveFile f(settingsPath());  // written whole or not at all
    if (!f.open(QIODevice::WriteOnly)) {
        if (error) *error = f.errorString();
        return false;
    }
    f.write(QJsonDocument(s).toJson(QJsonDocument::Indented));
    if (!f.commit()) {
        if (error) *error = f.errorString();
        return false;
    }
    return true;
}

Client::Client(QObject* parent) : QObject(parent) {
    port_ = readSettings().value("port").toInt(8790);
    connect(&health_, &QTimer::timeout, this, &Client::poll);
    health_.start(2000);
    poll();
}

void Client::setPort(int p) {
    port_ = p;
    poll();
}

void Client::send(const QString& method, const QString& path, const QByteArray& body, Done done) {
    QNetworkRequest rq(QUrl(base() + path));
    rq.setRawHeader("Origin", base().toUtf8());  // the server takes writes only from its own origin
    rq.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    rq.setTransferTimeout(30000);
    QNetworkReply* r = method == "GET"    ? net_.get(rq)
                       : method == "POST" ? net_.post(rq, body)
                                          : net_.sendCustomRequest(rq, method.toUtf8(), body);
    connect(r, &QNetworkReply::finished, this, [r, done] {
        r->deleteLater();
        int status = r->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        QJsonDocument doc = QJsonDocument::fromJson(r->readAll());
        QString err;
        if (r->error() != QNetworkReply::NoError && status == 0) err = r->errorString();
        else if (status >= 400) err = doc.object().value("error").toString(QStringLiteral("HTTP %1").arg(status));
        if (done) done(doc, status, err);
    });
}

void Client::get(const QString& path, Done done) { send("GET", path, {}, std::move(done)); }
void Client::post(const QString& path, const QJsonObject& body, Done done) {
    send("POST", path, QJsonDocument(body).toJson(QJsonDocument::Compact), std::move(done));
}
void Client::del(const QString& path, Done done) { send("DELETE", path, {}, std::move(done)); }

void Client::poll() {
    get("/v1/health", [this](const QJsonDocument& d, int status, const QString&) {
        bool up = status == 200 && d.object().value("ok").toBool();
        if (up) starting_ = false;
        if (up != online_) {
            online_ = up;
            emit onlineChanged(up);
            if (up) listen();
        }
        if (up && !stream_) listen();
    });
}

// the live events (server-sent events), one JSON object per "data:" line
void Client::listen() {
    if (stream_) return;
    QNetworkRequest rq(QUrl(base() + "/v1/events"));
    rq.setAttribute(QNetworkRequest::CacheLoadControlAttribute, QNetworkRequest::AlwaysNetwork);
    stream_ = net_.get(rq);
    buf_.clear();
    connect(stream_, &QNetworkReply::readyRead, this, [this] {
        buf_ += stream_->readAll();
        if (qEnvironmentVariableIsSet("KESTREL_APP_DEBUG")) qWarning("stream: %lld bytes buffered", (long long)buf_.size());
        int k;
        while ((k = buf_.indexOf("\n\n")) >= 0) {
            QByteArray block = buf_.left(k);
            buf_.remove(0, k + 2);
            int p = block.indexOf("data: ");
            if (p < 0) continue;
            QJsonDocument d = QJsonDocument::fromJson(block.mid(p + 6));
            if (d.isObject()) emit event(d.object());
        }
    });
    connect(stream_, &QNetworkReply::finished, this, [this] {
        if (qEnvironmentVariableIsSet("KESTREL_APP_DEBUG")) qWarning("stream ended: %s", qPrintable(stream_->errorString()));
        stream_->deleteLater();
        stream_ = nullptr;  // reconnected by the next health check
    });
}

void Client::ensureServer() {
    if (online_ || starting_ || owned()) return;
    starting_ = true;
    proc_ = new QProcess(this);
    proc_->setProcessChannelMode(QProcess::MergedChannels);
    connect(proc_, &QProcess::readyRead, this, [this] {
        const QString text = QString::fromUtf8(proc_->readAll());
        log_ += text;
        if (log_.size() > 200000) log_ = log_.right(150000);
        for (const QString& l : text.split('\n', Qt::SkipEmptyParts)) emit logLine(l);
    });
    connect(proc_, &QProcess::finished, this, [this](int code, QProcess::ExitStatus) {
        emit logLine(QStringLiteral("kestreld stopped (exit %1)").arg(code));
        starting_ = false;
        poll();
    });
    emit logLine("starting " + kestreldPath() + " start");
    proc_->start(kestreldPath(), {"start"});
    if (!proc_->waitForStarted(3000)) {
        emit logLine("could not start kestreld: " + proc_->errorString());
        starting_ = false;
    }
}

void Client::stopServer(std::function<void()> then) {
    post("/v1/admin/shutdown", {}, [this, then](const QJsonDocument&, int, const QString& err) {
        if (!err.isEmpty()) emit logLine("stop: " + err);
        QTimer::singleShot(800, this, [this, then] {
            poll();
            if (then) then();
        });
    });
}

void Client::restartServer() {
    const int next = readSettings().value("port").toInt(8790);  // the old server is stopped on its own port first
    stopServer([this, next] {
        port_ = next;
        auto wait = new QTimer(this);
        int* tries = new int(0);
        connect(wait, &QTimer::timeout, this, [this, wait, tries] {
            if ((!owned() && !online_) || ++*tries > 20) {
                wait->stop();
                wait->deleteLater();
                delete tries;
                ensureServer();
            }
        });
        wait->start(250);
    });
}

}  // namespace kestrel_app
