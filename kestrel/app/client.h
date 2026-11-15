// The desktop app's link to Kestrel: the same local REST API and live event stream the web app uses, the user's
// folders (settings, connectors, models), and the kestreld process it starts when no server is running.
#pragma once
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QObject>
#include <QPointer>
#include <QProcess>
#include <QTimer>
#include <functional>

namespace kestrel_app {

// the folders kestreld reads (tools/kestreld.cpp: config_dir, default_dir, default_models_dir)
QString configDir();
QString settingsPath();
QString connectorsDir();
QString modelsDir();
QString kestreldPath();  // beside this program, else on PATH

QJsonObject readSettings();
bool writeSettings(const QJsonObject& s, QString* error);

class Client : public QObject {
    Q_OBJECT
  public:
    using Done = std::function<void(const QJsonDocument& body, int status, const QString& error)>;
    explicit Client(QObject* parent = nullptr);

    int port() const { return port_; }
    QString base() const { return QStringLiteral("http://127.0.0.1:%1").arg(port_); }
    void setPort(int p);

    void get(const QString& path, Done done);
    void post(const QString& path, const QJsonObject& body, Done done);
    void del(const QString& path, Done done);

    // the server: one already running is used; otherwise the app starts `kestreld start` and owns it
    bool online() const { return online_; }
    bool owned() const { return proc_ && proc_->state() != QProcess::NotRunning; }
    void ensureServer();
    void stopServer(std::function<void()> then = nullptr);
    void restartServer();
    QString serverLog() const { return log_; }

  signals:
    void onlineChanged(bool online);
    void event(const QJsonObject& e);
    void logLine(const QString& line);

  private:
    void send(const QString& method, const QString& path, const QByteArray& body, Done done);
    void poll();
    void listen();
    QNetworkAccessManager net_;
    QPointer<QProcess> proc_;
    QPointer<QNetworkReply> stream_;
    QByteArray buf_;
    QTimer health_;
    QString log_;
    int port_ = 8790;
    bool online_ = false, starting_ = false;
};

}  // namespace kestrel_app
