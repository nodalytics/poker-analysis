#include <QApplication>

#include "window.h"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QApplication::setOrganizationName("Kestrel");
    QApplication::setApplicationName("Kestrel");
    kestrel_app::Window w;
    const QStringList args = app.arguments();
    if (int k = args.indexOf("--tab"); k > 0 && k + 1 < args.size()) w.showTab(args[k + 1]);
    w.show();
    return app.exec();
}
