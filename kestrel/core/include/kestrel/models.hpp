// Installed models: packages from research/scripts/model_pipeline.py (a <name>/ folder with model.json, the
// weights, net.bin/net.json and test vectors), checked file by file against the sha256 in model.json.
#pragma once
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace kestrel {

std::string sha256_file(const std::string& path);
std::string default_models_dir();  // $XDG_DATA_HOME/kestrel/models, else ~/.local/share/kestrel/models

struct ModelInfo {
    std::string name, version, game, about, dir;
    int players = 0;
    bool verified = false;
    std::string problem;  // why it does not verify
};

std::vector<ModelInfo> list_models(const std::string& dir);
ModelInfo verify_model(const std::string& model_dir);
// checks PACKAGE.tar.gz against PACKAGE.tar.gz.sha256, unpacks it, verifies every file, then puts it in dir/<name>
// (replacing an older version of the same name); throws with the reason when anything does not match
ModelInfo install_model(const std::string& package, const std::string& dir);

}  // namespace kestrel
