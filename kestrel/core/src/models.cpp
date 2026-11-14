#include "kestrel/models.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

namespace kestrel {

namespace fs = std::filesystem;

namespace {

// SHA-256 (FIPS 180-4)
struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t buf[64];
    size_t len = 0;
    uint64_t bits = 0;
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const uint8_t* p) {
        static const uint32_t K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) w[i] = uint32_t(p[4 * i]) << 24 | uint32_t(p[4 * i + 1]) << 16 | uint32_t(p[4 * i + 2]) << 8 | p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t t1 = k + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            k = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
        }
        h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += k;
    }
    void update(const uint8_t* p, size_t n) {
        bits += uint64_t(n) * 8;
        while (n--) {
            buf[len++] = *p++;
            if (len == 64) block(buf), len = 0;
        }
    }
    std::string hex() {
        uint64_t total = bits;
        uint8_t one = 0x80, zero = 0;
        update(&one, 1);
        while (len != 56) update(&zero, 1);
        uint8_t l[8];
        for (int i = 0; i < 8; ++i) l[i] = uint8_t(total >> (56 - 8 * i));
        update(l, 8);
        char out[65];
        for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
        return std::string(out, 64);
    }
};

std::string read_sidecar(const std::string& path) {
    std::ifstream f(path);
    std::string hex;
    f >> hex;
    return hex;
}

std::string quote(const std::string& s) {
#ifdef _WIN32
    return "\"" + s + "\"";
#else
    std::string q = "'";
    for (char c : s) q += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return q + "'";
#endif
}

}  // namespace

std::string sha256_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    Sha256 s;
    std::vector<char> b(1 << 20);
    while (f) {
        f.read(b.data(), std::streamsize(b.size()));
        s.update(reinterpret_cast<const uint8_t*>(b.data()), size_t(f.gcount()));
    }
    return s.hex();
}

std::string default_models_dir() {
    if (const char* x = std::getenv("XDG_DATA_HOME")) return std::string(x) + "/kestrel/models";
    const char* h = std::getenv("HOME");
    if (!h) h = std::getenv("USERPROFILE");
    return std::string(h ? h : ".") + "/.local/share/kestrel/models";
}

ModelInfo verify_model(const std::string& dir) {
    ModelInfo m;
    m.dir = dir;
    fs::path mj = fs::path(dir) / "model.json";
    if (!fs::exists(mj)) {
        m.name = fs::path(dir).filename().string();
        m.problem = "no model.json";
        return m;
    }
    std::ifstream f(mj);
    auto j = nlohmann::json::parse(f, nullptr, false);
    if (j.is_discarded()) {
        m.problem = "model.json is not JSON";
        return m;
    }
    m.name = j.value("name", ""), m.version = j.value("version", ""), m.game = j.value("game", "");
    m.about = j.value("about", ""), m.players = j.value("players", 0);
    if (!j.contains("files") || j["files"].empty()) {
        m.problem = "model.json lists no files";
        return m;
    }
    for (auto& [file, want] : j["files"].items()) {
        if (file.find("..") != std::string::npos || file.find('/') != std::string::npos) {
            m.problem = "bad file name in model.json: " + file;
            return m;
        }
        fs::path p = fs::path(dir) / file;
        if (!fs::exists(p)) {
            m.problem = "missing " + file;
            return m;
        }
        if (sha256_file(p.string()) != want.get<std::string>()) {
            m.problem = file + " does not match its sha256";
            return m;
        }
    }
    m.verified = true;
    return m;
}

std::vector<ModelInfo> list_models(const std::string& dir) {
    std::vector<ModelInfo> out;
    std::error_code ec;
    for (auto& e : fs::directory_iterator(dir, ec))
        if (e.is_directory() && fs::exists(e.path() / "model.json")) out.push_back(verify_model(e.path().string()));
    std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.name < b.name; });
    return out;
}

ModelInfo install_model(const std::string& package, const std::string& dir) {
    if (!fs::exists(package)) throw std::runtime_error("no " + package);
    std::string side = package + ".sha256";
    if (!fs::exists(side)) throw std::runtime_error("no " + side + ": a package is installed only with its checksum");
    if (sha256_file(package) != read_sidecar(side)) throw std::runtime_error(package + " does not match its .sha256");
    std::mt19937_64 rng(std::random_device{}());
    fs::path tmp = fs::temp_directory_path() / ("kestrel-model-" + std::to_string(rng()));
    fs::create_directories(tmp);
    std::string cmd = "tar -xzf " + quote(package) + " -C " + quote(tmp.string());
    if (std::system(cmd.c_str()) != 0) {
        fs::remove_all(tmp);
        throw std::runtime_error("could not unpack " + package);
    }
    fs::path inner;
    for (auto& e : fs::directory_iterator(tmp))
        if (e.is_directory()) inner = e.path();
    if (inner.empty()) {
        fs::remove_all(tmp);
        throw std::runtime_error(package + " holds no model folder");
    }
    auto m = verify_model(inner.string());
    if (!m.verified) {
        fs::remove_all(tmp);
        throw std::runtime_error(package + ": " + m.problem);
    }
    if (m.name.empty() || m.name.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-") !=
                              std::string::npos) {
        fs::remove_all(tmp);
        throw std::runtime_error("bad model name in " + package);
    }
    fs::create_directories(dir);
    fs::path dest = fs::path(dir) / m.name;
    std::error_code ec;
    fs::remove_all(dest, ec);
    fs::rename(inner, dest, ec);
    if (ec) {
        fs::copy(inner, dest, fs::copy_options::recursive);
    }
    fs::remove_all(tmp, ec);
    m.dir = dest.string();
    return m;
}

}  // namespace kestrel
