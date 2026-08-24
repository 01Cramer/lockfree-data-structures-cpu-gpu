// Energy measurement for the CPU benchmarks via the Linux powercap (RAPL)
// interface.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#if defined(CPU_ENABLE_RAPL) && defined(__linux__)
#define CPU_RAPL_ACTIVE 1
#else
#define CPU_RAPL_ACTIVE 0
#endif

#if CPU_RAPL_ACTIVE
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace cpu {

// The powercap driver refreshes the counters on roughly a 1 ms cadence; a
// window shorter than this reads as zero or as one quantum of noise, so the
// harness should refuse to report energy for it rather than print a small
// number.
inline constexpr long raplUpdateMicros = 1000;

// A snapshot of every discovered counter, in domainNames() order. Obtain one
// from makeReading() so that sample() never allocates.
using EnergyReading = std::vector<std::uint64_t>;

class EnergyMeter {
public:
  // Discovers and opens every RAPL domain; costs a handful of open() calls, so
  // do it once outside any timed region.
  EnergyMeter() { discover(); }

  ~EnergyMeter() {
#if CPU_RAPL_ACTIVE
    for (Domain &domain : m_domains) {
      if (domain.fd >= 0) {
        ::close(domain.fd);
      }
    }
#endif
  }

  EnergyMeter(const EnergyMeter &) = delete;
  EnergyMeter &operator=(const EnergyMeter &) = delete;
  EnergyMeter(EnergyMeter &&) = delete;
  EnergyMeter &operator=(EnergyMeter &&) = delete;

  bool available() const { return !m_domains.empty(); }

  std::size_t domainCount() const { return m_domains.size(); }

  // Stable label for column i, e.g. "package-0" or "package-0/dram".
  const std::string &domainName(std::size_t i) const {
    return m_domains[i].name;
  }

  // Counter wrap point for column i, in microjoules. Divide by the expected
  // draw to get the longest window that can be measured unambiguously; see
  // joulesBetween().
  std::uint64_t domainRangeMicroJoules(std::size_t i) const {
    return m_domains[i].rangeMicroJoules;
  }

  EnergyReading makeReading() const {
    return EnergyReading(m_domains.size(), 0);
  }

  void sample(EnergyReading &out) const {
#if CPU_RAPL_ACTIVE
    for (std::size_t i = 0; i < m_domains.size(); ++i) {
      out[i] = readOpenFd(m_domains[i].fd, m_domains[i].name.c_str());
    }
#else
    (void)out;
#endif
  }

  // Per-domain joules between two samples.
  std::vector<double> joulesBetween(const EnergyReading &before,
                                    const EnergyReading &after,
                                    bool *decreased = nullptr) const {
    std::vector<double> joules(m_domains.size(), 0.0);
    if (decreased != nullptr) {
      *decreased = false;
    }
    for (std::size_t i = 0; i < m_domains.size(); ++i) {
      const std::uint64_t start = before[i];
      const std::uint64_t end = after[i];
      if (end >= start) {
        joules[i] = static_cast<double>(end - start) * 1e-6;
        continue;
      }

      if (decreased != nullptr) {
        *decreased = true;
      }
      joules[i] = static_cast<double>(cycleMicroJoules(i) - start + end) * 1e-6;
    }
    return joules;
  }

private:
  struct Domain {
    std::string name;
    std::uint64_t rangeMicroJoules = 0;
    int fd = -1;
  };

  std::vector<Domain> m_domains;

  // max_energy_range_uj is a max, not a period; range/0xffffffff is the unit.
  std::uint64_t cycleMicroJoules(std::size_t i) const {
    const std::uint64_t range = m_domains[i].rangeMicroJoules;
    constexpr std::uint64_t maxRawCounter = 0xffffffffULL;
    return range + (range / maxRawCounter);
  }

#if CPU_RAPL_ACTIVE
  static constexpr const char *kPowercapRoot = "/sys/class/powercap/intel-rapl";

  void discover() {
    for (int package = 0;; ++package) {
      char base[64];
      std::snprintf(base, sizeof(base), "%s:%d", kPowercapRoot, package);

      std::string label;
      if (!readName(base, label)) {
        break;
      }
      addDomain(base, label);

      for (int nested = 0;; ++nested) {
        char child[96];
        std::snprintf(child, sizeof(child), "%s:%d", base, nested);

        std::string childLabel;
        if (!readName(child, childLabel)) {
          break;
        }
        addDomain(child, label + "/" + childLabel);
      }
    }

    if (m_domains.empty()) {
      failUnavailable();
    }
  }

  void addDomain(const char *directory, const std::string &label) {
    Domain domain;
    domain.name = label;
    domain.fd = openAt(directory, "energy_uj");
    if (domain.fd < 0) {
      failPermissions(directory, errno);
    }
    domain.rangeMicroJoules = readValueAt(directory, "max_energy_range_uj");
    m_domains.push_back(std::move(domain));
  }

  static int openAt(const char *directory, const char *file) {
    char path[160];
    std::snprintf(path, sizeof(path), "%s/%s", directory, file);
    return ::open(path, O_RDONLY | O_CLOEXEC);
  }

  static bool readName(const char *directory, std::string &out) {
    const int fd = openAt(directory, "name");
    if (fd < 0) {
      return false;
    }
    char buf[64];
    const ssize_t n = ::pread(fd, buf, sizeof(buf) - 1, 0);
    ::close(fd);
    if (n <= 0) {
      return false;
    }
    buf[n] = '\0';
    out.assign(buf);
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) {
      out.pop_back();
    }
    return !out.empty();
  }

  static std::uint64_t readValueAt(const char *directory, const char *file) {
    const int fd = openAt(directory, file);
    if (fd < 0) {
      failPermissions(directory, errno);
    }
    const std::uint64_t value = readOpenFd(fd, directory);
    ::close(fd);
    return value;
  }

  static std::uint64_t readOpenFd(int fd, const char *what) {
    char buf[32];
    const ssize_t n = ::pread(fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0) {
      failRead(what, errno);
    }
    buf[n] = '\0';
    return std::strtoull(buf, nullptr, 10);
  }

  [[noreturn]] static void failUnavailable() {
    std::fprintf(stderr,
                 "cpu::EnergyMeter: built with ENABLE_RAPL but found no RAPL "
                 "domains under %s. Either the CPU is not Intel, the kernel "
                 "lacks the intel_rapl_common/intel_rapl_msr modules, or this "
                 "is a VM or WSL, where the interface is not exposed.\n",
                 kPowercapRoot);
    std::abort();
  }

  [[noreturn]] static void failPermissions(const char *directory, int err) {
    std::fprintf(stderr,
                 "cpu::EnergyMeter: cannot read the counters in %s (%s). Since "
                 "the Platypus disclosure these are root-only. Run as root, or "
                 "grant read access with:\n"
                 "  chmod -R a+r /sys/class/powercap/intel-rapl\n",
                 directory, std::strerror(err));
    std::abort();
  }

  [[noreturn]] static void failRead(const char *what, int err) {
    std::fprintf(stderr,
                 "cpu::EnergyMeter: failed to read the counter for domain %s "
                 "(%s). Any energy figure from this run would be invalid.\n",
                 what, std::strerror(err));
    std::abort();
  }
#else
  void discover() {}
#endif
};

} // namespace cpu
