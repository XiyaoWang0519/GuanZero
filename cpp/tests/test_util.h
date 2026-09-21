// A dependency-free test harness. Register with TEST(name) { ... }.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace gdtest {

struct Case { const char* name; std::function<void()> fn; };
inline std::vector<Case>& registry() { static std::vector<Case> r; return r; }
inline int& failures() { static int f = 0; return f; }
inline const char*& current() { static const char* c = ""; return c; }

struct Register {
  Register(const char* name, std::function<void()> fn) { registry().push_back({name, fn}); }
};

inline void fail(const char* file, int line, const std::string& msg) {
  std::fprintf(stderr, "  FAIL %s:%d in %s: %s\n", file, line, current(), msg.c_str());
  ++failures();
}

inline int run_all() {
  int ran = 0;
  for (auto& c : registry()) {
    current() = c.name;
    const int before = failures();
    c.fn();
    ++ran;
    if (failures() != before) std::fprintf(stderr, "  [ %s ]\n", c.name);
  }
  std::fprintf(stderr, "%d test cases, %d failed assertions\n", ran, failures());
  return failures() == 0 ? 0 : 1;
}

}  // namespace gdtest

#define GD_CONCAT_(a, b) a##b
#define GD_CONCAT(a, b) GD_CONCAT_(a, b)
#define TEST(name)                                                        \
  static void GD_CONCAT(gd_test_fn_, __LINE__)();                         \
  static ::gdtest::Register GD_CONCAT(gd_test_reg_, __LINE__)(            \
      name, GD_CONCAT(gd_test_fn_, __LINE__));                            \
  static void GD_CONCAT(gd_test_fn_, __LINE__)()

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) ::gdtest::fail(__FILE__, __LINE__, "CHECK(" #cond ")");   \
  } while (0)

#define CHECK_EQ(a, b)                                                    \
  do {                                                                    \
    auto&& gd_a_ = (a);                                                   \
    auto&& gd_b_ = (b);                                                   \
    if (!(gd_a_ == gd_b_))                                                \
      ::gdtest::fail(__FILE__, __LINE__,                                  \
                     std::string("CHECK_EQ(" #a ", " #b ") -> ") +        \
                         std::to_string(gd_a_) + " vs " +                 \
                         std::to_string(gd_b_));                          \
  } while (0)
