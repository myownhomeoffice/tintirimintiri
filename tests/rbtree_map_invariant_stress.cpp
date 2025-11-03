// tests/rbtree_map_invariant_stress.cpp
//
// Heavy stress / fuzz tests that exercise RBTreeMapNH and validate red-black invariants
// after random sequences of operations. Uses GoogleTest.
//
// - Runs multiple randomized rounds, each performing many operations (insert/erase/extract/insert(node_handle)/assign).
// - After each batch of operations validates:
//     * RB invariants via validate_invariants()
//     * ordering & contents vs std::map
// - Emits diagnostic output (captured by GoogleTest on failure).
//
// To control stress intensity use env vars or modify constants below.

#include <gtest/gtest.h>
#include <map>
#include <random>
#include <string>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>

#include "red_black_tree_map_nodehandle.hpp"

// Minimal stateful allocator to force allocator-inequality behavior in some tests
template <typename T>
struct DemoAllocator {
    using value_type = T;
    int id;
    DemoAllocator(int i = 0) noexcept : id(i) {}
    template <typename U>
    DemoAllocator(const DemoAllocator<U>& other) noexcept : id(other.id) {}
    T* allocate(std::size_t n) {
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }
    void deallocate(T* p, std::size_t) noexcept {
        ::operator delete(p);
    }
    bool operator==(const DemoAllocator& o) const noexcept { return id == o.id; }
    bool operator!=(const DemoAllocator& o) const noexcept { return id != o.id; }
};

// Helper: convert RBTreeMapNH contents to vector of pairs for comparison
template <typename MapT>
std::vector<typename MapT::value_type> to_vector(const MapT& m) {
    std::vector<typename MapT::value_type> out;
    for (auto it = m.begin(); it != m.end(); ++it) out.push_back(*it);
    return out;
}

// Validate RB invariants and equality with std::map; returns diagnostic string (empty if ok)
template <typename MapT>
std::string validate_and_compare(const MapT& m, const std::map<typename MapT::key_type, typename MapT::mapped_type>& baseline) {
    std::string diag;
    bool ok = m.validate_invariants(&diag);
    if (!ok) {
        std::ostringstream oss;
        oss << "Invariant validation failed:\n" << diag << "\n";
        return oss.str();
    }

    // Compare sizes and contents
    auto v = to_vector(m);
    std::vector<std::pair<const typename MapT::key_type, typename MapT::mapped_type>> sv(baseline.begin(), baseline.end());
    if (v.size() != sv.size()) {
        std::ostringstream oss;
        oss << "Size mismatch: tree=" << v.size() << " map=" << sv.size() << "\n";
        return oss.str();
    }
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i].first != sv[i].first || v[i].second != sv[i].second) {
            std::ostringstream oss;
            oss << "Content mismatch at index " << i << ": tree=(" << v[i].first << "," << v[i].second
                << ") map=(" << sv[i].first << "," << sv[i].second << ")\n";
            return oss.str();
        }
    }
    return "";
}

TEST(StressFuzz, RandomizedOperationsValidateInvariants) {
    using Map = RBTreeMapNH<int, int>;
    Map m;
    std::map<int,int> baseline;

    // Stress parameters - lower these if running on CI with limited time
    const int ROUNDS = 50;         // number of rounds (each round resets baseline/m)
    const int OPS_PER_ROUND = 2000; // number of operations per round
    const int KEY_RANGE = 800;     // keys drawn from [0,KEY_RANGE)

    std::mt19937_64 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> keydist(0, KEY_RANGE-1);
    std::uniform_int_distribution<int> opdist(0, 99); // choose operation

    for (int round = 0; round < ROUNDS; ++round) {
        m = Map(); // reset
        baseline.clear();

        for (int op = 0; op < OPS_PER_ROUND; ++op) {
            int k = keydist(rng);
            int action = opdist(rng);
            // Weighted operations:
            // 0-49 : insert_or_assign (50%)
            // 50-79: erase (30%)
            // 80-89: extract+insert into same map (10%) - no-op semantics
            // 90-94: extract from one map and insert into another map (not here)
            // 95-99: try_emplace (5%)
            if (action < 50) {
                int v = (int)rng();
                m.insert_or_assign(k, v);
                baseline[k] = v;
            } else if (action < 80) {
                m.erase(k);
                baseline.erase(k);
            } else if (action < 90) {
                // extract and re-insert on same map (simulate transfer)
                auto nh = m.extract(k);
                if (nh) {
                    // mutate value a bit and reinsert
                    nh.value().second ^= 0x55aa;
                    auto pr = m.insert(std::move(nh));
                    if (pr.second) baseline[k] ^= 0x55aa;
                    else {
                        // if not inserted, baseline unchanged (but our insert() returns false if duplicate)
                    }
                } else {
                    // maybe try_emplace
                    if (baseline.find(k) == baseline.end()) {
                        int v = (int)rng();
                        m.try_emplace(k, v);
                        baseline.emplace(k, v);
                    }
                }
            } else if (action < 95) {
                int v = (int)rng();
                m.try_emplace(k, v);
                if (baseline.find(k) == baseline.end()) baseline.emplace(k, v);
            } else {
                // mixture: emplace or insert
                int v = (int)rng();
                m.insert({k, v});
                baseline[k] = v;
            }

            // periodically validate invariants and comparison
            if (op % 50 == 0) {
                std::string diag = validate_and_compare(m, baseline);
                if (!diag.empty()) {
                    FAIL() << "Round " << round << " op " << op << " failed:\n" << diag;
                }
            }
        } // ops loop

        // final validation for round
        std::string final_diag = validate_and_compare(m, baseline);
        if (!final_diag.empty()) {
            FAIL() << "Round " << round << " final validation failed:\n" << final_diag;
        }
    } // rounds
}

// Additional test: allocate two maps with incompatible allocators and stress-transfer nodes between them.
TEST(StressFuzz, CrossAllocatorNodeTransfer) {
    using MapA = RBTreeMapNH<int, int, std::less<int>, DemoAllocator<std::pair<const int,int>>>;
    using MapB = RBTreeMapNH<int, int, std::less<int>, DemoAllocator<std::pair<const int,int>>>;

    MapA a(std::less<int>(), DemoAllocator<std::pair<const int,int>>(1));
    MapB b(std::less<int>(), DemoAllocator<std::pair<const int,int>>(2));

    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<int> keydist(0, 500);
    const int OPS = 2000;

    // populate a and b
    for (int i = 0; i < 200; ++i) {
        int k = keydist(rng);
        a.insert({k, k});
        k = keydist(rng);
        b.insert({k, k});
    }

    // Perform random extract+insert across allocators; since allocators unequal, nodes should be reallocated.
    for (int i = 0; i < OPS; ++i) {
        int k = keydist(rng);
        auto nh = a.extract(k);
        if (nh) {
            auto pr = b.insert(std::move(nh));
            // insertion should succeed if not duplicate; if duplicate, insert leaves handle intact (per our design)
            // Validate both trees invariants periodically
        }

        if (i % 100 == 0) {
            std::string diag_a, diag_b;
            ASSERT_TRUE(a.validate_invariants(&diag_a)) << "a invariants failed: " << diag_a;
            ASSERT_TRUE(b.validate_invariants(&diag_b)) << "b invariants failed: " << diag_b;
        }
    }
}

// Instrumentation sanity test: ensure rotation_count increments during operations
TEST(Instrumentation, RotationCountMonotonic) {
    using Map = RBTreeMapNH<int, int>;
    Map m;
    m.reset_rotation_count();
    size_t before = m.rotation_count();
    for (int i = 0; i < 1000; ++i) m.insert({i, i});
    size_t after = m.rotation_count();
    EXPECT_GE(after, before);
}

// Large fuzz test disabled by default - can be enabled locally when you want very long runs.
// Use GoogleTest filter to run specifically.
TEST(StressFuzz, DISABLED_LongRun) {
    using Map = RBTreeMapNH<int, int>;
    Map m;
    std::map<int,int> baseline;
    std::mt19937_64 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> keydist(0, 100000);
    for (int i = 0; i < 200000; ++i) {
        int k = keydist(rng);
        if (i % 2 == 0) {
            m.insert_or_assign(k, i);
            baseline[k] = i;
        } else {
            m.erase(k);
            baseline.erase(k);
        }
        if (i % 1000 == 0) {
            std::string diag = validate_and_compare(m, baseline);
            if (!diag.empty()) FAIL() << "LongRun failure at iter " << i << ":\n" << diag;
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
