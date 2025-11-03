// Comprehensive unit tests for RBTreeMapNH using GoogleTest
// Place the RBTreeMapNH header in include/ (or adjust RBTREE_MAP_INCLUDE_DIR in CMake)
#include <gtest/gtest.h>
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <sstream>

// Include the header for the implementation under test.
// Adjust include path as necessary.
#include "red_black_tree_map_nodehandle.hpp"

// Minimal demo stateful allocator for tests (to force unequal allocators)
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
    for (auto it = m.begin(); it != m.end(); ++it) {
        out.push_back(*it);
    }
    return out;
}

// Compare RBTreeMapNH contents with std::map counterpart
template <typename Key, typename T>
void expect_equal_to_std_map(const RBTreeMapNH<Key, T>& rbt, const std::map<Key, T>& sm) {
    ASSERT_EQ(rbt.size(), sm.size());
    auto rv = to_vector(rbt);
    std::vector<std::pair<const Key, T>> sv(sm.begin(), sm.end());
    ASSERT_EQ(rv.size(), sv.size());
    for (size_t i = 0; i < rv.size(); ++i) {
        EXPECT_EQ(rv[i].first, sv[i].first);
        EXPECT_EQ(rv[i].second, sv[i].second);
    }
}

TEST(BasicOperations, InsertFindErase) {
    using Map = RBTreeMapNH<int, std::string>;
    Map m;
    EXPECT_TRUE(m.empty());
    m.insert({10, "ten"});
    m.insert({5, "five"});
    m.insert({20, "twenty"});
    EXPECT_EQ(m.size(), 3u);

    auto it = m.find(10);
    ASSERT_NE(it, m.end());
    EXPECT_EQ(it->second, "ten");

    // insert_or_assign
    m.insert_or_assign(10, std::string("TEN"));
    auto it10 = m.find(10);
    ASSERT_NE(it10, m.end());
    EXPECT_EQ(it10->second, "TEN");

    // try_emplace (existing)
    auto pr1 = m.try_emplace(10, "ignored");
    EXPECT_FALSE(pr1.second);
    // try_emplace (new)
    auto pr2 = m.try_emplace(15, "fifteen");
    EXPECT_TRUE(pr2.second);
    EXPECT_EQ(m.size(), 4u);

    // erase by key
    EXPECT_EQ(m.erase(5), 1u);
    EXPECT_EQ(m.size(), 3u);
    EXPECT_EQ(m.contains(5), false);

    // erase by iterator
    auto it20 = m.find(20);
    ASSERT_NE(it20, m.end());
    m.erase(it20);
    EXPECT_EQ(m.contains(20), false);
    EXPECT_EQ(m.size(), 2u);
}

TEST(IteratorsAndBounds, LowerUpperEqualRange) {
    using Map = RBTreeMapNH<int, int>;
    Map m;
    for (int i = 0; i < 10; ++i) m.insert({i*2, i});
    EXPECT_EQ(m.size(), 10u);

    auto lb = m.lower_bound(6); // first >= 6 -> key 6
    ASSERT_NE(lb, m.end());
    EXPECT_EQ(lb->first, 6);

    auto ub = m.upper_bound(6); // first > 6 -> key 8
    ASSERT_NE(ub, m.end());
    EXPECT_EQ(ub->first, 8);

    auto er = m.equal_range(6);
    EXPECT_EQ(er.first->first, 6);
    EXPECT_EQ(er.second->first, 8);
}

TEST(NodeHandle, ExtractInsertSameAllocatorReuse) {
    using Map = RBTreeMapNH<int, std::string>;
    Map a;
    a.insert({1, "one"});
    a.insert({2, "two"});
    a.insert({3, "three"});

    EXPECT_TRUE(a.contains(2));
    auto nh = a.extract(2);
    EXPECT_FALSE(a.contains(2));
    EXPECT_TRUE(nh); // handle should be non-empty

    Map b;
    auto pr = b.insert(std::move(nh));
    EXPECT_TRUE(pr.second); // inserted
    EXPECT_EQ(b.size(), 1u);
    EXPECT_TRUE(b.contains(2));
}

TEST(NodeHandle, ExtractInsertIncompatibleAllocatorRealloc) {
    using MapA = RBTreeMapNH<int, std::string, std::less<int>, DemoAllocator<std::pair<const int, std::string>>>;
    using MapB = RBTreeMapNH<int, std::string, std::less<int>, DemoAllocator<std::pair<const int, std::string>>>;

    MapA a(std::less<int>(), DemoAllocator<std::pair<const int, std::string>>(1));
    MapB b(std::less<int>(), DemoAllocator<std::pair<const int, std::string>>(2));

    a.insert({10, "ten"});
    a.insert({20, "twenty"});

    auto nh = a.extract(10);
    EXPECT_TRUE(nh);
    EXPECT_FALSE(a.contains(10));

    auto pr = b.insert(std::move(nh));
    EXPECT_TRUE(pr.second);
    EXPECT_TRUE(b.contains(10));
    EXPECT_FALSE(nh); // handle should have been consumed
}

TEST(MergeAndMove, MergeAndMoveAssignmentBehavior) {
    using Map = RBTreeMapNH<int, std::string>;
    Map a;
    Map b;
    for (int i = 0; i < 20; ++i) {
        if (i % 2 == 0) a.insert({i, "a" + std::to_string(i)});
        else b.insert({i, "b" + std::to_string(i)});
    }

    size_t total = a.size() + b.size();
    a.merge(b);
    EXPECT_EQ(a.size(), total);
    // b may be empty or unchanged depending on merge implementation; in our implementation, merge extracts nodes -> b empty
    EXPECT_TRUE(b.empty() || b.size() <= 1u);

    // test move assignment with same allocator: nodes should be moved/stealed
    Map c;
    c = std::move(a);
    EXPECT_EQ(c.size(), total);
}

TEST(RandomizedPropertyComparison, CompareToStdMap) {
    using Key = int;
    using Val = std::string;
    using Map = RBTreeMapNH<Key, Val>;

    Map rbt;
    std::map<Key, Val> sm;

    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<int> keydist(0, 200);

    for (int i = 0; i < 1000; ++i) {
        int k = keydist(rng);
        std::string s = "v" + std::to_string(i);
        if (i % 3 == 0) {
            rbt.insert({k, s});
            sm.insert({k, s});
        } else if (i % 3 == 1) {
            rbt.insert_or_assign(k, s);
            sm[k] = s;
        } else {
            rbt.try_emplace(k, s);
            if (sm.find(k) == sm.end()) sm.emplace(k, s);
        }
    }

    // convert both to vector and compare
    auto rv = to_vector(rbt);
    std::vector<std::pair<const Key, Val>> sv(sm.begin(), sm.end());
    ASSERT_EQ(rv.size(), sv.size());
    for (size_t i = 0; i < rv.size(); ++i) {
        EXPECT_EQ(rv[i].first, sv[i].first);
        EXPECT_EQ(rv[i].second, sv[i].second);
    }
}

TEST(EdgeCases, ExtractNonexistentAndEraseRange) {
    using Map = RBTreeMapNH<int, int>;
    Map m;
    for (int i = 0; i < 10; ++i) m.insert({i, i*10});
    auto nh = m.extract(100); // not present
    EXPECT_TRUE(nh.empty());

    // erase a range
    auto it1 = m.lower_bound(3);
    auto it2 = m.upper_bound(6);
    m.erase(it1, it2); // removes keys 3,4,5,6?
    // validate tree properties by comparing to std::map baseline
    std::map<int,int> baseline;
    for (int i = 0; i < 10; ++i) baseline.emplace(i, i*10);
    baseline.erase(3);
    baseline.erase(4);
    baseline.erase(5);
    baseline.erase(6);

    expect_equal_to_std_map(m, baseline);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
