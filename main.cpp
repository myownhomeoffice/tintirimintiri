
// -----------------------------
// Examples (compile-time guard)
// -----------------------------
// Define RB_TREE_MAP_NODEHANDLE_EXAMPLE_MAIN to compile and run examples demonstrating node_handle behavior.
//
// Example 1: simple extract/insert within same allocator (node reuse).
// Example 2: incompatible allocator example: a small stateful allocator is used so insert(node_handle)
//            must reallocate into the target container and deallocate the source node using source allocator.
// To build examples compile with: -DRB_TREE_MAP_NODEHANDLE_EXAMPLE_MAIN

#ifdef RB_TREE_MAP_NODEHANDLE_EXAMPLE_MAIN
#include <string>

// Simple stateful allocator used to demonstrate allocator equality/inequality behavior.
// It is minimal and only intended for demonstration — not production ready.
template <typename T>
struct DemoAllocator {
    using value_type = T;
    int id; // allocator identity; allocators with different id are considered unequal
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

int main() {
    using Map = RBTreeMapNH<int, std::string, std::less<int>, std::allocator<std::pair<const int, std::string>>>;
    {
        std::cout << "Example 1: extract/insert with same allocator (node reuse expected)\n";
        Map m;
        m.insert({1, "one"});
        m.insert({2, "two"});
        m.insert({3, "three"});

        // extract key 2 -> get node_handle owning the raw Node* (allocated by m)
        auto nh = m.extract(2);
        std::cout << "m contains 2 after extract? " << (m.contains(2) ? "yes" : "no") << "\n";

        // insert node into another container with same allocator type and same allocator state (std::allocator)
        Map m2;
        auto pr = m2.insert(std::move(nh));
        std::cout << "Inserted into m2? " << (pr.second ? "yes (node reused)" : "no") << "\n";
        std::cout << "m2 contains 2? " << (m2.contains(2) ? "yes" : "no") << "\n";
    }

    // Example 2: show behavior when node allocator differs (reallocate+deallocate path)
    {
        std::cout << "\nExample 2: incompatible allocator (DemoAllocator with different id)\n";
        using DemoMapA = RBTreeMapNH<int, std::string, std::less<int>, DemoAllocator<std::pair<const int, std::string>>>;
        using DemoMapB = RBTreeMapNH<int, std::string, std::less<int>, DemoAllocator<std::pair<const int, std::string>>>;
        DemoMapA a((std::less<int>()), DemoAllocator<std::pair<const int, std::string>>(1));
        DemoMapB b((std::less<int>()), DemoAllocator<std::pair<const int, std::string>>(2));

        a.insert({10, "ten"});
        a.insert({20, "twenty"});

        // extract from a (node allocated using allocator id=1)
        auto nh = a.extract(10);
        std::cout << "a contains 10 after extract? " << (a.contains(10) ? "yes" : "no") << "\n";

        // insert into b (allocator ids differ -> node memory cannot be reused).
        // insertion will allocate a fresh node in b's allocator and free the original node using its allocator.
        auto pr = b.insert(std::move(nh));
        std::cout << "Inserted into b? " << (pr.second ? "yes (copied/moved into b's allocator)" : "no") << "\n";
        std::cout << "b contains 10? " << (b.contains(10) ? "yes" : "no") << "\n";
    }

    return 0;
}
