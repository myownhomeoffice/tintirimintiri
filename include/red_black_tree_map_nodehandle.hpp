// red_black_tree_map_nodehandle.hpp
// Red-Black Tree map with true node_handle (preserves Node* and allocator state) and allocator propagation.
// Extended with debug instrumentation and invariant validation helpers.
//
// - C++17 header-only.
// - Adds:
//     * rotation_count_ instrumentation (incremented on left/right rotate)
//     * validate_invariants(std::string* out = nullptr) const to check red-black properties:
//         - root is black
//         - no consecutive red nodes
//         - every path from node to NIL has the same black-height
//         - binary-search-tree ordering (left < node < right)
//         - parent/child link consistency
//       Returns true if invariants hold, false otherwise. If `out` provided, appends human-readable diagnostics.
//     * rotation_count() / reset_rotation_count() accessors.
//
// NOTE: This header is based on the RBTreeMapNH implementation you already have; the debug helpers are
// implemented as public const methods that traverse the internal structure (they access private members).
//
// Usage (in tests):
//   RBTreeMapNH<int, std::string> m;
//   ... do operations ...
//   std::string diag;
//   ASSERT_TRUE(m.validate_invariants(&diag)) << diag;
//
// Keep the file in include/ or adjust include paths accordingly.

#ifndef RED_BLACK_TREE_MAP_NODEHANDLE_HPP
#define RED_BLACK_TREE_MAP_NODEHANDLE_HPP

#include <memory>
#include <utility>
#include <functional>
#include <iterator>
#include <type_traits>
#include <cassert>
#include <limits>
#include <iostream>
#include <string>

template <
    typename Key,
    typename T,
    typename Compare = std::less<Key>,
    typename Alloc = std::allocator<std::pair<const Key, T>>
>
class RBTreeMapNH {
public:
    // STL-like typedefs
    using key_type        = Key;
    using mapped_type     = T;
    using value_type      = std::pair<const Key, T>;
    using key_compare     = Compare;
    using allocator_type  = Alloc;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

private:
    enum Color { RED = 1, BLACK = 0 };

    // Internal node structure. value holds pair<const Key,T>.
    struct Node {
        value_type value;
        Color color;
        Node* left;
        Node* right;
        Node* parent;
        Node(const value_type& v, Color c, Node* nil)
            : value(v), color(c), left(nil), right(nil), parent(nil) {}
        Node(value_type&& v, Color c, Node* nil)
            : value(std::move(v)), color(c), left(nil), right(nil), parent(nil) {}
    };

    // rebind a node allocator from the container allocator
    using AllocTraits = std::allocator_traits<allocator_type>;
    using NodeAlloc = typename AllocTraits::template rebind_alloc<Node>;
    using NodeAllocTraits = std::allocator_traits<NodeAlloc>;

public:
    // iterator and const_iterator (bidirectional)
    class const_iterator;
    class iterator {
        friend class RBTreeMapNH;
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type        = RBTreeMapNH::value_type;
        using reference         = value_type&;
        using pointer           = value_type*;
        using difference_type   = RBTreeMapNH::difference_type;

        iterator() noexcept : node_(nullptr), tree_(nullptr) {}
        reference operator*() const { return node_->value; }
        pointer operator->() const { return &node_->value; }

        iterator& operator++() { node_ = tree_->successor(node_); return *this; }
        iterator operator++(int) { iterator tmp = *this; ++(*this); return tmp; }

        iterator& operator--() {
            if (node_ == tree_->NIL_) { node_ = tree_->maximum(tree_->root_); }
            else node_ = tree_->predecessor(node_);
            return *this;
        }
        iterator operator--(int) { iterator tmp = *this; --(*this); return tmp; }

        bool operator==(const iterator& o) const { return node_ == o.node_; }
        bool operator!=(const iterator& o) const { return node_ != o.node_; }

    private:
        Node* node_;
        const RBTreeMapNH* tree_;
        iterator(Node* n, const RBTreeMapNH* t) noexcept : node_(n), tree_(t) {}
    };

    class const_iterator {
        friend class RBTreeMapNH;
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type        = RBTreeMapNH::value_type;
        using reference         = const value_type&;
        using pointer           = const value_type*;
        using difference_type   = RBTreeMapNH::difference_type;

        const_iterator() noexcept : node_(nullptr), tree_(nullptr) {}
        const_iterator(const iterator& it) noexcept : node_(it.node_), tree_(it.tree_) {}
        reference operator*() const { return node_->value; }
        pointer operator->() const { return &node_->value; }

        const_iterator& operator++() { node_ = tree_->successor(node_); return *this; }
        const_iterator operator++(int) { const_iterator tmp = *this; ++(*this); return tmp; }

        const_iterator& operator--() {
            if (node_ == tree_->NIL_) { node_ = tree_->maximum(tree_->root_); }
            else node_ = tree_->predecessor(node_);
            return *this;
        }
        const_iterator operator--(int) { const_iterator tmp = *this; --(*this); return tmp; }

        bool operator==(const const_iterator& o) const { return node_ == o.node_; }
        bool operator!=(const const_iterator& o) const { return node_ != o.node_; }

    private:
        const Node* node_;
        const RBTreeMapNH* tree_;
        const_iterator(const Node* n, const RBTreeMapNH* t) noexcept : node_(n), tree_(t) {}
    };

    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    // true node_type/node_handle: stores raw Node* and Node allocator instance.
    class node_type {
    public:
        node_type() noexcept : node_ptr_(nullptr), empty_(true) {}
        node_type(Node* n, const NodeAlloc& a) noexcept : node_ptr_(n), alloc_(a), empty_(false) {}
        node_type(node_type&& o) noexcept
            : node_ptr_(o.node_ptr_), alloc_(std::move(o.alloc_)), empty_(o.empty_) { o.node_ptr_ = nullptr; o.empty_ = true; }
        node_type& operator=(node_type&& o) noexcept {
            if (this != &o) {
                release();
                node_ptr_ = o.node_ptr_;
                alloc_ = std::move(o.alloc_);
                empty_ = o.empty_;
                o.node_ptr_ = nullptr;
                o.empty_ = true;
            }
            return *this;
        }
        node_type(const node_type&) = delete;
        node_type& operator=(const node_type&) = delete;
        ~node_type() { release(); }

        bool empty() const noexcept { return empty_; }
        explicit operator bool() const noexcept { return !empty_; }

        value_type& value() & { assert(!empty_); return node_ptr_->value; }
        value_type&& value() && { assert(!empty_); return std::move(node_ptr_->value); }

        Node* release_node() noexcept { Node* tmp = node_ptr_; node_ptr_ = nullptr; empty_ = true; return tmp; }

        const NodeAlloc& node_alloc() const noexcept { return alloc_; }

    private:
        friend class RBTreeMapNH;
        Node* node_ptr_;
        NodeAlloc alloc_;
        bool empty_;

        void release() noexcept {
            if (node_ptr_) {
                NodeAllocTraits::destroy(alloc_, node_ptr_);
                NodeAllocTraits::deallocate(alloc_, node_ptr_, 1);
                node_ptr_ = nullptr;
                empty_ = true;
            }
        }
    };

    // constructors / destructor / assignment
    explicit RBTreeMapNH(const key_compare& comp = key_compare(), const allocator_type& alloc = allocator_type())
        : comp_(comp), alloc_(alloc), node_alloc_(NodeAlloc()), size_(0), rotation_count_(0)
    {
        NIL_ = NodeAllocTraits::allocate(node_alloc_, 1);
        NodeAllocTraits::construct(node_alloc_, NIL_, value_type(), BLACK, nullptr);
        NIL_->left = NIL_->right = NIL_->parent = NIL_;
        root_ = NIL_;
    }

    RBTreeMapNH(const RBTreeMapNH& other)
        : comp_(other.comp_), alloc_(AllocTraits::select_on_container_copy_construction(other.alloc_)),
          node_alloc_(NodeAlloc()), size_(0), rotation_count_(0)
    {
        NIL_ = NodeAllocTraits::allocate(node_alloc_, 1);
        NodeAllocTraits::construct(node_alloc_, NIL_, value_type(), BLACK, nullptr);
        NIL_->left = NIL_->right = NIL_->parent = NIL_;
        root_ = NIL_;
        for (const auto& p : other) insert(p);
    }

    RBTreeMapNH(RBTreeMapNH&& other) noexcept
        : comp_(std::move(other.comp_)), alloc_(std::move(other.alloc_)), node_alloc_(std::move(other.node_alloc_)),
          root_(other.root_), NIL_(other.NIL_), size_(other.size_), rotation_count_(other.rotation_count_)
    {
        other.NIL_ = nullptr;
        other.root_ = nullptr;
        other.size_ = 0;
        other.rotation_count_ = 0;
    }

    RBTreeMapNH& operator=(RBTreeMapNH&& other) noexcept {
        if (this == &other) return *this;

        using POCMA = typename NodeAllocTraits::propagate_on_container_move_assignment;
        if constexpr (POCMA::value) {
            clear();
            if (NIL_) {
                NodeAllocTraits::destroy(node_alloc_, NIL_);
                NodeAllocTraits::deallocate(node_alloc_, NIL_, 1);
            }
            comp_ = std::move(other.comp_);
            alloc_ = std::move(other.alloc_);
            node_alloc_ = std::move(other.node_alloc_);
            root_ = other.root_;
            NIL_ = other.NIL_;
            size_ = other.size_;
            rotation_count_ = other.rotation_count_;
            other.NIL_ = nullptr;
            other.root_ = nullptr;
            other.size_ = 0;
            other.rotation_count_ = 0;
        } else {
            bool allocs_equal = NodeAllocTraits::is_always_equal::value || (node_alloc_ == other.node_alloc_);
            if (allocs_equal) {
                clear();
                if (NIL_) {
                    NodeAllocTraits::destroy(node_alloc_, NIL_);
                    NodeAllocTraits::deallocate(node_alloc_, NIL_, 1);
                }
                comp_ = std::move(other.comp_);
                root_ = other.root_;
                NIL_ = other.NIL_;
                size_ = other.size_;
                rotation_count_ = other.rotation_count_;
                other.NIL_ = nullptr;
                other.root_ = nullptr;
                other.size_ = 0;
                other.rotation_count_ = 0;
            } else {
                for (auto it = other.begin(); it != other.end(); ) {
                    auto key = it->first;
                    auto nh = other.extract(key);
                    insert(std::move(nh));
                    it = other.begin();
                }
            }
        }
        return *this;
    }

    RBTreeMapNH& operator=(const RBTreeMapNH& other) {
        if (this != &other) {
            clear();
            comp_ = other.comp_;
            alloc_ = AllocTraits::select_on_container_copy_construction(other.alloc_);
            for (const auto& p : other) insert(p);
        }
        return *this;
    }

    ~RBTreeMapNH() {
        clear();
        if (NIL_) {
            NodeAllocTraits::destroy(node_alloc_, NIL_);
            NodeAllocTraits::deallocate(node_alloc_, NIL_, 1);
            NIL_ = nullptr;
        }
    }

    // capacity
    bool empty() const noexcept { return size_ == 0; }
    size_type size() const noexcept { return size_; }
    size_type max_size() const noexcept { return std::numeric_limits<size_type>::max() / sizeof(Node); }

    // iterators
    iterator begin() noexcept { return iterator(minimum(root_), this); }
    const_iterator begin() const noexcept { return const_iterator(minimum(root_), this); }
    const_iterator cbegin() const noexcept { return begin(); }

    iterator end() noexcept { return iterator(NIL_, this); }
    const_iterator end() const noexcept { return const_iterator(NIL_, this); }
    const_iterator cend() const noexcept { return end(); }

    reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
    const_reverse_iterator crbegin() const noexcept { return rbegin(); }

    reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
    const_reverse_iterator crend() const noexcept { return rend(); }

    // element access
    mapped_type& operator[](const key_type& k) {
        auto it_bool = try_emplace(k);
        return it_bool.first->second;
    }

    // modifiers: insert/emplace/erase (omitted here for brevity; same as prior implementation)
    // For the full implementation, refer to the prior RBTreeMapNH code (rotations, insert_fixup, erase, etc.).
    // ... (full implementation kept unchanged) ...

    // For brevity in this snippet, I'm including the crucial rotation functions and invariant helpers.
    // The rest of the implementation (insert/emplace/extract/erase/merge/etc.) should remain as previously provided.

    // Instrumentation accessors
    size_t rotation_count() const noexcept { return rotation_count_; }
    void reset_rotation_count() noexcept { rotation_count_ = 0; }

    // validate_invariants:
    // Walks the tree and checks standard red-black properties + BST ordering + parent/child integrity.
    // If 'out' provided, appends diagnostic messages.
    bool validate_invariants(std::string* out = nullptr) const {
        if (!out) {
            std::string dummy;
            return validate_invariants_impl(dummy, false);
        } else {
            return validate_invariants_impl(*out, true);
        }
    }

private:
    // members (core)
    Node* root_;
    Node* NIL_;
    key_compare comp_;
    allocator_type alloc_;
    NodeAlloc node_alloc_;
    size_type size_;

    // instrumentation: number of rotations performed (approximate, increments in rotate functions)
    size_t rotation_count_;

    // rotation helpers (we increment rotation_count_ here)
    void left_rotate(Node* x) {
        Node* y = x->right;
        x->right = y->left;
        if (y->left != NIL_) y->left->parent = x;
        y->parent = x->parent;
        if (x->parent == NIL_) root_ = y;
        else if (x == x->parent->left) x->parent->left = y;
        else x->parent->right = y;
        y->left = x;
        x->parent = y;
        ++rotation_count_;
    }

    void right_rotate(Node* y) {
        Node* x = y->left;
        y->left = x->right;
        if (x->right != NIL_) x->right->parent = y;
        x->parent = y->parent;
        if (y->parent == NIL_) root_ = x;
        else if (y == y->parent->right) y->parent->right = x;
        else y->parent->left = x;
        x->right = y;
        y->parent = x;
        ++rotation_count_;
    }

    // The other methods (insert_fixup, transplant, erase_fixup, etc.) are unchanged
    // and must call left_rotate()/right_rotate() so rotation_count_ is incremented.

    // utility accessors for iterators
    Node* minimum(Node* x) const {
        Node* cur = x;
        while (cur != NIL_ && cur->left != NIL_) cur = cur->left;
        return (cur == NIL_) ? NIL_ : cur;
    }

    Node* maximum(Node* x) const {
        Node* cur = x;
        while (cur != NIL_ && cur->right != NIL_) cur = cur->right;
        return (cur == NIL_) ? NIL_ : cur;
    }

    Node* successor(Node* x) const {
        if (x == NIL_) return NIL_;
        if (x->right != NIL_) return minimum(x->right);
        Node* y = x->parent;
        while (y != NIL_ && x == y->right) { x = y; y = y->parent; }
        return y;
    }

    Node* predecessor(Node* x) const {
        if (x == NIL_) return NIL_;
        if (x->left != NIL_) return maximum(x->left);
        Node* y = x->parent;
        while (y != NIL_ && x == y->left) { x = y; y = y->parent; }
        return y;
    }

    // find helpers
    Node* find_node(const key_type& key) {
        Node* x = root_;
        while (x != NIL_) {
            if (comp_(key, x->value.first)) x = x->left;
            else if (comp_(x->value.first, key)) x = x->right;
            else return x;
        }
        return NIL_;
    }

    Node* find_node_const(const key_type& key) const {
        Node* x = root_;
        while (x != NIL_) {
            if (comp_(key, x->value.first)) x = x->left;
            else if (comp_(x->value.first, key)) x = x->right;
            else return x;
        }
        return NIL_;
    }

    // Remove helpers and clear (omitted here; they remain the same as prior code)
    void deallocate_node(Node* n) {
        NodeAllocTraits::destroy(node_alloc_, n);
        NodeAllocTraits::deallocate(node_alloc_, n, 1);
    }

    // Recursively validate RB invariants:
    // Returns pair(valid, black_height). Appends diagnostics to diag if verbose==true.
    std::pair<bool,int> validate_node(const Node* node, const Node* parent, const Key* min_key, const Key* max_key, std::string& diag, bool verbose) const {
        // Base sentinel: treat NIL_ as black leaf with black-height 0
        if (node == NIL_) return { true, 0 };

        // parent-pointer check
        if (node->parent != parent) {
            if (verbose) diag += "Parent pointer mismatch for key " + key_to_string(node->value.first) + "\n";
            return { false, 0 };
        }

        // BST order check: node->value.first must be > min_key and < max_key if bounds provided
        if (min_key && comp_(node->value.first, *min_key) == false && !comp_(*min_key, node->value.first) && node->value.first == *min_key) {
            // equal to min allowed only if from correct side; skip special handling
        }
        if (min_key && comp_(node->value.first, *min_key)) {
            if (verbose) diag += "BST violation: node key " + key_to_string(node->value.first) + " < min bound " + key_to_string(*min_key) + "\n";
            return { false, 0 };
        }
        if (max_key && comp_(*max_key, node->value.first)) {
            if (verbose) diag += "BST violation: node key " + key_to_string(node->value.first) + " > max bound " + key_to_string(*max_key) + "\n";
            return { false, 0 };
        }

        // Color checks: red node cannot have red child
        if (node->color == RED) {
            if (node->left != NIL_ && node->left->color == RED) {
                if (verbose) diag += "Red violation: node " + key_to_string(node->value.first) + " and left child both red\n";
                return { false, 0 };
            }
            if (node->right != NIL_ && node->right->color == RED) {
                if (verbose) diag += "Red violation: node " + key_to_string(node->value.first) + " and right child both red\n";
                return { false, 0 };
            }
        }

        // Recurse left and right, with updated bounds
        std::pair<bool,int> left = validate_node(node->left, node, min_key, &node->value.first, diag, verbose);
        if (!left.first) return { false, 0 };
        std::pair<bool,int> right = validate_node(node->right, node, &node->value.first, max_key, diag, verbose);
        if (!right.first) return { false, 0 };

        // black-height equality
        int add = (node->color == BLACK) ? 1 : 0;
        if (left.second + add != right.second + add) {
            if (verbose) {
                diag += "Black-height mismatch at key " + key_to_string(node->value.first) +
                        " left bh=" + std::to_string(left.second) +
                        " right bh=" + std::to_string(right.second) + "\n";
            }
            return { false, 0 };
        }

        return { true, left.second + add };
    }

    // helper wrapper that returns bool and optionally appends diagnostics
    bool validate_invariants_impl(std::string& diag, bool verbose) const {
        if (!NIL_) {
            if (verbose) diag += "NIL sentinel null\n";
            return false;
        }
        // NIL must be black (by construction in this implementation NIL_->color == BLACK)
        if (NIL_->color != BLACK) {
            if (verbose) diag += "NIL sentinel is not black\n";
            return false;
        }
        // root must be black
        if (root_ == nullptr) {
            if (verbose) diag += "Root is null\n";
            return false;
        }
        if (root_ != NIL_ && root_->color != BLACK) {
            if (verbose) diag += "Root is not black\n";
            return false;
        }

        // Validate recursively from root (allow no bounds)
        auto p = validate_node(root_, NIL_, nullptr, nullptr, diag, verbose);
        if (!p.first) return false;

        // count nodes and ensure size_ matches actual count
        size_t counted = 0;
        bool ok_count = true;
        std::function<void(const Node*)> count_fn = [&](const Node* n) {
            if (n == NIL_) return;
            ++counted;
            count_fn(n->left);
            count_fn(n->right);
        };
        count_fn(root_);
        if (counted != size_) {
            ok_count = false;
            if (verbose) diag += "Size mismatch: size_=" + std::to_string(size_) + " actual=" + std::to_string(counted) + "\n";
        }

        return p.first && ok_count;
    }

    // pretty-printing helper for key -> string (requires Key to be streamable)
    template <typename K = Key>
    static std::string key_to_string(const K& k) {
        std::ostringstream oss;
        oss << k;
        return oss.str();
    }

    // The remainder of the full RBTreeMapNH implementation must be present here:
    // - emplace_impl, insert(node_type&&), insert(value_type), try_emplace, insert_or_assign
    // - extract, merge, remove_node_and_deallocate, erase_fixup, transplant, etc.
    // - clear_nodes, allocate_node, deallocate_node_with, etc.
    //
    // For brevity in this snippet the full body is not duplicated; in your project,
    // keep the original full implementation and add/merge the debug helpers and rotation_count_ instrumentation.
};

#endif // RED_BLACK_TREE_MAP_NODEHANDLE_HPP
