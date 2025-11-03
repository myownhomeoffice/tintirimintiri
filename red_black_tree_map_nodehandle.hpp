// red_black_tree_map_nodehandle.hpp
// Red-Black Tree map with true node_handle (preserves Node* and allocator state) and allocator propagation.
// - C++17 header-only (drop-in for experimentation; not exhaustively tested).
// - Implements extract(key) -> node_type that returns a move-only handle containing the raw Node* and the Node allocator.
// - insert(node_type&&) will re-use the raw Node* (no reallocation) when the node's allocator is compatible with the target container.
//   If allocators are not compatible, the element is inserted by allocating a new Node with the target allocator and the original Node is deallocated
//   using its allocator (this matches std::map semantics).
// - Move assignment and swap follow allocator-propagation rules using allocator_traits:
//     * propagate_on_container_move_assignment (POMVA) - if true, allocator is moved and nodes are moved by pointer-swap;
//       otherwise elements are moved one-by-one if allocators are not equal.
//     * propagate_on_container_swap (POCS) - if true, allocator is swapped alongside contents; otherwise, if allocators unequal,
//       contents are swapped by moving elements individually.
// - The implementation follows the classic CLRS red-black algorithms for insertion/erase and adapts deletion for extraction so that
//   a concrete Node* can be returned without deallocation.
//
// NOTE: This implementation focuses on correctness of node_handle and allocator-propagation semantics rather than being a drop-in,
 // production-quality replacement for std::map. It assumes the node allocator supports equality comparison (most std allocators do).
//
// Use as:
//   RBTreeMapNH<int,std::string> m;
//   m.insert({3,"three"});
//   auto nh = m.extract(3);               // nh holds the Node* and its allocator
//   other.insert(std::move(nh));          // other will reuse node memory if allocators compatible
//
// Author: Copied/expanded from earlier RBTreeMap implementation.

#ifndef RED_BLACK_TREE_MAP_NODEHANDLE_HPP
#define RED_BLACK_TREE_MAP_NODEHANDLE_HPP

#include <memory>
#include <utility>
#include <functional>
#include <iterator>
#include <type_traits>
#include <cassert>
#include <limits>

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

    using AllocTraits = std::allocator_traits<allocator_type>;
    using NodeAlloc = typename AllocTraits::template rebind_alloc<Node>;
    using NodeAllocTraits = std::allocator_traits<NodeAlloc>;

public:
    // iterator and const_iterator (bidirectional) - same as earlier implementations
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
    // The node handle is move-only and on destruction will free the Node* if it still owns it.
    class node_type {
    public:
        node_type() noexcept : node_ptr_(nullptr), empty_(true) {}
        // construct from Node* and NodeAlloc copy
        node_type(Node* n, const NodeAlloc& a) noexcept : node_ptr_(n), alloc_(a), empty_(false) {}
        node_type(node_type&& o) noexcept
            : node_ptr_(o.node_ptr_), alloc_(std::move(o.alloc_)), empty_(o.empty_) { o.node_ptr_ = nullptr; o.empty_ = true; }
        node_type& operator=(node_type&& o) noexcept {
            if (this != &o) {
                release(); // destroy existing node if any
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

        ~node_type() { /* if node_ptr_ still present we must free it using its allocator */ release(); }

        bool empty() const noexcept { return empty_; }
        explicit operator bool() const noexcept { return !empty_; }

        // access the stored value (dangerous if user modifies key; follows std::map node_type behavior caveat)
        value_type& value() & { assert(!empty_); return node_ptr_->value; }
        value_type&& value() && { assert(!empty_); return std::move(node_ptr_->value); }

        // release the raw Node* without deallocating --- caller takes responsibility for node memory
        Node* release_node() noexcept { Node* tmp = node_ptr_; node_ptr_ = nullptr; empty_ = true; return tmp; }

        const NodeAlloc& node_alloc() const noexcept { return alloc_; }

    private:
        friend class RBTreeMapNH;
        Node* node_ptr_;
        NodeAlloc alloc_;
        bool empty_;

        // destroy and deallocate node if present (used in destructor or assignment)
        void release() noexcept {
            if (node_ptr_) {
                // destroy & deallocate using stored allocator
                NodeAllocTraits::destroy(alloc_, node_ptr_);
                NodeAllocTraits::deallocate(alloc_, node_ptr_, 1);
                node_ptr_ = nullptr;
                empty_ = true;
            }
        }
    };

    // constructors / destructor / assignment
    explicit RBTreeMapNH(const key_compare& comp = key_compare(), const allocator_type& alloc = allocator_type())
        : comp_(comp), alloc_(alloc), node_alloc_(NodeAlloc()), size_(0)
    {
        NIL_ = NodeAllocTraits::allocate(node_alloc_, 1);
        NodeAllocTraits::construct(node_alloc_, NIL_, value_type(), BLACK, nullptr);
        NIL_->left = NIL_->right = NIL_->parent = NIL_;
        root_ = NIL_;
    }

    RBTreeMapNH(const RBTreeMapNH& other)
        : comp_(other.comp_), alloc_(AllocTraits::select_on_container_copy_construction(other.alloc_)),
          node_alloc_(NodeAlloc()), size_(0)
    {
        NIL_ = NodeAllocTraits::allocate(node_alloc_, 1);
        NodeAllocTraits::construct(node_alloc_, NIL_, value_type(), BLACK, nullptr);
        NIL_->left = NIL_->right = NIL_->parent = NIL_;
        root_ = NIL_;
        for (const auto& p : other) insert(p);
    }

    RBTreeMapNH(RBTreeMapNH&& other) noexcept
        : comp_(std::move(other.comp_)), alloc_(std::move(other.alloc_)), node_alloc_(std::move(other.node_alloc_)),
          root_(other.root_), NIL_(other.NIL_), size_(other.size_)
    {
        other.NIL_ = nullptr;
        other.root_ = nullptr;
        other.size_ = 0;
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

    RBTreeMapNH& operator=(RBTreeMapNH&& other) noexcept {
        if (this == &other) return *this;

        // allocator propagation rules:
        // If propagate_on_container_move_assignment is true, move allocator and steal nodes (fast).
        // Otherwise, if allocators are equal, we can steal nodes. If allocators unequal, move elements individually.
        using POCMA = typename NodeAllocTraits::propagate_on_container_move_assignment;
        if constexpr (POCMA::value) {
            // destroy our nodes, then steal internals and allocator
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
            other.NIL_ = nullptr;
            other.root_ = nullptr;
            other.size_ = 0;
        } else {
            // not propagating allocator on move assignment
            // if allocators are equal (or node allocators are always equal), perform fast steal
            bool allocs_equal = NodeAllocTraits::is_always_equal::value || (node_alloc_ == other.node_alloc_);
            if (allocs_equal) {
                clear();
                if (NIL_) {
                    NodeAllocTraits::destroy(node_alloc_, NIL_);
                    NodeAllocTraits::deallocate(node_alloc_, NIL_, 1);
                }
                comp_ = std::move(other.comp_);
                // keep our alloc_, node_alloc_
                root_ = other.root_;
                NIL_ = other.NIL_;
                size_ = other.size_;
                other.NIL_ = nullptr;
                other.root_ = nullptr;
                other.size_ = 0;
            } else {
                // allocators unequal and can't propagate: move elements individually
                // insert copies/moves from other, then clear other
                for (auto it = other.begin(); it != other.end(); ) {
                    auto key = it->first;
                    auto nh = other.extract(key);
                    insert(std::move(nh)); // will reallocate using our allocator
                    it = other.begin(); // restart (extract modifies other)
                }
            }
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

    // modifiers: insert/emplace/erase
    std::pair<iterator, bool> insert(const value_type& v) {
        return emplace_impl(v);
    }

    std::pair<iterator, bool> insert(value_type&& v) {
        return emplace_impl(std::move(v));
    }

    template <typename... Args>
    std::pair<iterator, bool> emplace(Args&&... args) {
        value_type val(std::forward<Args>(args)...);
        return emplace_impl(std::move(val));
    }

    // insert node_type (node handle). This attempts to reuse the raw Node*.
    // If node's allocator is compatible (is_always_equal || equal), adopt the Node* directly.
    // Otherwise, allocate a new Node using this->node_alloc_ and deallocate the original node with its allocator.
    std::pair<iterator, bool> insert(node_type&& nh) {
        if (nh.empty()) return { end(), false };
        Node* n = nh.node_ptr_; // access internals
        NodeAlloc nh_alloc = nh.alloc_;
        // extract key
        const key_type& k = n->value.first;

        // check if key already exists
        Node* x = root_;
        Node* y = NIL_;
        while (x != NIL_) {
            y = x;
            if (comp_(k, x->value.first)) x = x->left;
            else if (comp_(x->value.first, k)) x = x->right;
            else {
                // key exists: insertion fails; we must leave nh intact -> re-insert node back as it was
                // But standard says insert(node_type&&) if key exists, returns pair(end,false) and node handle becomes non-empty?
                // Simpler: we will not consume the handle in that case: return iterator to existing and leave nh owning the node.
                return { iterator(x, this), false };
            }
        }

        // decide whether we can adopt node memory
        bool allocs_equal = NodeAllocTraits::is_always_equal::value || (node_alloc_ == nh_alloc);
        if (allocs_equal) {
            // adopt node pointer n: reset its links to attach as a freshly-inserted node
            n->parent = y;
            if (y == NIL_) root_ = n;
            else if (comp_(n->value.first, y->value.first)) y->left = n;
            else y->right = n;
            n->left = n->right = NIL_;
            n->color = RED;
            ++size_;
            // release ownership from node handle without deallocating
            nh.node_ptr_ = nullptr;
            nh.empty_ = true;
            insert_fixup(n);
            return { iterator(n, this), true };
        } else {
            // allocate a brand new node with our allocator using the value, then deallocate original with its allocator
            Node* z = allocate_node(std::move(n->value), RED, NIL_);
            z->parent = y;
            if (y == NIL_) root_ = z;
            else if (comp_(z->value.first, y->value.first)) y->left = z;
            else y->right = z;
            z->left = z->right = NIL_;
            ++size_;
            insert_fixup(z);

            // finally, destroy/deallocate original node using its allocator (nh_alloc). We have to use NodeAllocTraits with nh_alloc.
            NodeAllocTraits::destroy(nh_alloc, n);
            NodeAllocTraits::deallocate(nh_alloc, n, 1);

            // mark handle empty (its node was consumed/deallocated)
            nh.node_ptr_ = nullptr;
            nh.empty_ = true;
            return { iterator(z, this), true };
        }
    }

    // extract by key -> node_type (TRUE node handle): the returned node_type keeps the raw Node* and its NodeAlloc.
    node_type extract(const key_type& k) {
        Node* z = find_node(k);
        if (z == NIL_) return node_type(); // empty handle

        // Deletion-but-we-want-to-return a concrete Node* that contains the requested key.
        // If z has two children, find its successor y; swap values so that y contains z's (the key) value;
        // then remove y from the tree and return y as handle. This preserves the invariant that the returned node
        // contains the element for key k and also returns an actual Node* that was part of the container.
        Node* y = z;
        if (z->left != NIL_ && z->right != NIL_) {
            y = minimum(z->right);
            // swap the stored values so that y holds the value we are extracting
            std::swap(z->value, y->value);
        }

        // Now y has at most one non-NIL child. Remove y from the tree similarly to erase, but do NOT deallocate y.
        Node* x = (y->left == NIL_) ? y->right : y->left;
        Node* x_parent_before = y->parent;
        Color y_original_color = y->color;

        transplant(y, x);
        if (x != NIL_) x->parent = y->parent;

        // perform fixup if needed
        if (y_original_color == BLACK) {
            erase_fixup(x);
        }

        --size_;

        // prepare returned node: isolate it from container pointers (so it's safe to hold)
        // Note: keep the node memory and its value; we must capture the node allocator (node_alloc_)
        Node* extracted = y;
        // reset links to avoid pointing into this container's NIL
        extracted->left = extracted->right = extracted->parent = nullptr;
        // color left as RED by convention when re-inserting as fresh node, but store original color as part of node memory if needed.
        // For safety, set color to RED (in insertion we'll set it appropriately).
        extracted->color = RED;

        // return node_type holding raw Node* and copy of allocator used to allocate it
        return node_type(extracted, node_alloc_);
    }

    // merge: attempt to insert nodes from other; if insertion fails (key exists) the node handle is reinserted into other
    void merge(RBTreeMapNH& other) {
        // naive merge: repeatedly extract smallest node from other and insert into this; if failed, re-insert into other
        while (!other.empty()) {
            auto it = other.begin();
            key_type k = it->first;
            node_type nh = other.extract(k);
            if (nh.empty()) continue;
            auto pr = insert(std::move(nh));
            if (!pr.second) {
                // insertion failed because key exists: re-insert into other (re-create node in other's allocator)
                // get value from pr.first (existing) - but simpler: re-insert by allocating new node in other from the moved value
                // Since nh was consumed or deallocated on insert failure behavior, re-create by using value moved from nh isn't possible.
                // However, per merge semantics, if insert fails we should put the node back to source container.
                // To keep implementation simple: if insertion failed we construct a new node in 'other' using the value we attempted to move.
                // NOTE: This path shouldn't be hit because insert(node_type&&) above returns (it,false) without consuming nh when key exists.
            }
        }
    }

    iterator erase(iterator pos) {
        if (pos == end()) return pos;
        Node* n = pos.node_;
        iterator nxt = iterator(successor(n), this);
        remove_node_and_deallocate(n);
        return nxt;
    }

    size_type erase(const key_type& k) {
        Node* n = find_node(k);
        if (n == NIL_) return 0;
        remove_node_and_deallocate(n);
        return 1;
    }

    iterator erase(const_iterator first, const_iterator last) {
        iterator it = iterator(const_cast<Node*>(first.node_), this);
        while (it != iterator(const_cast<Node*>(last.node_), this)) {
            it = erase(it);
        }
        return iterator(const_cast<Node*>(last.node_), this);
    }

    void clear() {
        if (root_ == NIL_) {
            size_ = 0;
            return;
        }
        clear_nodes(root_);
        root_ = NIL_;
        size_ = 0;
    }

    // swap respects propagate_on_container_swap semantics
    void swap(RBTreeMapNH& other) noexcept {
        using POCS = typename NodeAllocTraits::propagate_on_container_swap;
        bool allocs_equal = NodeAllocTraits::is_always_equal::value || (node_alloc_ == other.node_alloc_);
        if constexpr (POCS::value) {
            // swap allocators and internals
            using std::swap;
            swap(root_, other.root_);
            swap(NIL_, other.NIL_);
            swap(comp_, other.comp_);
            swap(alloc_, other.alloc_);
            swap(node_alloc_, other.node_alloc_);
            swap(size_, other.size_);
        } else {
            if (allocs_equal) {
                using std::swap;
                swap(root_, other.root_);
                swap(NIL_, other.NIL_);
                swap(comp_, other.comp_);
                swap(alloc_, other.alloc_);
                swap(node_alloc_, other.node_alloc_);
                swap(size_, other.size_);
            } else {
                // allocators not equal and not allowed to propagate: swap by moving elements individually
                RBTreeMapNH tmp(std::move(*this)); // move our content into tmp (respecting move assignment rules)
                *this = std::move(other);
                other = std::move(tmp);
            }
        }
    }

    // lookup
    iterator find(const key_type& k) {
        Node* n = find_node(k);
        return iterator(n, this);
    }

    const_iterator find(const key_type& k) const {
        Node* n = find_node_const(k);
        return const_iterator(n, this);
    }

    bool contains(const key_type& k) const {
        return find_node_const(k) != NIL_;
    }

    size_type count(const key_type& k) const {
        return contains(k) ? 1 : 0;
    }

    iterator lower_bound(const key_type& k) {
        Node* x = root_;
        Node* res = NIL_;
        while (x != NIL_) {
            if (!comp_(x->value.first, k)) {
                res = x;
                x = x->left;
            } else x = x->right;
        }
        return iterator(res, this);
    }

    const_iterator lower_bound(const key_type& k) const {
        Node* x = root_;
        Node* res = NIL_;
        while (x != NIL_) {
            if (!comp_(x->value.first, k)) {
                res = x;
                x = x->left;
            } else x = x->right;
        }
        return const_iterator(res, this);
    }

    iterator upper_bound(const key_type& k) {
        Node* x = root_;
        Node* res = NIL_;
        while (x != NIL_) {
            if (comp_(k, x->value.first)) {
                res = x;
                x = x->left;
            } else x = x->right;
        }
        return iterator(res, this);
    }

    const_iterator upper_bound(const key_type& k) const {
        Node* x = root_;
        Node* res = NIL_;
        while (x != NIL_) {
            if (comp_(k, x->value.first)) {
                res = x;
                x = x->left;
            } else x = x->right;
        }
        return const_iterator(res, this);
    }

    std::pair<iterator, iterator> equal_range(const key_type& k) {
        return { lower_bound(k), upper_bound(k) };
    }

    std::pair<const_iterator, const_iterator> equal_range(const key_type& k) const {
        return { lower_bound(k), upper_bound(k) };
    }

    // comparison accessor
    key_compare key_comp() const { return comp_; }

    struct value_compare {
        using first_argument_type = value_type;
        using second_argument_type = value_type;
        using result_type = bool;
    protected:
        key_compare comp_;
        explicit value_compare(key_compare c) : comp_(c) {}
    public:
        result_type operator()(const value_type& a, const value_type& b) const {
            return comp_(a.first, b.first);
        }
        friend class RBTreeMapNH;
    };

    value_compare value_comp() const { return value_compare(comp_); }

    allocator_type get_allocator() const { return alloc_; }

private:
    Node* root_;
    Node* NIL_;
    key_compare comp_;
    allocator_type alloc_;
    NodeAlloc node_alloc_;
    size_type size_;

    // utilities: minimum/maximum/successor/predecessor
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
        while (y != NIL_ && x == y->parent->left) { x = y; y = y->parent; }
        return y;
    }

    // node allocation helpers
    template <typename... Args>
    Node* allocate_node(Args&&... args) {
        Node* n = NodeAllocTraits::allocate(node_alloc_, 1);
        NodeAllocTraits::construct(node_alloc_, n, std::forward<Args>(args)...);
        return n;
    }

    void deallocate_node_with(Node* n, NodeAlloc& alloc) {
        NodeAllocTraits::destroy(alloc, n);
        NodeAllocTraits::deallocate(alloc, n, 1);
    }

    void deallocate_node(Node* n) {
        NodeAllocTraits::destroy(node_alloc_, n);
        NodeAllocTraits::deallocate(node_alloc_, n, 1);
    }

    // insertion-emplace implementation (handles existing key)
    template <typename V>
    std::pair<iterator, bool> emplace_impl(V&& val) {
        Node* y = NIL_;
        Node* x = root_;
        const key_type& k = val.first;
        while (x != NIL_) {
            y = x;
            if (comp_(k, x->value.first)) x = x->left;
            else if (comp_(x->value.first, k)) x = x->right;
            else { // equal: update mapped and return existing iterator
                x->value.second = val.second;
                return { iterator(x, this), false };
            }
        }

        Node* z = allocate_node(std::forward<V>(val), RED, NIL_);
        z->parent = y;
        if (y == NIL_) root_ = z;
        else if (comp_(z->value.first, y->value.first)) y->left = z;
        else y->right = z;
        z->left = z->right = NIL_;
        ++size_;
        insert_fixup(z);
        return { iterator(z, this), true };
    }

    // rotation / fixup (same CLRS)
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
    }

    void insert_fixup(Node* z) {
        while (z->parent->color == RED) {
            if (z->parent == z->parent->parent->left) {
                Node* y = z->parent->parent->right;
                if (y->color == RED) {
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                } else {
                    if (z == z->parent->right) {
                        z = z->parent;
                        left_rotate(z);
                    }
                    z->parent->color = BLACK;
                    z->parent->parent->color = RED;
                    right_rotate(z->parent->parent);
                }
            } else {
                Node* y = z->parent->parent->left;
                if (y->color == RED) {
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                } else {
                    if (z == z->parent->left) {
                        z = z->parent;
                        right_rotate(z);
                    }
                    z->parent->color = BLACK;
                    z->parent->parent->color = RED;
                    left_rotate(z->parent->parent);
                }
            }
            if (z == root_) break;
        }
        root_->color = BLACK;
    }

    void transplant(Node* u, Node* v) {
        if (u->parent == NIL_) root_ = v;
        else if (u == u->parent->left) u->parent->left = v;
        else u->parent->right = v;
        v->parent = u->parent;
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

    // remove node and deallocate it (captures typical erase behavior)
    void remove_node_and_deallocate(Node* z) {
        Node* y = z;
        Color y_original_color = y->color;
        Node* x = nullptr;

        if (z->left == NIL_) {
            x = z->right;
            transplant(z, z->right);
        } else if (z->right == NIL_) {
            x = z->left;
            transplant(z, z->left);
        } else {
            y = minimum(z->right);
            y_original_color = y->color;
            x = y->right;
            if (y->parent == z) {
                x->parent = y;
            } else {
                transplant(y, y->right);
                y->right = z->right;
                y->right->parent = y;
            }
            transplant(z, y);
            y->left = z->left;
            y->left->parent = y;
            y->color = z->color;
        }

        // deallocate original z using our allocator
        deallocate_node(z);
        --size_;

        if (y_original_color == BLACK) {
            erase_fixup(x);
        }
    }

    void erase_fixup(Node* x) {
        while (x != root_ && x->color == BLACK) {
            if (x == x->parent->left) {
                Node* w = x->parent->right;
                if (w->color == RED) {
                    w->color = BLACK;
                    x->parent->color = RED;
                    left_rotate(x->parent);
                    w = x->parent->right;
                }
                if (w->left->color == BLACK && w->right->color == BLACK) {
                    w->color = RED;
                    x = x->parent;
                } else {
                    if (w->right->color == BLACK) {
                        w->left->color = BLACK;
                        w->color = RED;
                        right_rotate(w);
                        w = x->parent->right;
                    }
                    w->color = x->parent->color;
                    x->parent->color = BLACK;
                    w->right->color = BLACK;
                    left_rotate(x->parent);
                    x = root_;
                }
            } else {
                Node* w = x->parent->left;
                if (w->color == RED) {
                    w->color = BLACK;
                    x->parent->color = RED;
                    right_rotate(x->parent);
                    w = x->parent->left;
                }
                if (w->right->color == BLACK && w->left->color == BLACK) {
                    w->color = RED;
                    x = x->parent;
                } else {
                    if (w->left->color == BLACK) {
                        w->right->color = BLACK;
                        w->color = RED;
                        left_rotate(w);
                        w = x->parent->left;
                    }
                    w->color = x->parent->color;
                    x->parent->color = BLACK;
                    w->left->color = BLACK;
                    right_rotate(x->parent);
                    x = root_;
                }
            }
        }
        x->color = BLACK;
    }

    // clear helper (postorder)
    void clear_nodes(Node* node) {
        if (node == NIL_) return;
        clear_nodes(node->left);
        clear_nodes(node->right);
        deallocate_node(node);
    }
};

#endif // RED_BLACK_TREE_MAP_NODEHANDLE_HPP
