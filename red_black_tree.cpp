// Simple Red-Black Tree (C++17)
// - Supports: insert, erase, find, inorder traversal (with color)
// - Uses a NIL sentinel node (like CLRS)
// - Key type: int (easy to adapt to templates)

#include <iostream>
#include <vector>
#include <initializer_list>

enum Color { RED, BLACK };

struct RBNode {
    int key;
    Color color;
    RBNode *left;
    RBNode *right;
    RBNode *parent;

    RBNode(int k = 0, Color c = BLACK, RBNode* nil = nullptr)
        : key(k), color(c), left(nil), right(nil), parent(nil) {}
};

class RBTree {
public:
    RBTree() {
        NIL = new RBNode(0, BLACK, nullptr);
        NIL->left = NIL->right = NIL->parent = NIL;
        root = NIL;
    }

    ~RBTree() {
        clear(root);
        delete NIL;
    }

    void insert(int key) {
        RBNode* z = new RBNode(key, RED, NIL);
        RBNode* y = NIL;
        RBNode* x = root;
        while (x != NIL) {
            y = x;
            if (z->key < x->key) x = x->left;
            else x = x->right;
        }
        z->parent = y;
        if (y == NIL) root = z;
        else if (z->key < y->key) y->left = z;
        else y->right = z;

        z->left = z->right = NIL;
        insert_fixup(z);
    }

    bool erase(int key) {
        RBNode* z = find_node(key);
        if (z == NIL) return false;
        erase_node(z);
        return true;
    }

    bool contains(int key) const {
        return find_node(key) != NIL;
    }

    void inorder_print() const {
        inorder_print(root);
        std::cout << "\n";
    }

    std::vector<int> inorder_vector() const {
        std::vector<int> out;
        inorder_collect(root, out);
        return out;
    }

private:
    RBNode *root;
    RBNode *NIL;

    void left_rotate(RBNode* x) {
        RBNode* y = x->right;
        x->right = y->left;
        if (y->left != NIL) y->left->parent = x;
        y->parent = x->parent;
        if (x->parent == NIL) root = y;
        else if (x == x->parent->left) x->parent->left = y;
        else x->parent->right = y;
        y->left = x;
        x->parent = y;
    }

    void right_rotate(RBNode* y) {
        RBNode* x = y->left;
        y->left = x->right;
        if (x->right != NIL) x->right->parent = y;
        x->parent = y->parent;
        if (y->parent == NIL) root = x;
        else if (y == y->parent->right) y->parent->right = x;
        else y->parent->left = x;
        x->right = y;
        y->parent = x;
    }

    void insert_fixup(RBNode* z) {
        while (z->parent->color == RED) {
            if (z->parent == z->parent->parent->left) {
                RBNode* y = z->parent->parent->right; // uncle
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
            } else { // mirror
                RBNode* y = z->parent->parent->left;
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
        }
        root->color = BLACK;
    }

    void transplant(RBNode* u, RBNode* v) {
        if (u->parent == NIL) root = v;
        else if (u == u->parent->left) u->parent->left = v;
        else u->parent->right = v;
        v->parent = u->parent;
    }

    RBNode* minimum(RBNode* x) const {
        while (x->left != NIL) x = x->left;
        return x;
    }

    void erase_node(RBNode* z) {
        RBNode* y = z;
        Color y_original_color = y->color;
        RBNode* x = nullptr;

        if (z->left == NIL) {
            x = z->right;
            transplant(z, z->right);
        } else if (z->right == NIL) {
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

        delete z;

        if (y_original_color == BLACK) {
            erase_fixup(x);
        }
    }

    void erase_fixup(RBNode* x) {
        while (x != root && x->color == BLACK) {
            if (x == x->parent->left) {
                RBNode* w = x->parent->right;
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
                    x = root;
                }
            } else { // mirror
                RBNode* w = x->parent->left;
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
                    x = root;
                }
            }
        }
        x->color = BLACK;
    }

    RBNode* find_node(int key) const {
        RBNode* x = root;
        while (x != NIL) {
            if (key == x->key) return x;
            if (key < x->key) x = x->left;
            else x = x->right;
        }
        return NIL;
    }

    void inorder_print(RBNode* node) const {
        if (node == NIL) return;
        inorder_print(node->left);
        std::cout << node->key << (node->color == RED ? " (R) " : " (B) ");
        inorder_print(node->right);
    }

    void inorder_collect(RBNode* node, std::vector<int>& out) const {
        if (node == NIL) return;
        inorder_collect(node->left, out);
        out.push_back(node->key);
        inorder_collect(node->right, out);
    }

    void clear(RBNode* node) {
        if (node == NIL) return;
        clear(node->left);
        clear(node->right);
        delete node;
    }
};

// Example usage / basic test
int main() {
    RBTree t;
    std::vector<int> vals = {20, 15, 25, 10, 5, 1, 8, 30, 28, 40};
    for (int v : vals) t.insert(v);

    std::cout << "Inorder after inserts (key (color)):\n";
    t.inorder_print();

    std::cout << "Contains 8? " << (t.contains(8) ? "yes" : "no") << "\n";
    std::cout << "Contains 100? " << (t.contains(100) ? "yes" : "no") << "\n";

    std::cout << "Erasing 15, 20, 1\n";
    t.erase(15);
    t.erase(20);
    t.erase(1);

    std::cout << "Inorder after deletes:\n";
    t.inorder_print();

    return 0;
}
