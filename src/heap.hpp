#pragma once

#include <vector>
#include <algorithm>
#include <numeric>

namespace lljbash {

template <class Key, class Compare>
class Heap {
public:
    Heap(const Compare& comp = Compare()) : key_compare_(comp) {
        Clear();
    }

    int size() const { return size_; }

    void Clear() {
        size_ = 0;
        data_.resize(1);
        index_.resize(1);
        location_.clear();
    }

    void Heapify(const Key* data, int n) {
        size_ = n;
        data_.resize(n + 1);
        location_.resize(n);
        std::copy_n(data, n, data_.data() + 1);
        std::iota(index_.begin() + 1, location_.end(), 0);
        std::iota(location_.begin(), location_.end(), 1);
        for (int i = n / 2; i; --i) {
            Down(i);
        }
    }

    void Push(Key element) {
        ++size_;
        data_.push_back(element);
        index_.push_back(size_ - 1);
        location_.push_back(size_);
        Up(size_);
    }

    void Pop() {
        int i = location_[size_-1];
        index_[i] = index_[1];
        location_[index_[i]] = i;
        index_[1] = index_[size_];
        location_[index_[1]] = 1;
        data_[1] = data_[size_];
        --size_;
        data_.pop_back();
        index_.pop_back();
        location_.pop_back();
        Down(1);
    }

    void ModifyKey(int key, Key element) {
        int i = location_[key];
        bool decrease_key = key_compare_(element, data_[i]) < 0;
        data_[i] = std::move(element);
        if (decrease_key) {
            Up(i);
        }
        else {
            Down(i);
        }
    }

    std::pair<int, const Key&> Top() const {
        return {index_[1], data_[1]};
    }

private:
    bool strict_less(int i, int j) {
        auto ret = key_compare_(data_[i], data_[j]);
        return ret < 0 || (ret == 0 && index_[i] < index_[j]);
    }

    void SwapElements(int i, int j) {
        std::swap(data_[i], data_[j]);
        std::swap(index_[i], index_[j]);
        location_[index_[i]] = i;
        location_[index_[j]] = j;
    }

    int Up(int i) {
        while (i > 1 && strict_less(i, i / 2)) {
            SwapElements(i, i / 2);
            i /= 2;
        }
        return i;
    }

    int Down(int i) {
        while (i * 2 <= size_) {
            int j = i * 2;
            if (j + 1 <= size_ && strict_less(j + 1, j)) {
                ++j;
            }
            if (!strict_less(j, i)) {
                break;
            }
            SwapElements(i, j);
            i = j;
        }
        return i;
    }

    Compare key_compare_;
    int size_;
    std::vector<Key> data_;
    std::vector<int> index_;
    std::vector<int> location_;
};

} // namespace lljbash
