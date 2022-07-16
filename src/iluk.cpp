#include "iluk.hpp"
#include <vector>
#include <map>

namespace lljbash {

int IlukSymbolic(const CsrMatrix* a, int lof, CsrMatrix* f, int* nzmap) {
    int n = a->size;
    int nnz = GetCsrNonzeros(a);
    int knnz = 0;
    std::vector<std::map<int, int>> row_elements(n);
    for (int i = 0; i < n; ++i) {
        for (int ji = a->row_ptr[i]; ji < a->row_ptr[i+1]; ++ji) {
            row_elements[i][a->col_idx[ji]] = 0;
        }
        for (auto kit = row_elements[i].begin(); kit != row_elements[i].end(); ++kit) {
            int k = kit->first;
            if (k >= i) {
                break;
            }
            if (kit->second >= lof) {
                continue;
            }
            for (auto jit = row_elements[k].find(k); jit != row_elements[k].end(); ++jit) {
                int level = kit->second + jit->second + 1;
                if (level <= lof) {
                    row_elements[i].try_emplace(jit->first, level);
                }
            }
        }
        knnz += static_cast<int>(row_elements[i].size());
    }
    SetupCsrMatrix(f, n, knnz);
    int innz = 0;
    int iknnz = 0;
    for (int i = 0; i < n; ++i) {
        f->row_ptr[i] = iknnz;
        for (auto [j, level] : row_elements[i]) {
            f->col_idx[iknnz] = j;
            if (level == 0) {
                f->value[iknnz] = a->value[innz];
                if (nzmap) {
                    nzmap[innz] = iknnz;
                }
                ++innz;
            }
            ++iknnz;
        }
    }
    f->row_ptr[n] = knnz;
    if (innz != nnz || iknnz != knnz) {
        std::puts("error");
        std::exit(-1);
    }
    return knnz;
}

} // namespace lljbash
