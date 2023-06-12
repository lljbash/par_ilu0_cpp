#include "iluk.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <set>
#include <map>

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

namespace {

// return the number of remaining nonzeros K (K <= p)
// rem_nz_idx[0:K] keeps the sorted indices of the remaining nonzeros
int DropElements(std::vector<int>& row_nz_idx, const double* row, int p,
                 double tol, int* rem_nz_idx) {
    int row_nnz = static_cast<int>(row_nz_idx.size());
    if (row_nnz <= p) {
        p = row_nnz;
    } else {
        std::nth_element(
            row_nz_idx.begin(),
            row_nz_idx.begin() + p,
            row_nz_idx.end(),
            [row](int a, int b) {
                return std::abs(row[a]) > std::abs(row[b]);
            }
        );
    }
    int k = 0;
    for (int i = 0; i < p; ++i) {
        if (std::abs(row[row_nz_idx[i]]) >= tol) {
            rem_nz_idx[k++] = row_nz_idx[i];
        }
    }
    std::sort(rem_nz_idx, rem_nz_idx + k);
    return k;
}

}

int Ilut(const CsrMatrix* a, int p, double tau, CsrMatrix* f, int* nzmap) {
    int n = a->size;
    int knnz = 0;
    int max_knnz = n * (p + 1 + p);
    SetupCsrMatrix(f, n, max_knnz);
    std::vector<double> row(n, NAN);
    std::set<int> row_l_nz_idx_set;
    std::vector<int> row_l_nz_idx;
    std::vector<int> row_u_nz_idx;
    std::vector<int> rem_nz_idx(p);
    std::vector<int> f_diag_ptr(n, -1);
    for (int i = 0; i < n; ++i) {
        double nrm2 = 0.;
        for (int ji = a->row_ptr[i]; ji < a->row_ptr[i+1]; ++ji) {
            int j = a->col_idx[ji];
            double v_ij = a->value[ji];
            nrm2 += v_ij * v_ij;
            row[j] = v_ij;
            if (j < i) {
                row_l_nz_idx_set.insert(j);
            } else if (j > i) {
                row_u_nz_idx.push_back(j);
            }
        }
        nrm2 = std::sqrt(nrm2);
        double tol = tau * nrm2;
        for (int k : row_l_nz_idx_set) {
            row[k] /= f->value[f_diag_ptr[k]];
            if (std::abs(row[k]) < tol) {
                row[k] = 0;
                break;
            }
            for (int ji = f_diag_ptr[k] + 1; ji < f->row_ptr[k+1]; ++ji) {
                int j = f->col_idx[ji];
                double v_kj = f->value[ji];
                if (std::isnan(row[j])) {
                    row[j] = 0.;
                    if (j < i) {
                        row_l_nz_idx_set.insert(j);
                    } else if (j > i) {
                        row_u_nz_idx.push_back(j);
                    }
                }
                row[j] -= row[k] * v_kj;
            }
        }
        f->row_ptr[i] = knnz;
        row_l_nz_idx.assign(row_l_nz_idx_set.begin(), row_l_nz_idx_set.end());
        int rem = DropElements(row_l_nz_idx, row.data(), p, tol, rem_nz_idx.data());
        for (int ri = 0; ri < rem; ++ri) {
            f->col_idx[knnz] = rem_nz_idx[ri];
            f->value[knnz] = row[rem_nz_idx[ri]];
            ++knnz;
        }
        if (std::isnan(row[i])) {
            std::printf("Fatal: A(%d, %d) is 0", i, i);
            std::exit(EXIT_FAILURE);
        }
        f->col_idx[knnz] = i;
        f->value[knnz] = row[i];
        f_diag_ptr[i] = knnz;
        ++knnz;
        rem = DropElements(row_u_nz_idx, row.data(), p, tol, rem_nz_idx.data());
        for (int ri = 0; ri < rem; ++ri) {
            f->col_idx[knnz] = rem_nz_idx[ri];
            f->value[knnz] = row[rem_nz_idx[ri]];
            ++knnz;
        }
        // cleanup
        for (auto j : row_l_nz_idx_set) {
            row[j] = NAN;
        }
        row_l_nz_idx_set.clear();
        row[i] = NAN;
        for (auto j : row_u_nz_idx) {
            row[j] = NAN;
        }
        row_u_nz_idx.clear();
    }
    f->row_ptr[n] = knnz;
    if (nzmap) {
        for (int i = 0; i < n; ++i) {
            int iknnz = f->row_ptr[i];
            for (int innz = a->row_ptr[i]; innz < a->row_ptr[i+1]; ++innz) {
                while (iknnz < f->row_ptr[i+1] &&
                       f->col_idx[iknnz] < a->col_idx[innz]) {
                    ++iknnz;
                }
                nzmap[innz] = f->col_idx[iknnz] == a->col_idx[innz] ? iknnz : -1;
            }
        }
    }
    return knnz;
}
