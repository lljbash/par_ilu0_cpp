#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "csr_matrix.h"

void PrintCsrPattern(CsrMatrix* csr, int* p) {
    int n = csr->size;
    char* buf = (char*) malloc(n);
    for (int i = 0; i < n; ++i) {
        memset(buf, '.', n);
        for (int j = csr->row_ptr[p[i]]; j < csr->row_ptr[p[i]+1]; ++j) {
            buf[csr->col_idx[j]] = '*';
        }
        for (int j = 0; j < n; ++j) {
            printf(" %c", buf[p[j]]);
        }
        printf("\n");
    }
    free(buf);
}

int main() {
    int n = 5;
    int nnz = 14;
    int Ap[] = {0, 2, 6, 9, 11, 14};
    int Ai[] = {0, 1, 0, 1, 2, 4, 1, 2, 3, 2, 3, 1, 2, 4};
    int p[] = {0, 1, 2, 3, 4};
    int ip[5];
    CsrMatrix csr = CSR_MATRIX_DEFAULT;
    SetupCsrMatrix(&csr, n, nnz);
    memcpy(csr.row_ptr, Ap, sizeof(Ap));
    memcpy(csr.col_idx, Ai, sizeof(Ai));
    puts("NO:");
    PrintCsrPattern(&csr, p);
    int ret = CsrAmdOrder(&csr, p, ip);
    puts("\nAMD:");
    PrintCsrPattern(&csr, p);
    puts("\nP:");
    for (int i = 0; i < n; ++i) {
        printf(" %d", p[i]);
    }
    puts("\n\ninvP:");
    for (int i = 0; i < n; ++i) {
        printf(" %d", ip[i]);
    }
    printf("\n\nret: %d\n", ret);
    DestroyCsrMatrix(&csr);
    return 0;
}
