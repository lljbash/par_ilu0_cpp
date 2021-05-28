#ifndef SUBTREE
#define SUBTREE

#include <string.h>
#include <algorithm>

/*
    n     - matrix dimension
    nproc - number of processors
    partitions[part_ptr[i]     : part_ptr[i+1]]     - columns assigned to processor i, in topological order (depth first)
    partitions[part_ptr[nproc] : part_ptr[nproc+1]] - columns remain in the global task queue, in topological order (breadth first)
 */

static void partition_subtree(int n, int nproc, const long *col_ptr, const long *diagnal_ptr, const int *row_index,
                              int *part_ptr /* output: length nproc + 2 */,
                              int *partitions /* output: length n */)
{
    constexpr int etree_empty = -1, end_dfs = -5;
    int *etree = new int[n];
    int *root = partitions;
    int *first_child = new int[n + 1];
    int *next_sibling = new int[n + 1];
    int *desc = new int[n + 1];
    int *nchild = new int[n + 1];
    int *subtrees = new int[n];
    std::atomic_int *weights = new std::atomic_int[nproc];
    int nsubtree, ngen, ichain, max_weight, max_weight_pos, sum_weight;

    // construct tree
    for (int j = 0; j < n; ++j)
    {
        etree[j] = n;
        root[j] = n;
    }
    for (int j = 0; j < n; ++j)
    {
        for (long k = col_ptr[j]; k < diagnal_ptr[j]; ++k)
        {
            int i = row_index[k];
            while (root[i] != n)
            {
                int i_temp = root[i];
                root[i] = j; // path compression
                i = i_temp;
            }
            if (i != j)
            {
                root[i] = j;
                etree[i] = j;
            }
        }
    }

    for (int j = 0; j < n + 1; ++j)
    {
        first_child[j] = -1;
        next_sibling[j] = -1;
        nchild[j] = 0;
        desc[j] = 0;
    }
    for (int v = 0; v < n; ++v)
    {
        int p = etree[v];
        nchild[p] += 1;
        desc[p] += desc[v] + 1;
    }
    for (int v = 0; v <= n; first_child[v++] = etree_empty)
        ;
    for (int v = n - 1; v >= 0; v--)
    {
        int p = etree[v];
        next_sibling[v] = first_child[p];
        first_child[p] = v;
        desc[v] += 1;
    }

    // divide subtrees
    ngen = 1;
    ichain = n;
    max_weight = n;
    max_weight_pos = 0;
    sum_weight = n;
    subtrees[0] = n;
    while (ngen > 0 && max_weight * nproc > sum_weight)
    {
        int t = subtrees[max_weight_pos];
        sum_weight -= desc[t];
        while (nchild[t] == 1)
        {
            t = first_child[t];
        }
        if (nchild[t] == 0)
        { // move to chain part
            subtrees[--ichain] = subtrees[max_weight_pos];
            subtrees[max_weight_pos] = subtrees[--ngen];
        }
        else
        {
            int k = first_child[t];
            subtrees[max_weight_pos] = k;
            while (next_sibling[k] >= 0)
            {
                k = next_sibling[k];
                subtrees[ngen++] = k;
                sum_weight += desc[k];
            }
        }
        max_weight = 0, max_weight_pos = 0;
        for (int i = 0; i < ngen; ++i)
        {
            if (max_weight < desc[subtrees[i]])
            {
                max_weight = desc[subtrees[i]];
                max_weight_pos = i;
            }
        }
    }
    memcpy(subtrees + ngen, subtrees + ichain, sizeof(int) * (n - ichain));
    nsubtree = ngen + (n - ichain);
    std::sort(subtrees, subtrees + nsubtree, [desc](int x, int y) { return desc[x] > desc[y]; });

    // schedule subtrees
    int *assign = nchild; // reuse
    for (int i = 0; i < n; ++i)
    {
        assign[i] = nproc;
    }
    for (int i = 0; i <= nproc; ++i)
    {
        part_ptr[i] = 0;
    }
    for (int i = 0; i < nsubtree; ++i)
    {
        int t = subtrees[i], m = n, mi;
        for (int i = 0; i < nproc; ++i)
        {
            if (m > part_ptr[i])
            {
                m = part_ptr[i];
                mi = i;
            }
        }
        assign[t] = mi;
        part_ptr[mi] += desc[t];
    }
    for (int i = 0, pre = 0; i <= nproc; ++i)
    {
        int tmp = part_ptr[i];
        part_ptr[i] = pre;
        pre += tmp;
        weights[i].store(part_ptr[i], std::memory_order_relaxed);
    }
#pragma omp parallel for
    for (int i = 0; i < nsubtree; ++i)
    {
        int t = subtrees[i], proc = assign[t];
        int nv = desc[t];
        int k = weights[proc].fetch_add(nv, std::memory_order_relaxed), end = k + nv;
        next_sibling[t] = end_dfs;
        for (int current = t; k < end;)
        {
            int next;

            /* find the first leaf */
            for (int first = first_child[current]; first != etree_empty; first = first_child[first])
            {
                current = first;
            }

            /* assign this leaf */
            partitions[k++] = current;
            assign[current] = proc;

            /* look for the next */
            /* next_sibling[n] == END_DFS ends this loop */
            for (next = next_sibling[current]; next == etree_empty; next = next_sibling[current])
            {

                /* no more kids : back to the parent node */
                current = etree[current];

                /* assign the parent node */
                partitions[k++] = current;
                assign[current] = proc;
            }

            /* go to the next */
            current = next;
        }
    }
    for (int i = 1, pre = 0; i <= nproc; ++i)
    {
        part_ptr[i] = weights[i - 1].load(std::memory_order_relaxed);
    }
    for (int i = 0, k = part_ptr[nproc]; i < n; ++i)
    {
        if (assign[i] == nproc)
        {
            partitions[k++] = i;
        }
    }
    part_ptr[nproc + 1] = n;

    // level-set sort
    int *level = etree; // reuse
    for (int i = 0; i < n; ++i)
    {
        level[i] = n;
    }
    for (int j = n - 1; j >= 0; --j)
    {
        if (assign[j] == nproc)
        {
            for (long k = col_ptr[j]; k < diagnal_ptr[j]; ++k)
            {
                int i = row_index[k], l = level[j] - 1;
                if (level[i] > l)
                {
                    level[i] = l;
                }
            }
        }
    }
    std::sort(partitions + part_ptr[nproc], partitions + n, [level](int x, int y) { return level[x] < level[y]; });

    delete[] etree;
    delete[] first_child;
    delete[] next_sibling;
    delete[] nchild;
    delete[] desc;
    delete[] subtrees;
    delete[] weights;
}

#endif