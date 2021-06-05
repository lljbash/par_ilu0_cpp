#ifndef SUBTREE
#define SUBTREE

#include <algorithm>
#include <atomic>

constexpr const int etree_empty = -1, end_dfs = -5;

template<typename Rank>
static void sort_by_rank_ascend(int* x_begin, int* x_end, const Rank* rank) {
    std::sort(x_begin, x_end, [rank](int x, int y) { return rank[x] < rank[y]; });
}

static void build_etree_parent(int n, int vertex_begin, int vertex_end, int vertex_delta,
                               const long *edge_begins, const long *edge_ends, const int *edge_dst,
                               int *parent, int *root /* temporary */)
{
    for (int v = vertex_begin; v != vertex_end; v += vertex_delta)
    {
        parent[v] = n;
        root[v] = n;
    }
    for (int j = vertex_begin; j != vertex_end; j += vertex_delta)
    {
        for (long k = edge_begins[j]; k < edge_ends[j]; ++k)
        {
            int i = edge_dst[k];
            while (root[i] != n)
            {
                int i_temp = root[i];
                root[i] = j; // path compression
                i = i_temp;
            }
            if (i != j)
            {
                root[i] = j;
                parent[i] = j;
            }
        }
    }
}

static void build_etree_child_sibling(int n, const int *parent, int *first_child, int *next_sibling)
{
    for (int v = 0; v <= n; ++v)
    {
        first_child[v] = etree_empty;
        next_sibling[v] = etree_empty;
    }
    for (int v = n - 1; v >= 0; v--)
    {
        int p = parent[v];
        next_sibling[v] = first_child[p];
        first_child[p] = v;
    }
}

template <typename weight_t>
static void init_subtree(int vertex_begin, int vertex_end, int vertex_delta,
                         const int *parent, int *subtree_size, weight_t *subtree_weight)
{
    for (int v = vertex_begin; v != vertex_end; v += vertex_delta)
    {
        int p = parent[v];
        subtree_weight[p] += subtree_weight[v];
        subtree_size[p] += subtree_size[v];
    }
}

template <typename LoadBalancer>
static int divide_subtree(int n, int nproc, const int *first_child, const int *next_sibling, \
                          const typename LoadBalancer::weight_t *subtree_weight, \
                          int *subtrees, LoadBalancer &&load_balancer)
{
    using weight_t = typename LoadBalancer::weight_t;
    int nsubtree;
    int ngen = 1;
    int ichain = n;
    int max_weight_pos = 0;
    weight_t max_weight = subtree_weight[n];
    weight_t sum_weight = subtree_weight[n];
    subtrees[0] = n;
    while (ngen > 0 && !load_balancer.is_balanced(nproc, nsubtree, max_weight, sum_weight))
    {
        int t = subtrees[max_weight_pos], f = first_child[t];
        while (f != etree_empty && next_sibling[f] == etree_empty)
        {
            t = f;
            f = first_child[t];
        }
        if (f == etree_empty) // 暂时排除无分叉的长链
        {
            subtrees[--ichain] = subtrees[max_weight_pos];
            subtrees[max_weight_pos] = subtrees[--ngen];
        }
        else
        {
            int ch;
            sum_weight -= subtree_weight[subtrees[max_weight_pos]];
            ch = first_child[t];
            subtrees[max_weight_pos] = ch;
            sum_weight += subtree_weight[ch];
            while (next_sibling[ch] >= 0)
            {
                ch = next_sibling[ch];
                subtrees[ngen++] = ch;
                sum_weight += subtree_weight[ch];
            }
        }
        // 线程数较多的时候，子树数量也会较多，可以改成heap
        max_weight = 0, max_weight_pos = 0;
        for (int i = 0; i < ngen; ++i)
        {
            if (max_weight < subtree_weight[subtrees[i]])
            {
                max_weight = subtree_weight[subtrees[i]];
                max_weight_pos = i;
            }
        }
    }
    if (ngen > 0)
    {
        std::swap(subtrees[ngen - 1], subtrees[max_weight_pos]);
        sort_by_rank_ascend(subtrees, subtrees + ngen - 1, subtree_weight);
    }
    nsubtree = std::copy(subtrees + ichain, subtrees + n, subtrees + ngen) - subtrees; // chain 部分已经有序

    if (nproc >= 2)
    {
        while (!load_balancer.fallback(nproc, nsubtree) && \
                !load_balancer.is_balanced(nproc, nsubtree, subtree_weight[subtrees[nsubtree - 1]], sum_weight)) // 将最长的长链缩短
        {
            int last = subtrees[nsubtree - 1], second_last = subtrees[nsubtree - 2];
            sum_weight -= subtree_weight[last];
            while (last != etree_empty && subtree_weight[last] > subtree_weight[second_last])
            {
                last = first_child[last];
            }
            if (last == etree_empty)
            {
                --nsubtree;
            }
            else
            {
                int new_pos;
                weight_t new_weight = subtree_weight[last];
                sum_weight += new_weight;
                for (new_pos = nsubtree - 2; new_pos >= 0 && subtree_weight[subtrees[new_pos]] > new_weight; --new_pos)
                {
                    subtrees[new_pos + 1] = subtrees[new_pos];
                }
                subtrees[new_pos] = last;
            }
        }
    }
    if (load_balancer.fallback(nproc, nsubtree))
    { // fall back to queue
        return 0;
    }
    return nsubtree;
}

template <typename weight_t>
static void schedule_subtree(int n, int nproc, int nsubtree, const int *subtrees, const int *subtree_size,
                             const weight_t *subtree_weight, int *assign, int *part_ptr)
{
    weight_t *weight_per_proc = new weight_t[nproc];
    for (int i = 0; i < n; ++i)
    {
        assign[i] = etree_empty;
    }
    for (int i = 0; i <= nproc; ++i)
    {
        weight_per_proc[i] = 0;
        part_ptr[i] = 0;
    }
    for (int i = nsubtree - 1; i >= 0; --i)
    {
        int t = subtrees[i], m = n, mi;
        for (int i = 0; i < nproc; ++i) // 如果线程数较多可以改用heap
        {
            if (m > weight_per_proc[i])
            {
                m = weight_per_proc[i];
                mi = i;
            }
        }
        assign[t] = mi;
        weight_per_proc[mi] += subtree_weight[t];
        part_ptr[mi] += subtree_size[t];
    }

    for (int i = 0, pre = 0; i <= nproc; ++i)
    {
        int tmp = part_ptr[i];
        part_ptr[i] = pre;
        pre += tmp;
    }
    delete[] weight_per_proc;
}

static void build_partitions(int n, int nproc, int nsubtree, const int *subtrees, const int *subtree_size,
                             const int *parent, const int *first_child, int *next_sibling,
                             const int *part_ptr, int *partitions, int *assign)
{

    std::atomic_int *part_ptr_atomic = new std::atomic_int[nproc];

    for (int i = 0; i < nproc; ++i)
    {
        part_ptr_atomic[i].store(part_ptr[i], std::memory_order_relaxed);
    }
#pragma omp parallel for num_threads(nproc) schedule(dynamic)
    for (int i = 0; i < nsubtree; ++i)
    {
        int t = subtrees[nsubtree - 1 - i], proc = assign[t];
        int nv = subtree_size[t];
        int k = part_ptr_atomic[proc].fetch_add(nv, std::memory_order_relaxed), end = k + nv;
        next_sibling[t] = end_dfs;
        for (int current = t; k < end;) // dfs
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
                current = parent[current];

                /* assign the parent node */
                partitions[k++] = current;
                assign[current] = proc;
            }

            /* go to the next */
            current = next;
        }
    }

    for (int i = 0, k = part_ptr[nproc]; i < n; ++i)
    {
        if (assign[i] == etree_empty)
        {
            partitions[k++] = i;
        }
    }

    delete[] part_ptr_atomic;
}

template<typename weight_t>
static void calculate_levels(int n, int vertex_begin, int vertex_end, int vertex_delta,
                             const long *edge_begins, const long *edge_ends, const int *edge_dst,
                             const int *assign, const weight_t *node_weight, weight_t *level)
{
    int vertex_rbegin = vertex_end - vertex_delta, vertex_rend = vertex_begin - vertex_delta;
    for (int i = 0; i < n; ++i)
    {
        level[i] = - node_weight[i];
    }
    for (int j = vertex_rbegin; j != vertex_rend; j -= vertex_delta)
    {
        if (assign[j] == etree_empty)
        {
            for (long k = edge_begins[j]; k < edge_ends[j]; ++k)
            {
                int i = edge_dst[k];
                weight_t l = level[j] - node_weight[i];
                if (level[i] > l)
                {
                    level[i] = l;
                }
            }
        }
    }
}

template<typename LoadBalancer>
struct result_t {
    using weight_t = typename LoadBalancer::weight_t;
    weight_t cost;   // 对调度结果的评分，cost越低效果越好
};

/*
    n     - matrix dimension
    nproc - number of processors
    partitions[part_ptr[i]     : part_ptr[i+1]]     - columns assigned to processor i, in topological order (depth first)
    partitions[part_ptr[nproc] : n]                 - columns remain in the global task queue, in topological order (breadth first)

    DAG definition:
        v depends on edge_dst[edge_begins[v] : edge_ends[v]]
        edge_dst[edge_begins[v] : edge_ends[v]] is an unordered subset of vertex_begin : vertex_delta : v
    
    typename LoadBalancer::weight_t                                     is the data type of weights
    load_balancer.get_weight(v)                                         returns the weight of the vertex v
    load_balancer.is_balanced(nproc, nsubtree, max_weight, sum_weight)  tells whether the current subtrees are balanced
    load_balancer.fallback(nproc, nsubtree)                             tells whether we should fall back to queues
 */
template <typename LoadBalancer>
static result_t<LoadBalancer> partition_subtree(int nproc, int vertex_begin, int vertex_end, int vertex_delta,
                              const long *edge_begins, const long *edge_ends, const int *edge_dst,
                              int *part_ptr /* output: length nproc + 1 */,
                              int *partitions /* output: length n */,
                              LoadBalancer &&load_balancer)
{
    using weight_t = typename LoadBalancer::weight_t;
    int n = (vertex_begin > vertex_end) ? (vertex_begin - vertex_end) : (vertex_end - vertex_begin);
    int *parent = new int[n];
    int *first_child = new int[n + 1];
    int *next_sibling = new int[n + 1];
    weight_t *node_weight = new weight_t[n];
    weight_t *subtree_weight = new weight_t[n + 1];
    int *subtree_size = new int[n + 1];
    int *subtrees = new int[n];
    int *assign = new int[n];

    // construct tree
    build_etree_parent(n, vertex_begin, vertex_end, vertex_delta, edge_begins, edge_ends, edge_dst, parent, first_child);
    build_etree_child_sibling(n, parent, first_child, next_sibling);

    for (int v = vertex_begin; v != vertex_end; v += vertex_delta)
    {
        subtree_weight[v] = node_weight[v] = load_balancer.get_weight(v);
        subtree_size[v] = 1;
    }
    subtree_weight[n] = 0;
    subtree_size[n] = 0;

    init_subtree(vertex_begin, vertex_end, vertex_delta, parent, subtree_size, subtree_weight);

    // divide subtrees
    int nsubtree = divide_subtree(n, nproc, first_child, next_sibling, subtree_weight, subtrees, std::forward<LoadBalancer&&>(load_balancer));

    schedule_subtree(n, nproc, nsubtree, subtrees, subtree_size, subtree_weight, assign, part_ptr);

    build_partitions(n, nproc, nsubtree, subtrees, subtree_size, parent, first_child, next_sibling, part_ptr, partitions, assign);

    delete[] first_child;
    delete[] next_sibling;
    delete[] subtree_size;

    // level-set sort
    weight_t *level = new weight_t[n];
    calculate_levels(n, vertex_begin, vertex_end, vertex_delta, edge_begins, edge_ends, edge_dst, assign, node_weight, level);
    sort_by_rank_ascend(partitions + part_ptr[nproc], partitions + n, level);

    weight_t last_level = level[partitions[n - 1]], longest = last_level - level[partitions[part_ptr[nproc]];
    for(int i = 0; i < nsubtree; ++i) {
        weight_t diff = last_level - level[parent[subtrees[i]]];
        weight_t path = diff + subtree_weight[subtrees[i]];
        longest = (longest > path) ? longest : path;
    }

    delete[] parent;
    delete[] subtrees;
    delete[] assign;
    delete[] node_weight;
    delete[] subtree_weight;
    delete[] level;

    return {longest};
}

#endif