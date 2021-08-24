#ifndef SUBTREE_H
#define SUBTREE_H

#include <limits>
#include <memory>
#include <algorithm>
#include <atomic>

template <class T>
class ConstBiasArray
{
public:
    ConstBiasArray(T *data) : data_(data), bias_(0) {}
    ConstBiasArray(T *data, T bias) : data_(data), bias_(bias) {}
    T operator[](size_t index) const { return data_[index] + bias_; }

private:
    T *data_;
    T bias_;
};

constexpr const int etree_empty = -1;
using unique_int_ptr = std::unique_ptr<int[]>;

template <typename Rank>
static void sort_by_rank_ascend(int *x_begin, int *x_end, const Rank *rank)
{
    std::sort(x_begin, x_end, [rank](int x, int y) { return rank[x] < rank[y]; });
}

static void build_etree_parent(int n, int vertex_begin, int vertex_end, int vertex_delta,
                               const ConstBiasArray<long> &edge_begins, const long *edge_ends, const int *edge_dst,
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

template <typename weight_t>
static int divide_subtree(int n, int nproc, const int *first_child, const int *next_sibling,
                          const weight_t *subtree_weight, int *subtrees)
{
    int nsubtree;
    int ngen = 1;
    int ichain = n;
    int max_weight_pos = 0;
    weight_t max_weight = subtree_weight[n];
    weight_t sum_weight = subtree_weight[n];
    subtrees[0] = n;

    while (ngen > 0 && max_weight * nproc > sum_weight)
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
        while (nsubtree >= nproc && subtree_weight[subtrees[nsubtree - 1]] * nproc > sum_weight) // 将最长的长链缩短
        {
            int last = subtrees[nsubtree - 1], ch = first_child[last];
            weight_t second_weight = subtree_weight[subtrees[nsubtree - 2]], th = sum_weight / nproc;
            th = (second_weight < th) ? second_weight : th;
            sum_weight -= subtree_weight[last];
            while (ch != etree_empty && subtree_weight[ch] > th)
            {
                ch = first_child[ch];
            }
            if (ch == etree_empty)
            {
                --nsubtree;
            }
            else
            {
                int new_pos;
                weight_t new_weight = subtree_weight[ch];
                sum_weight += new_weight;
                for (new_pos = nsubtree - 2; new_pos >= 0 && subtree_weight[subtrees[new_pos]] > new_weight; --new_pos)
                {
                    subtrees[new_pos + 1] = subtrees[new_pos];
                }
                subtrees[new_pos + 1] = ch;
            }
        }
    }
    if (nsubtree < nproc)
    { // fall back to queue
        return 0;
    }
    return nsubtree;
}

template <typename weight_t>
static void schedule_subtree(int n, int nproc, int nsubtree, const int *subtrees, const int *subtree_size,
                             const weight_t *subtree_weight, int *assign, int *part_ptr)
{
    std::unique_ptr<weight_t[]> weight_per_proc(new weight_t[nproc]);
    for (int i = 0; i < n; ++i)
    {
        assign[i] = etree_empty;
    }
    for (int i = 0; i < nproc; ++i)
    {
        weight_per_proc[i] = 0;
        part_ptr[i] = 0;
    }
    for (int i = nsubtree - 1; i >= 0; --i)
    {
        int mi, t = subtrees[i], m = std::numeric_limits<weight_t>::max();
        for (int ii = 0; ii < nproc; ++ii) // 如果线程数较多可以改用heap
        {
            if (m > weight_per_proc[ii])
            {
                m = weight_per_proc[ii];
                mi = ii;
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
}

static void build_partitions(int n, int nproc, int nsubtree, const int *subtrees, const int *subtree_size,
                             const int *parent, const int *first_child, const int *next_sibling,
                             const int *part_ptr, int *partitions, int *assign)
{

    std::unique_ptr<std::atomic_int[]> part_ptr_atomic(new std::atomic_int[nproc]);

    for (int i = 0; i < nproc; ++i)
    {
        part_ptr_atomic[i].store(part_ptr[i], std::memory_order_relaxed);
    }
#pragma omp parallel num_threads(nproc)
    {
#pragma omp single
        for (int k = part_ptr[nproc], current = n; k < n;) // dfs
        {
            int first, next;

            auto next_unassigned = [](int current_, const int *next_, const int *assign_) {
                while ((current_ != etree_empty) && (assign_[current_] != etree_empty))
                {
                    current_ = next_[current_];
                }
                return current_;
            };

            /* find the first leaf */
            for (first = next_unassigned(first_child[current], next_sibling, assign); first != etree_empty;
                 first = next_unassigned(first_child[current], next_sibling, assign))
            {
                current = first;
            }

            /* assign this leaf */
            partitions[k++] = current;

            /* look for the next */
            for (next = next_unassigned(next_sibling[current], next_sibling, assign); (next == etree_empty) && (k < n);
                 next = next_unassigned(next_sibling[current], next_sibling, assign))
            {
                /* no more kids : back to the parent node */
                current = parent[current];

                /* assign the parent node */
                partitions[k++] = current;
            }

            /* go to the next */
            current = next;
        }

#pragma omp for nowait schedule(dynamic)
        for (int i = 0; i < nsubtree; ++i)
        {
            int t = subtrees[nsubtree - 1 - i], proc = assign[t];
            int nv = subtree_size[t];
            int k = part_ptr_atomic[proc].fetch_add(nv, std::memory_order_relaxed), end = k + nv;
            for (int current = t; k < end;) // dfs
            {
                int first, next;

                /* find the first leaf */
                for (first = first_child[current]; first != etree_empty; first = first_child[first])
                {
                    current = first;
                }

                /* assign this leaf */
                partitions[k++] = current;

                /* look for the next */
                for (next = next_sibling[current]; (next == etree_empty) && (k < end); next = next_sibling[current])
                {
                    /* no more kids : back to the parent node */
                    current = parent[current];

                    /* assign the parent node */
                    partitions[k++] = current;
                }

                /* go to the next */
                current = next;
            }
        }
    }
}

static int aggregate_nodes(int n, int *etree /* destroyed */, const int *post_order, int part_begin,
                           int *sup2col, int *col2sup, int granularity)
{
    const int part_size = n - part_begin;
    int nsuper;
    int *post_order_inverse = sup2col;
    int *etree_temp = col2sup;

    /*
        if         etree[post_order[j]] == post_order[k]
        then       etree_reorder[j] == k
        that means etree[post_order[j]] == post_order[etree_reorder[j]]
        then       etree_temp[j] = etree[post_order[j]]
        and        post_order[etree_reorder[j]] == etree_temp[j]
        then       etree_reorder[j] = post_order_inverse[etree_temp[j]]
    */
    for (int j = part_begin; j < n; ++j)
    {
        int p = post_order[j];
        etree_temp[j] = etree[p];
        post_order_inverse[p] = j;
    }
    post_order_inverse[n] = n;
    for (int j = part_begin; j < n; ++j)
    {
        etree[j] = post_order_inverse[etree_temp[j]];
    }

    for (int j = 0; j < n; ++j)
    {
        col2sup[j] = etree_empty;
    }
    /* aggregation, already in post-order */
    nsuper = 0;
    sup2col[nsuper] = part_begin;
    for (int j = part_begin; j < n;)
    {
        int parent = etree[j];
        int first = j;
        while (parent != n && (parent - first) < granularity)
        {
            j = parent;
            parent = etree[j];
        }

        ++j;

        for (int k = first; k < j; ++k)
        {
            col2sup[post_order[k]] = nsuper;
        }
        sup2col[++nsuper] = j;
    }
    return nsuper;
}

template <typename LoadBalancer>
static void calculate_levels(int nsuper, const int *col2sup, const int *sup2col, const int *post_order,
                             const ConstBiasArray<long> &edge_begins, const long *edge_ends, const int *edge_dst,
                             typename LoadBalancer::weight_t *level, const LoadBalancer &load_balancer)
{
    using weight_t = typename LoadBalancer::weight_t;
    std::unique_ptr<weight_t[]> super_weight(new weight_t[nsuper]);

    for (int i = 0; i < nsuper; ++i)
    {
        weight_t l = 0;
        for (int j = sup2col[i], j_end = sup2col[i + 1]; j < j_end; ++j)
        {
            l += load_balancer.weight_in_queue(j);
        }
        super_weight[i] = l;
        level[i] = nsuper - l;
    }

    for (int i = nsuper - 1; i >= 0; --i)
    {
        weight_t l = level[i];
        for (int v = sup2col[i], v_end = sup2col[i + 1]; v < v_end; ++v)
        {
            int j = post_order[v];
            for (long k = edge_begins[j]; k < edge_ends[j]; ++k)
            {
                int ii = col2sup[edge_dst[k]];
                if ((ii != etree_empty) && (ii != i))
                { // belongs to the queue part && not the same super
                    weight_t ll = l - super_weight[ii];
                    if (level[ii] > ll)
                    {
                        level[ii] = ll;
                    }
                }
            }
        }
    }
}

static void build_queue(int n, int nsuper, const int *sup2col, const int *sup_perm,
                        int *task_queue, int *partitions, int queue_part_begin, int *temp)
{
    int *t = temp + queue_part_begin;
    for (int i = 0; i < nsuper; ++i)
    {
        int isup = sup_perm[i];
        task_queue[i] = t - temp;
        t = std::copy(partitions + sup2col[isup], partitions + sup2col[isup + 1], t);
    }
    task_queue[nsuper] = n;
    std::copy(temp + queue_part_begin, temp + n, partitions + queue_part_begin);
}

template <typename LoadBalancer>
static void partition_subtree(int n, int nproc, int vertex_begin, int vertex_end, int vertex_delta,
                              ConstBiasArray<long> edge_begins, const long *edge_ends, const int *edge_dst,
                              int *part_ptr, int *partitions, int *parent,
                              const LoadBalancer &load_balancer)
{
    using weight_t = typename LoadBalancer::weight_t;
    using unique_wt_ptr = std::unique_ptr<weight_t[]>;
    unique_int_ptr first_child(new int[n + 1]);
    unique_int_ptr next_sibling(new int[n + 1]);
    unique_wt_ptr subtree_weight(new weight_t[n + 1]);
    unique_int_ptr subtree_size(new int[n + 1]);
    unique_int_ptr subtrees(new int[n]);
    unique_int_ptr assign(new int[n]);

    // construct tree
    build_etree_parent(n, vertex_begin, vertex_end, vertex_delta, edge_begins, edge_ends, edge_dst, parent, first_child.get());
    build_etree_child_sibling(n, parent, first_child.get(), next_sibling.get());

    for (int v = vertex_begin; v != vertex_end; v += vertex_delta)
    {
        subtree_weight[v] = load_balancer.weight_in_subtree(v);
        subtree_size[v] = 1;
    }
    subtree_weight[n] = 0;
    subtree_size[n] = 0;

    init_subtree(vertex_begin, vertex_end, vertex_delta, parent, subtree_size.get(), subtree_weight.get());

    // divide subtrees
    int nsubtree = divide_subtree(n, nproc, first_child.get(), next_sibling.get(), subtree_weight.get(), subtrees.get());

    schedule_subtree(n, nproc, nsubtree, subtrees.get(), subtree_size.get(), subtree_weight.get(), assign.get(), part_ptr);

    build_partitions(n, nproc, nsubtree, subtrees.get(), subtree_size.get(),
                     parent, first_child.get(), next_sibling.get(), part_ptr, partitions, assign.get());
}

class NaiveLoadBalancer
{
public:
    using weight_t = int;
    weight_t weight_in_subtree(int vertex) const
    {
        return 1;
    }
    weight_t weight_in_queue(int vertex) const
    {
        return 1;
    }
    constexpr int queue_granularity() const
    {
        return 4;
    }
};

/*
    n     - matrix dimension
    nproc - number of processors
    partitions[part_ptr[i]     : part_ptr[i+1]]     - columns assigned to processor i, in topological order
    partitions[part_ptr[nproc] : n]                 - columns remain in the global task queue, in topological order

    DAG definition:
        v depends on edge_dst[edge_begins[v] : edge_ends[v]]
        edge_dst[edge_begins[v] : edge_ends[v]] is an unordered subset of vertex_begin : vertex_delta : v
    
    typename LoadBalancer::weight_t                          is the data type of weights
    load_balancer.weight_in_subtree(v)                       returns the weight of the vertex v
    load_balancer.weight_in_queue(v)                         returns the weight of the vertex v
    load_balancer.queue_granularity()                        returns the granularity of a task in the queue

    returns the size of the task queue
    the columns of the i-th task is
        partitions[task_queue[i] : task_queue[i + 1]]
            where 0 <= i < return_value 
               && task_queue[i + 1] - task_queue[i] <= load_balancer.granularity()
 */
template <typename LoadBalancer>
static int
tree_schedule(int nproc, int vertex_begin, int vertex_end, int vertex_delta,
              ConstBiasArray<long> edge_begins, const long *edge_ends, const int *edge_dst,
              int *part_ptr /* output: length nproc + 1 */,
              int *partitions /* output: length n */,
              int *&task_queue /* output: allocated by `new`*/,
              const LoadBalancer &load_balancer)
{
    using weight_t = typename LoadBalancer::weight_t;
    using unique_wt_ptr = std::unique_ptr<weight_t[]>;
    int n = (vertex_begin > vertex_end) ? (vertex_begin - vertex_end) : (vertex_end - vertex_begin);
    unique_int_ptr parent(new int[n]);
    partition_subtree(n, nproc, vertex_begin, vertex_end, vertex_delta, edge_begins, edge_ends, edge_dst,
                      part_ptr, partitions, parent.get(),
                      load_balancer);

    unique_int_ptr col2sup(new int[n]);
    unique_int_ptr sup2col(new int[n + 1]);
    int queue_part_begin = part_ptr[nproc];
    int nsuper = aggregate_nodes(n, parent.get(), partitions, queue_part_begin,
                                 sup2col.get(), col2sup.get(), load_balancer.queue_granularity());
    parent.reset();

    unique_wt_ptr level(new weight_t[nsuper]);
    unique_int_ptr sup_perm(new int[nsuper]);

    // level-set sort
    calculate_levels(nsuper, col2sup.get(), sup2col.get(), partitions, edge_begins, edge_ends, edge_dst, level.get(),
                     load_balancer);
    for (int i = 0; i < nsuper; ++i)
    {
        sup_perm[i] = i;
    }
    sort_by_rank_ascend(sup_perm.get(), sup_perm.get() + nsuper, level.get());
    level.reset();
    task_queue = new int[nsuper + 1];
    build_queue(n, nsuper, sup2col.get(), sup_perm.get(), task_queue, partitions, queue_part_begin, col2sup.get());
    return nsuper;
}

#endif
