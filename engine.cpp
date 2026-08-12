// YOU MAY EDIT THIS FILE
#include <vector>
#include <mpi.h>
#include <limits>
#include <algorithm>
#include <map>
#include <cstring>
#include "engine.h"

using namespace std;

void Engine::KNN(Params &p, std::vector<DataPoint> &dataset, std::vector<Query> &queries) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Phase 1: Broadcast params
    int meta[3];
    if (rank == 0) {
        meta[0] = p.num_data;
        meta[1] = p.num_queries;
        meta[2] = p.num_attrs;
    }
    MPI_Bcast(meta, 3, MPI_INT, 0, MPI_COMM_WORLD);
    int num_data    = meta[0];
    int num_queries = meta[1];
    int num_attrs   = meta[2];
    if (num_data == 0 || num_queries == 0) return;

    auto cmp = [](const pair<double,int>& a, const pair<double,int>& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second > b.second;
    };

    const int BATCH = 8;

    // Choose strategy: when Q is small relative to N, scatter data + bcast queries
    // reduces cross-node traffic. Otherwise keep current query distribution.
    if (num_queries * 2 < num_data) {
        // ===== DATA DISTRIBUTION: scatter data, broadcast queries, tree merge =====

        // Scatter dataset rows across ranks
        vector<int> d_counts(size), d_displs(size);
        for (int r = 0; r < size; r++)
            d_counts[r] = num_data / size + (r < num_data % size ? 1 : 0);
        d_displs[0] = 0;
        for (int r = 1; r < size; r++) d_displs[r] = d_displs[r-1] + d_counts[r-1];
        int local_n = d_counts[rank];
        int local_start = d_displs[rank];

        vector<int> sc_d(size), sd_d(size);
        for (int r = 0; r < size; r++) {
            sc_d[r] = d_counts[r] * num_attrs;
            sd_d[r] = d_displs[r] * num_attrs;
        }

        vector<double> flat_data;
        if (rank == 0) {
            flat_data.resize((long long)num_data * num_attrs);
            for (int i = 0; i < num_data; i++)
                for (int j = 0; j < num_attrs; j++)
                    flat_data[(long long)i * num_attrs + j] = dataset[i].attrs[j];
        }
        vector<double> local_data((long long)local_n * num_attrs);
        MPI_Scatterv(rank == 0 ? flat_data.data() : nullptr,
                     sc_d.data(), sd_d.data(), MPI_DOUBLE,
                     local_data.data(), local_n * num_attrs, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
        if (rank != 0) { flat_data.clear(); flat_data.shrink_to_fit(); }

        // Broadcast all queries (small: Q × A × 8B)
        vector<int> q_ints(num_queries * 2);
        vector<double> q_dbls((long long)num_queries * num_attrs);
        if (rank == 0) {
            for (int i = 0; i < num_queries; i++) {
                q_ints[i*2]   = queries[i].id;
                q_ints[i*2+1] = queries[i].k;
                for (int j = 0; j < num_attrs; j++)
                    q_dbls[(long long)i * num_attrs + j] = queries[i].attrs[j];
            }
        }
        MPI_Bcast(q_ints.data(), num_queries * 2, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(q_dbls.data(), (int)((long long)num_queries * num_attrs),
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);

        int k_max = 0;
        for (int i = 0; i < num_queries; i++) k_max = max(k_max, q_ints[i*2+1]);

        long long total_entries = (long long)num_queries * k_max;
        vector<double> local_dists(total_entries, numeric_limits<double>::infinity());
        vector<int> local_idxs(total_entries, -1);

        // Local compute: all queries vs local data (dataset fits in L2)
        {
            vector<pair<double,int>> heaps[BATCH];
            for (int b = 0; b < BATCH; b++) heaps[b].reserve(k_max);
            vector<double> qa_buf((long long)BATCH * num_attrs);

            const double* da_base = local_data.data();
            const double* qa_base = q_dbls.data();

            for (int qi_base = 0; qi_base < num_queries; qi_base += BATCH) {
                int batch = min(BATCH, num_queries - qi_base);

                int takes[BATCH];
                for (int b = 0; b < batch; b++) {
                    const double* src = qa_base + (long long)(qi_base + b) * num_attrs;
                    double* dst = qa_buf.data() + (long long)b * num_attrs;
                    for (int a = 0; a < num_attrs; a++) dst[a] = src[a];
                    takes[b] = min(q_ints[(qi_base+b)*2+1], local_n);
                    heaps[b].clear();
                }

                const double* qas[BATCH];
                for (int b = 0; b < batch; b++)
                    qas[b] = qa_buf.data() + (long long)b * num_attrs;

                double s[BATCH];

                for (int di = 0; di < local_n; di++) {
                    const double* da = da_base + (long long)di * num_attrs;
                    int global_di = local_start + di;

                    for (int b = 0; b < batch; b++) s[b] = 0.0;
                    for (int a = 0; a < num_attrs; a++) {
                        double da_a = da[a];
                        for (int b = 0; b < batch; b++) {
                            double diff = qas[b][a] - da_a;
                            s[b] += diff * diff;
                        }
                    }

                    for (int b = 0; b < batch; b++) {
                        auto& h = heaps[b];
                        int take = takes[b];
                        if ((int)h.size() < take) {
                            h.push_back({s[b], global_di});
                            if ((int)h.size() == take)
                                make_heap(h.begin(), h.end(), cmp);
                        } else if (take > 0 && (s[b] < h[0].first || (s[b] == h[0].first && global_di > h[0].second))) {
                            pop_heap(h.begin(), h.end(), cmp);
                            h.back() = {s[b], global_di};
                            push_heap(h.begin(), h.end(), cmp);
                        }
                    }
                }

                for (int b = 0; b < batch; b++) {
                    int qi = qi_base + b;
                    auto& h = heaps[b];
                    if ((int)h.size() == takes[b])
                        sort_heap(h.begin(), h.end(), cmp);
                    else
                        sort(h.begin(), h.end(), cmp);
                    for (int j = 0; j < (int)h.size(); j++) {
                        local_dists[(long long)qi*k_max + j] = h[j].first;
                        local_idxs [(long long)qi*k_max + j] = h[j].second;
                    }
                }
            }
        }
        local_data.clear(); local_data.shrink_to_fit();

        // Node-local communicator for intra-node tree merge
        MPI_Comm node_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                            rank, MPI_INFO_NULL, &node_comm);
        int node_rank, node_size;
        MPI_Comm_rank(node_comm, &node_rank);
        MPI_Comm_size(node_comm, &node_size);

        vector<double> tmp_dists(total_entries);
        vector<int> tmp_idxs(total_entries);
        vector<double> mrg_d(k_max);
        vector<int> mrg_i(k_max);

        // Intra-node tree reduction (shared memory, fast)
        for (int stride = 1; stride < node_size; stride *= 2) {
            if (node_rank % (2 * stride) == 0 && node_rank + stride < node_size) {
                MPI_Recv(tmp_dists.data(), (int)total_entries, MPI_DOUBLE,
                         node_rank + stride, 0, node_comm, MPI_STATUS_IGNORE);
                MPI_Recv(tmp_idxs.data(), (int)total_entries, MPI_INT,
                         node_rank + stride, 1, node_comm, MPI_STATUS_IGNORE);

                for (int qi = 0; qi < num_queries; qi++) {
                    int k = min(q_ints[qi*2+1], num_data);
                    long long base = (long long)qi * k_max;
                    double* ad = local_dists.data() + base;
                    int* ai = local_idxs.data() + base;
                    const double* bd = tmp_dists.data() + base;
                    const int* bi = tmp_idxs.data() + base;

                    int ia = 0, ib = 0, out = 0;
                    while (out < k) {
                        bool a_ok = (ia < k_max && ai[ia] != -1);
                        bool b_ok = (ib < k_max && bi[ib] != -1);
                        if (!a_ok && !b_ok) break;
                        bool take_a;
                        if (!b_ok) take_a = true;
                        else if (!a_ok) take_a = false;
                        else if (ad[ia] < bd[ib]) take_a = true;
                        else if (ad[ia] > bd[ib]) take_a = false;
                        else take_a = (ai[ia] > bi[ib]);

                        if (take_a) { mrg_d[out]=ad[ia]; mrg_i[out]=ai[ia]; ia++; }
                        else        { mrg_d[out]=bd[ib]; mrg_i[out]=bi[ib]; ib++; }
                        out++;
                    }
                    for (int j = out; j < k_max; j++) {
                        mrg_d[j] = numeric_limits<double>::infinity();
                        mrg_i[j] = -1;
                    }
                    memcpy(ad, mrg_d.data(), k_max * sizeof(double));
                    memcpy(ai, mrg_i.data(), k_max * sizeof(int));
                }
            } else if (node_rank % (2 * stride) == stride) {
                MPI_Send(local_dists.data(), (int)total_entries, MPI_DOUBLE,
                         node_rank - stride, 0, node_comm);
                MPI_Send(local_idxs.data(), (int)total_entries, MPI_INT,
                         node_rank - stride, 1, node_comm);
                break;
            }
        }

        // Cross-node merge: leaders send indices only (8MB vs 24MB full)
        MPI_Comm leaders_comm = MPI_COMM_NULL;
        MPI_Comm_split(MPI_COMM_WORLD, node_rank == 0 ? 0 : MPI_UNDEFINED,
                       rank, &leaders_comm);

        if (leaders_comm != MPI_COMM_NULL) {
            int leaders_rank, leaders_size;
            MPI_Comm_rank(leaders_comm, &leaders_rank);
            MPI_Comm_size(leaders_comm, &leaders_size);

            if (leaders_rank == 0) {
                vector<pair<double,int>> recomp(k_max);
                for (int src = 1; src < leaders_size; src++) {
                    MPI_Recv(tmp_idxs.data(), (int)total_entries, MPI_INT,
                             src, 0, leaders_comm, MPI_STATUS_IGNORE);

                    for (int qi = 0; qi < num_queries; qi++) {
                        int k = min(q_ints[qi*2+1], num_data);
                        long long base = (long long)qi * k_max;
                        const double* q_attrs = q_dbls.data() + (long long)qi * num_attrs;

                        // Recompute distances for received indices
                        int cnt = 0;
                        for (int j = 0; j < k_max; j++) {
                            int gidx = tmp_idxs[base + j];
                            if (gidx == -1) break;
                            const double* d_attrs = flat_data.data() + (long long)gidx * num_attrs;
                            double dist = 0.0;
                            for (int a = 0; a < num_attrs; a++) {
                                double diff = q_attrs[a] - d_attrs[a];
                                dist += diff * diff;
                            }
                            recomp[cnt++] = {dist, gidx};
                        }
                        sort(recomp.begin(), recomp.begin() + cnt, cmp);
                        for (int j = 0; j < cnt; j++) {
                            tmp_dists[base+j] = recomp[j].first;
                            tmp_idxs[base+j] = recomp[j].second;
                        }
                        for (int j = cnt; j < k_max; j++) {
                            tmp_dists[base+j] = numeric_limits<double>::infinity();
                            tmp_idxs[base+j] = -1;
                        }

                        // Merge with local results
                        double* ad = local_dists.data() + base;
                        int* ai = local_idxs.data() + base;

                        int ia = 0, ib = 0, out = 0;
                        while (out < k) {
                            bool a_ok = (ia < k_max && ai[ia] != -1);
                            bool b_ok = (ib < k_max && tmp_idxs[base+ib] != -1);
                            if (!a_ok && !b_ok) break;
                            bool take_a;
                            if (!b_ok) take_a = true;
                            else if (!a_ok) take_a = false;
                            else if (ad[ia] < tmp_dists[base+ib]) take_a = true;
                            else if (ad[ia] > tmp_dists[base+ib]) take_a = false;
                            else take_a = (ai[ia] > tmp_idxs[base+ib]);

                            if (take_a) { mrg_d[out]=ad[ia]; mrg_i[out]=ai[ia]; ia++; }
                            else { mrg_d[out]=tmp_dists[base+ib]; mrg_i[out]=tmp_idxs[base+ib]; ib++; }
                            out++;
                        }
                        for (int j = out; j < k_max; j++) {
                            mrg_d[j] = numeric_limits<double>::infinity();
                            mrg_i[j] = -1;
                        }
                        memcpy(ad, mrg_d.data(), k_max * sizeof(double));
                        memcpy(ai, mrg_i.data(), k_max * sizeof(int));
                    }
                }
            } else {
                MPI_Send(local_idxs.data(), (int)total_entries, MPI_INT,
                         0, 0, leaders_comm);
            }
            MPI_Comm_free(&leaders_comm);
        }

        MPI_Comm_free(&node_comm);

        // Output on rank 0
        if (rank == 0) {
            for (int qi = 0; qi < num_queries; qi++) {
                int k = queries[qi].k;
                long long base = (long long)qi * k_max;

                vector<pair<double,int>> candidates;
                candidates.reserve(k);
                for (int j = 0; j < k; j++) {
                    if (local_idxs[base+j] == -1) break;
                    candidates.push_back({local_dists[base+j], local_idxs[base+j]});
                }

                map<int,int> votes;
                for (auto& pr : candidates)
                    votes[dataset[pr.second].label]++;
                int best_label = -1, best_cnt = 0;
                for (auto& [lbl, cnt] : votes) {
                    if (cnt > best_cnt || (cnt == best_cnt && lbl > best_label)) {
                        best_cnt   = cnt;
                        best_label = lbl;
                    }
                }

                reportResult(queries[qi], candidates, best_label);
            }
        }

    } else {
        // ===== QUERY DISTRIBUTION: broadcast data, scatter queries =====

        vector<double> all_attrs((long long)num_data * num_attrs);
        if (rank == 0) {
            for (int i = 0; i < num_data; i++)
                for (int j = 0; j < num_attrs; j++)
                    all_attrs[(long long)i * num_attrs + j] = dataset[i].attrs[j];
        }
        MPI_Bcast(all_attrs.data(), num_data * num_attrs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        vector<int> q_counts(size), q_displs(size);
        for (int r = 0; r < size; r++)
            q_counts[r] = num_queries / size + (r < num_queries % size ? 1 : 0);
        q_displs[0] = 0;
        for (int r = 1; r < size; r++) q_displs[r] = q_displs[r-1] + q_counts[r-1];
        int local_q = q_counts[rank];

        vector<int> sc_qi(size), sd_qi(size), sc_qd(size), sd_qd(size);
        for (int r = 0; r < size; r++) {
            sc_qi[r] = q_counts[r] * 2;         sd_qi[r] = q_displs[r] * 2;
            sc_qd[r] = q_counts[r] * num_attrs;  sd_qd[r] = q_displs[r] * num_attrs;
        }

        vector<int>    q_int_all, q_int_local(local_q * 2);
        vector<double> q_dbl_all, q_dbl_local((long long)local_q * num_attrs);
        if (rank == 0) {
            q_int_all.resize(num_queries * 2);
            q_dbl_all.resize((long long)num_queries * num_attrs);
            for (int i = 0; i < num_queries; i++) {
                q_int_all[i*2]   = queries[i].id;
                q_int_all[i*2+1] = queries[i].k;
                for (int j = 0; j < num_attrs; j++)
                    q_dbl_all[(long long)i * num_attrs + j] = queries[i].attrs[j];
            }
        }
        MPI_Scatterv(rank==0 ? q_int_all.data() : nullptr,
                     sc_qi.data(), sd_qi.data(), MPI_INT,
                     q_int_local.data(), local_q*2, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(rank==0 ? q_dbl_all.data() : nullptr,
                     sc_qd.data(), sd_qd.data(), MPI_DOUBLE,
                     q_dbl_local.data(), local_q*num_attrs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        q_int_all.clear(); q_int_all.shrink_to_fit();
        q_dbl_all.clear(); q_dbl_all.shrink_to_fit();

        int k_max = 0;
        for (int qi = 0; qi < local_q; qi++) k_max = max(k_max, q_int_local[qi*2+1]);
        MPI_Allreduce(MPI_IN_PLACE, &k_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        vector<double> send_dists((long long)local_q * k_max, numeric_limits<double>::infinity());
        vector<int>    send_idxs ((long long)local_q * k_max, -1);

        vector<pair<double,int>> heaps[BATCH];
        for (int b = 0; b < BATCH; b++) heaps[b].reserve(k_max);
        vector<double> qa_buf((long long)BATCH * num_attrs);

        const double* da_base = all_attrs.data();
        const double* qa_base = q_dbl_local.data();

        for (int qi_base = 0; qi_base < local_q; qi_base += BATCH) {
            int batch = min(BATCH, local_q - qi_base);

            int takes[BATCH];
            for (int b = 0; b < batch; b++) {
                const double* src = qa_base + (long long)(qi_base + b) * num_attrs;
                double*       dst = qa_buf.data() + (long long)b * num_attrs;
                for (int a = 0; a < num_attrs; a++) dst[a] = src[a];
                takes[b] = min(q_int_local[(qi_base+b)*2+1], num_data);
                heaps[b].clear();
            }

            const double* qas[BATCH];
            for (int b = 0; b < batch; b++)
                qas[b] = qa_buf.data() + (long long)b * num_attrs;

            double s[BATCH];

            for (int di = 0; di < num_data; di++) {
                const double* da = da_base + (long long)di * num_attrs;

                for (int b = 0; b < batch; b++) s[b] = 0.0;
                for (int a = 0; a < num_attrs; a++) {
                    double da_a = da[a];
                    for (int b = 0; b < batch; b++) {
                        double diff = qas[b][a] - da_a;
                        s[b] += diff * diff;
                    }
                }

                for (int b = 0; b < batch; b++) {
                    auto& h = heaps[b];
                    int take = takes[b];
                    if ((int)h.size() < take) {
                        h.push_back({s[b], di});
                        if ((int)h.size() == take)
                            make_heap(h.begin(), h.end(), cmp);
                    } else if (take > 0 && (s[b] < h[0].first || (s[b] == h[0].first && di > h[0].second))) {
                        pop_heap(h.begin(), h.end(), cmp);
                        h.back() = {s[b], di};
                        push_heap(h.begin(), h.end(), cmp);
                    }
                }
            }

            for (int b = 0; b < batch; b++) {
                int qi = qi_base + b;
                auto& h = heaps[b];
                if ((int)h.size() == takes[b])
                    sort_heap(h.begin(), h.end(), cmp);
                else
                    sort(h.begin(), h.end(), cmp);
                for (int j = 0; j < (int)h.size(); j++) {
                    send_dists[(long long)qi*k_max + j] = h[j].first;
                    send_idxs [(long long)qi*k_max + j] = h[j].second;
                }
            }
        }
        all_attrs.clear(); all_attrs.shrink_to_fit();
        q_dbl_local.clear(); q_dbl_local.shrink_to_fit();

        vector<int> gcounts(size), gdispls(size);
        for (int r = 0; r < size; r++) gcounts[r] = q_counts[r] * k_max;
        gdispls[0] = 0;
        for (int r = 1; r < size; r++) gdispls[r] = gdispls[r-1] + gcounts[r-1];
        int local_send = local_q * k_max;

        vector<double> recv_dists;
        vector<int>    recv_idxs;
        if (rank == 0) {
            recv_dists.resize((long long)num_queries * k_max);
            recv_idxs .resize((long long)num_queries * k_max);
        }
        MPI_Gatherv(send_dists.data(), local_send, MPI_DOUBLE,
                    recv_dists.data(), gcounts.data(), gdispls.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        MPI_Gatherv(send_idxs.data(),  local_send, MPI_INT,
                    recv_idxs.data(),  gcounts.data(), gdispls.data(), MPI_INT,
                    0, MPI_COMM_WORLD);
        send_dists.clear(); send_dists.shrink_to_fit();
        send_idxs .clear(); send_idxs .shrink_to_fit();

        if (rank == 0) {
            for (int qi = 0; qi < num_queries; qi++) {
                int k = queries[qi].k;
                long long base = (long long)qi * k_max;

                vector<pair<double,int>> candidates;
                candidates.reserve(k);
                for (int j = 0; j < k; j++) {
                    if (recv_idxs[base+j] == -1) break;
                    candidates.push_back({recv_dists[base+j], recv_idxs[base+j]});
                }

                map<int,int> votes;
                for (auto& pr : candidates)
                    votes[dataset[pr.second].label]++;
                int best_label = -1, best_cnt = 0;
                for (auto& [lbl, cnt] : votes) {
                    if (cnt > best_cnt || (cnt == best_cnt && lbl > best_label)) {
                        best_cnt   = cnt;
                        best_label = lbl;
                    }
                }

                reportResult(queries[qi], candidates, best_label);
            }
        }
    }
}
