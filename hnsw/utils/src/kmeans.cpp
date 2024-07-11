#include "fmt/core.h"
#include <atomic>
#include <limits>
#include <mutex>
#include <utils/dist_func.h>
#include <utils/kmeans.h>
#include <utils/random_utils.h>
#include <vector>

namespace utils {
void kmeans(std::vector<std::vector<float>> &train,
            std::vector<std::vector<float>> &centroid, unsigned &cluster_num,
            const unsigned kmeans_iter) {
  size_t data_num = train.size();
  size_t data_dim = train[0].size();
  fmt::println("{} cluster kmeans", cluster_num);
  fmt::println("Data Number: {}, Data Dim: {}", data_num, data_dim);
  unsigned interval = data_num / cluster_num;
  std::vector<std::vector<unsigned>> t_ivf(cluster_num);
  std::vector<std::vector<float>> t_centroid(cluster_num);
  std::vector<unsigned> t_id(data_num);

  std::vector<std::pair<float, unsigned>> norm_id(data_num);
#pragma omp parallel for
  for (size_t i = 0; i < data_num; ++i) {
    norm_id[i].first = utils::NormSqrT<float>(train[i].data(), &data_dim);
    norm_id[i].second = i;
  }
  // for (int i=0; i<10; ++i) {
  //     std::cout << i << " : kmeans norm: " << norm_id[i].first << std::endl;
  // }
  std::sort(norm_id.begin(), norm_id.end());

  float avg_norm = 0;
#pragma omp parallel for reduction(+ : avg_norm)
  for (size_t i = 0; i < data_num; ++i) {
    avg_norm += norm_id[i].first;
  }
  fmt::println("Avg Norm: {}, Min Norm: {}, Max Norm: {}", avg_norm / data_num,
               norm_id[0].first, norm_id[data_num - 1].first);

#pragma omp parallel for
  for (size_t i = 0; i < cluster_num; ++i) {
    t_centroid[i].assign(train[norm_id[i * interval].second].begin(),
                         train[norm_id[i * interval].second].end());
  }

  std::vector<bool> centroid_empty(cluster_num, false);
  float g_err = std::numeric_limits<float>::max();
  unsigned iter = kmeans_iter;
  while (iter) {
#pragma omp parallel for
    for (size_t i = 0; i < data_num; ++i) {
      float min_dist = std::numeric_limits<float>::max();
      for (size_t j = 0; j < cluster_num; ++j) {
        if (centroid_empty[j])
          continue;
        float dist =
            utils::L2Sqr(train[i].data(), t_centroid[j].data(), &data_dim);
        if (dist < min_dist) {
          t_id[i] = j;
          min_dist = dist;
        }
      }
    }

#pragma omp parallel for
    for (size_t i = 0; i < cluster_num; ++i) {
      t_ivf[i].clear();
    }

    for (size_t i = 0; i < data_num; ++i) {
      t_ivf[t_id[i]].push_back(i);
    }

    std::atomic<unsigned> c_num = 0;
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < cluster_num; ++i) {
      if (t_ivf[i].size() <= 1) {
        t_centroid[i] = train[utils::gen_rand(data_num)];
      } else {
        t_centroid[i] = train[t_ivf[i][0]];
        for (size_t j = 1; j < t_ivf[i].size(); ++j) {
          utils::Add<float>(train[t_ivf[i][j]].data(), t_centroid[i].data(),
                            data_dim);
        }
        for (size_t j = 0; j < data_dim; ++j) {
          t_centroid[i][j] /= t_ivf[i].size();
        }
        c_num.fetch_add(1, std::memory_order_relaxed);
      }
    }

    std::vector<float> err_clusters(cluster_num, 0);
#pragma omp parallel for
    for (size_t i = 0; i < cluster_num; ++i) {
      if (t_ivf[i].size()) {
        float err = 0;
        for (size_t j = 0; j < t_ivf[i].size(); ++j) {
          err += utils::L2Sqr(t_centroid[i].data(), train[t_ivf[i][j]].data(),
                              &data_dim);
        }
        err_clusters[i] = err / t_ivf[i].size();
      }
    }
    std::sort(err_clusters.begin(), err_clusters.end());

    float avg_err = 0;
    for (size_t i = 0; i < cluster_num; ++i) {
      if (t_ivf[i].size()) {
        avg_err += err_clusters[i];
      }
    }
    avg_err /= c_num;
    std::cout << "Iter: " << iter-- << std::endl;
    std::cout << "Avg Err: " << avg_err << ", Min Err: " << err_clusters[0]
              << ", Max Err: " << err_clusters[c_num - 1] << std::endl;
    if (avg_err < g_err) {
      g_err = avg_err;
    } else {
      break;
    }
  } // while(iter)

  unsigned num_cluster = 0;
  centroid.clear();
  for (size_t i = 0; i < cluster_num; ++i) {
    if (centroid_empty[i])
      continue;
    centroid.push_back(t_centroid[i]);
    ++num_cluster;
  }
  cluster_num = num_cluster;

  std::cout << "Final cluster number " << cluster_num << std::endl;
}

void kmeans_assign(const std::vector<std::vector<float>> &centroid,
                   std::vector<std::vector<float>> &data,
                   std::vector<std::vector<unsigned int>> &bucket_ids,
                   std::vector<uint8_t> &id) {
  size_t data_num = data.size();
  size_t data_dim = data[0].size();
#pragma omp parallel for
  for (size_t i = 0; i < data_num; ++i) {
    float min_dist = std::numeric_limits<float>::max();
    for (size_t j = 0; j < centroid.size(); ++j) {
      float dist = utils::L2Sqr(data[i].data(), centroid[j].data(), &data_dim);
      if (dist < min_dist) {
        id[i] = j;
        min_dist = dist;
      }
    }
  }

  bucket_ids.resize(centroid.size());

#pragma omp parallel for
  for (size_t i = 0; i < centroid.size(); ++i) {
    bucket_ids[i].clear();
  }

  for (size_t i = 0; i < data_num; ++i) {
    bucket_ids[id[i]].push_back(i);
  }
}

void kmeans_assign(const std::vector<std::vector<float>> &centroid,
                   std::vector<std::vector<float>> &data,
                   std::vector<std::vector<unsigned int>> &bucket_ids,
                   std::vector<unsigned> &id) {
  size_t data_num = data.size();
  size_t data_dim = data[0].size();
#pragma omp parallel for
  for (size_t i = 0; i < data_num; ++i) {
    float min_dist = std::numeric_limits<float>::max();
    for (size_t j = 0; j < centroid.size(); ++j) {
      float dist = utils::L2Sqr(data[i].data(), centroid[j].data(), &data_dim);
      if (dist < min_dist) {
        id[i] = j;
        min_dist = dist;
      }
    }
  }

  // bucket_ids.resize(centroid.size());

  std::vector<std::vector<unsigned>> t_bucket_ids(centroid.size());

  for (size_t i = 0; i < data_num; ++i) {
    t_bucket_ids[id[i]].push_back(i);
  }

  for (size_t i = 0; i < centroid.size(); ++i) {
    if (t_bucket_ids[i].size()) {
      bucket_ids.push_back(t_bucket_ids[i]);
    }
  }
}

void kmeans(const float *data, const size_t data_num, const size_t data_dim,
            std::vector<std::vector<float>> &centroids, unsigned &cluster_num,
            const unsigned kmeans_iter) {
  std::vector<std::vector<float>> train(data_num, std::vector<float>(data_dim));
#pragma omp parallel for
  for (size_t i = 0; i < data_num; ++i) {
    const float *v_begin = data + i * data_dim;
    train[i].assign(v_begin, v_begin + data_dim);
  }

  unsigned interval = data_num / cluster_num;
  std::vector<std::vector<unsigned>> t_ivf(cluster_num);
  std::vector<std::vector<float>> t_centroid(cluster_num);
  std::vector<unsigned> t_id(data_num);

  std::vector<std::pair<float, unsigned>> norm_id(data_num);
#pragma omp parallel for
  for (size_t i = 0; i < data_num; ++i) {
    norm_id[i].first = utils::NormSqrT<float>(train[i].data(), &data_dim);
    norm_id[i].second = i;
  }
  std::sort(norm_id.begin(), norm_id.end());

  float avg_norm = 0;
#pragma omp parallel for reduction(+ : avg_norm)
  for (size_t i = 0; i < data_num; ++i) {
    avg_norm += norm_id[i].first;
  }

  fmt::println("Avg Norm: {}, Min Norm: {}, Max Norm: {}", avg_norm / data_num,
               norm_id[0].first, norm_id[data_num - 1].first);

#pragma omp parallel for
  for (size_t i = 0; i < cluster_num; ++i) {
    t_centroid[i].assign(train[norm_id[i * interval].second].begin(),
                         train[norm_id[i * interval].second].end());
  }

  std::vector<bool> centroid_empty(cluster_num, false);
  float g_err = std::numeric_limits<float>::max();
  unsigned iter = kmeans_iter;
  while (iter) {
#pragma omp parallel for
    for (size_t i = 0; i < data_num; ++i) {
      float min_dist = std::numeric_limits<float>::max();
      for (size_t j = 0; j < cluster_num; ++j) {
        if (centroid_empty[j])
          continue;
        float dist =
            utils::L2Sqr(train[i].data(), t_centroid[j].data(), &data_dim);
        if (dist < min_dist) {
          t_id[i] = j;
          min_dist = dist;
        }
      }
    }

#pragma omp parallel for
    for (size_t i = 0; i < cluster_num; ++i) {
      t_ivf[i].clear();
    }

    for (size_t i = 0; i < data_num; ++i) {
      t_ivf[t_id[i]].push_back(i);
    }

    std::vector<std::pair<unsigned, unsigned>> bucket_size(cluster_num);
#pragma omp parallel for
    for (size_t i = 0; i < cluster_num; ++i) {
      bucket_size[i].first = t_ivf[i].size();
      bucket_size[i].second = i;
    }
    std::sort(bucket_size.begin(), bucket_size.end(),
              [](const std::pair<unsigned, unsigned> &a,
                 const std::pair<unsigned, unsigned> &b) {
                return a.first > b.first;
              });

    std::cout << "max bucket size: " << bucket_size[0].first << std::endl;
    std::cout << "min bucket size: " << bucket_size[cluster_num - 1].first
              << std::endl;

    unsigned avg_num = data_num / cluster_num / 2;

    unsigned c_num = 0;
    unsigned b_id = 0;
    // #pragma omp parallel for
    for (size_t i = 0; i < cluster_num; ++i) {
      if ((iter == kmeans_iter && t_ivf[i].size() <= 1) ||
          (iter < kmeans_iter && t_ivf[i].size() <= 2)) {
        int r_id = utils::gen_rand(bucket_size[b_id].first - 1);
        t_centroid[i] = train[t_ivf[bucket_size[b_id].second][r_id]];
        ++b_id;
      } else {
        t_centroid[i] = train[t_ivf[i][0]];
        for (size_t j = 1; j < t_ivf[i].size(); ++j) {
          utils::Add<float>(train[t_ivf[i][j]].data(), t_centroid[i].data(),
                            data_dim);
        }
        for (size_t j = 0; j < data_dim; ++j) {
          t_centroid[i][j] /= t_ivf[i].size();
        }
        ++c_num;
      }
    }

    std::vector<float> err_clusters(cluster_num, 0);
#pragma omp parallel for
    for (size_t i = 0; i < cluster_num; ++i) {
      if (t_ivf[i].size()) {
        float err = 0;
        for (size_t j = 0; j < t_ivf[i].size(); ++j) {
          err += utils::L2Sqr(t_centroid[i].data(), train[t_ivf[i][j]].data(),
                              &data_dim);
        }
        err_clusters[i] = err / t_ivf[i].size();
      }
    }
    std::sort(err_clusters.begin(), err_clusters.end());

    float avg_err = 0;
    for (size_t i = 0; i < cluster_num; ++i) {
      if (t_ivf[i].size()) {
        avg_err += err_clusters[i];
      }
    }
    avg_err /= c_num;
    std::cout << "Iter: " << iter-- << std::endl;

    fmt::println("Avg Err: {}, Min Err: {}, Max Err: {}", avg_err,
                 err_clusters[0], err_clusters[c_num - 1]);
    // if (avg_err < g_err) {
    //     g_err = avg_err;
    // } else {
    //     break;
    // }
  } // while(iter)

  for (size_t i = 0; i < cluster_num; ++i) {
    if (t_ivf[i].size() > 1) {
      centroids.emplace_back(t_centroid[i]);
    }
  }
}

void cos_split_kmeans(std::vector<std::vector<float>> &train,
                      std::vector<std::vector<float>> &centroid,
                      // std::vector<std::vector<unsigned>>& bucket_ids,
                      unsigned cluster_num, const unsigned kmeans_iter) {
  size_t data_num = train.size();
  size_t data_dim = train[0].size();
  const size_t avg_bucket_size = data_num / cluster_num;
  fmt::println("{} cluster kmeans", cluster_num);
  fmt::println("Data Number: {}, Data Dim: {}", data_num, data_dim);
  unsigned interval = data_num / cluster_num;
  std::vector<std::vector<unsigned>> t_ivf(cluster_num);
  std::vector<std::vector<float>> t_centroid(cluster_num);

#pragma omp parallel for
  for (std::size_t i = 0; i < cluster_num; ++i) {
    t_centroid[i].assign(train[i * interval].begin(),
                         train[i * interval].end());
  }

  std::vector<unsigned> data_bucket(data_num);
  unsigned iter = kmeans_iter;
  while (iter) {
#pragma omp parallel for
    for (std::size_t i = 0; i < data_num; ++i) {
      float max_cos = -1.0;
      for (std::size_t j = 0; j < cluster_num; ++j) {
        // if (bucket_empty[j]) continue;
        float dist = utils::InnerProduct(train[i].data(), t_centroid[j].data(),
                                         &data_dim);
        if (dist > max_cos) {
          data_bucket[i] = j;
          max_cos = dist;
        }
      }
    }

#pragma omp parallel for
    for (std::size_t i = 0; i < cluster_num; ++i) {
      t_ivf[i].clear();
    }

    for (std::size_t i = 0; i < data_num; ++i) {
      t_ivf[data_bucket[i]].push_back(i);
    }

    std::vector<std::pair<unsigned, unsigned>> bucket_size(cluster_num);
#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < cluster_num; ++i) {
      bucket_size[i].first = t_ivf[i].size();
      bucket_size[i].second = i;
    }
    std::sort(bucket_size.begin(), bucket_size.end(),
              [](const std::pair<unsigned, unsigned> &a,
                 const std::pair<unsigned, unsigned> &b) {
                return a.first > b.first;
              });
    fmt::println("Max bucket size: {}", bucket_size[0].first);
    fmt::println("Min bucket size: {}", bucket_size[cluster_num - 1].first);

    // #pragma omp parallel for schedule(dynamic)
    unsigned large_bid = 0;
    unsigned large_c_num = 1;
    unsigned small_bucket_cnt = 0;
    std::vector<bool> bucket_flags(cluster_num, false);
    for (std::size_t i = 0; i < cluster_num; ++i) {
      if (iter < 5 && t_ivf[i].size() < 5) {
        std::vector<std::vector<float>> split_centriod;
        if (bucket_size[large_bid].first < avg_bucket_size * 5)
          large_bid++;
        if (bucket_size[large_bid].first > avg_bucket_size * 5) {
          unsigned large_size = bucket_size[large_bid].first;
          unsigned b_size = split_bucket(split_centriod, train, t_ivf[i]);
          t_centroid[large_bid] = split_centriod[0];
          bucket_size[large_bid].first = b_size;
          t_centroid[i] = split_centriod[1];
          bucket_size[i].first = large_size - b_size;
          bucket_flags[i] = true;
        }
        // bucket_flags[i] = true;
        // int r_id = utils::gen_rand(bucket_size[large_bid].first-1);
        // t_centroid[i] = train[t_ivf[bucket_size[large_bid].second][r_id]];
        // ++large_c_num;
        // ++small_bucket_cnt;
        // if (avg_bucket_size*3 > bucket_size[large_bid].first/large_c_num) {
        //     ++large_bid;
        //     large_c_num = 1;
        // }
      } else {
        t_centroid[i] = train[t_ivf[i][0]];
        for (std::size_t j = 1; j < t_ivf[i].size(); ++j) {
          utils::Add<float>(train[t_ivf[i][j]].data(), t_centroid[i].data(),
                            data_dim);
        }
        float norm = utils::NormSqrT<float>(t_centroid[i].data(), &data_dim);
        for (std::size_t j = 0; j < data_dim; ++j) {
          t_centroid[i][j] /= norm;
        }
      }
    }

    std::mutex mutex;
    std::vector<float> err_clusters;
#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < cluster_num; ++i) {
      if (bucket_flags[i])
        continue;
      float err = 0;
      for (const auto &j : t_ivf[i]) {
        err += utils::InnerProduct(t_centroid[i].data(), train[j].data(),
                                   &data_dim);
      }
      {
        std::lock_guard<std::mutex> lock(mutex);
        err_clusters.push_back(err / t_ivf[i].size());
      }
    }
    std::sort(err_clusters.begin(), err_clusters.end());

    float avg_err = 0;
#pragma omp parallel for reduction(+ : avg_err)
    for (std::size_t i = 0; i < cluster_num; ++i) {
      if (bucket_flags[i])
        continue;
      avg_err += err_clusters[i];
    }
    avg_err /= (cluster_num - small_bucket_cnt);
    fmt::println("Iter: {}, cluster_num: {}", iter--,
                 cluster_num - small_bucket_cnt);
    fmt::println("Avg Err: {}, Min Err: {}, Max Err: {}", avg_err,
                 err_clusters[0], err_clusters[cluster_num - 1]);
  } // while(iter)

  for (std::size_t i = 0; i < cluster_num; ++i) {
    centroid.emplace_back(t_centroid[i]);
  }
}

void cos_kmeans(std::vector<std::vector<float>> &train,
                std::vector<std::vector<float>> &centroid, unsigned cluster_num,
                const unsigned kmeans_iter) {
  size_t data_num = train.size();
  size_t data_dim = train[0].size();
  const size_t avg_bucket_size = data_num / cluster_num;
  fmt::println("{} cluster kmeans", cluster_num);
  fmt::println("Data Number: {}, Data Dim: {}", data_num, data_dim);
  unsigned interval = data_num / cluster_num;
  std::vector<std::vector<unsigned>> t_ivf(cluster_num);
  std::vector<std::vector<float>> t_centroid(cluster_num);

#pragma omp parallel for
  for (std::size_t i = 0; i < cluster_num; ++i) {
    t_centroid[i].assign(train[i * interval].begin(),
                         train[i * interval].end());
  }

  std::vector<unsigned> data_bucket(data_num);
  unsigned iter = kmeans_iter;
  while (iter) {
#pragma omp parallel for
    for (std::size_t i = 0; i < data_num; ++i) {
      float min_dist = std::numeric_limits<float>::max();
      for (std::size_t j = 0; j < cluster_num; ++j) {
        // if (bucket_empty[j]) continue;
        float dist =
            utils::L2Sqr(train[i].data(), t_centroid[j].data(), &data_dim);
        if (dist < min_dist) {
          data_bucket[i] = j;
          min_dist = dist;
        }
      }
    }

#pragma omp parallel for
    for (std::size_t i = 0; i < cluster_num; ++i) {
      t_ivf[i].clear();
    }

    for (std::size_t i = 0; i < data_num; ++i) {
      t_ivf[data_bucket[i]].push_back(i);
    }

    std::vector<std::pair<unsigned, unsigned>> bucket_size(cluster_num);
#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < cluster_num; ++i) {
      bucket_size[i].first = t_ivf[i].size();
      bucket_size[i].second = i;
    }
    std::sort(bucket_size.begin(), bucket_size.end(),
              [](const std::pair<unsigned, unsigned> &a,
                 const std::pair<unsigned, unsigned> &b) {
                return a.first > b.first;
              });
    fmt::println("Max bucket size: {}", bucket_size[0].first);
    fmt::println("Min bucket size: {}", bucket_size[cluster_num - 1].first);

    unsigned large_bid = 0;
    unsigned large_c_num = 1;
    unsigned small_bucket_cnt = 0;
    std::vector<bool> bucket_flags(cluster_num, false);
    for (std::size_t i = 0; i < cluster_num; ++i) {
      if (iter && t_ivf[i].size() < 3) {
        bucket_flags[i] = true;
        int r_id = utils::gen_rand(bucket_size[large_bid].first - 1);
        t_centroid[i] = train[t_ivf[bucket_size[large_bid].second][r_id]];
        ++large_c_num;
        ++small_bucket_cnt;
        if (avg_bucket_size * 3 > bucket_size[large_bid].first / large_c_num) {
          ++large_bid;
          large_c_num = 1;
        }
      } else {
        t_centroid[i] = train[t_ivf[i][0]];
        for (std::size_t j = 1; j < t_ivf[i].size(); ++j) {
          utils::Add<float>(train[t_ivf[i][j]].data(), t_centroid[i].data(),
                            data_dim);
        }
        float norm = utils::NormSqrT<float>(t_centroid[i].data(), &data_dim);
        for (std::size_t j = 0; j < data_dim; ++j) {
          t_centroid[i][j] /= norm;
        }
      }
    }

    std::mutex mutex;
    std::vector<float> err_clusters;
#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < cluster_num; ++i) {
      if (bucket_flags[i])
        continue;
      float err = 0;
      for (const auto &j : t_ivf[i]) {
        err += utils::L2Sqr(t_centroid[i].data(), train[j].data(), &data_dim);
      }
      {
        std::lock_guard<std::mutex> lock(mutex);
        err_clusters.push_back(err / t_ivf[i].size());
      }
    }
    std::sort(err_clusters.begin(), err_clusters.end());

    float avg_err = 0;
#pragma omp parallel for reduction(+ : avg_err)
    for (std::size_t i = 0; i < cluster_num; ++i) {
      if (bucket_flags[i])
        continue;
      avg_err += err_clusters[i];
    }
    avg_err /= (cluster_num - small_bucket_cnt);
    fmt::println("Iter: {}, cluster_num: {}", iter--,
                 cluster_num - small_bucket_cnt);
    fmt::println("Avg Err: {}, Min Err: {}, Max Err: {}", avg_err,
                 err_clusters[0], err_clusters[cluster_num - 1]);
  } // while(iter)

  for (std::size_t i = 0; i < cluster_num; ++i) {
    centroid.emplace_back(t_centroid[i]);
  }
  fmt::println("Cos kMeans Done");
}

int split_bucket(std::vector<std::vector<float>> &t_centroid,
                 const std::vector<std::vector<float>> &train,
                 const std::vector<unsigned> &split_ivf
                 //   const unsigned split_bucket_id,
                 //   const unsigned merge_bucket_id
) {
  std::size_t data_num = split_ivf.size();
  std::size_t data_dim = train[0].size();
  std::vector<std::vector<float>> t_data(data_num);
  t_centroid.resize(2);

  for (const auto &i : split_ivf) {
    t_data[i].assign(train[i].begin(), train[i].end());
  }

  unsigned first_id, second_id;
  float centroid_dist = 1;
  unsigned iter = 5;
  while (iter--) {
    unsigned t_first_id = utils::gen_rand(data_num);
    unsigned t_second_id;
    float t_dist = 1;
    for (std::size_t i = 0; i < data_num; ++i) {
      float tmp_dist = utils::InnerProduct(
          t_data[i].data(), t_data[t_first_id].data(), &data_dim);
      if (tmp_dist < t_dist) {
        t_dist = tmp_dist;
        t_second_id = i;
      }
    }
    if (t_dist < centroid_dist) {
      centroid_dist = t_dist;
      first_id = t_first_id;
      second_id = t_second_id;
    }
  }

  t_centroid[0].assign(train[first_id].begin(), train[first_id].end());
  t_centroid[1].assign(train[second_id].begin(), train[second_id].end());

  std::vector<unsigned> data_bucket(data_num);
  std::vector<std::vector<unsigned>> t_ivf(2);
  iter = 10;
  while (iter--) {
#pragma omp parallel for
    for (std::size_t i = 0; i < data_num; ++i) {
      float min_cos = -1.0;
      for (std::size_t j = 0; j < 2; ++j) {
        float dist = utils::InnerProduct(t_data[i].data(), t_centroid[j].data(),
                                         &data_dim);
        if (dist > min_cos) {
          data_bucket[i] = j;
          min_cos = dist;
        }
      }
    }
#pragma omp parallel for
    for (std::size_t i = 0; i < 2; ++i) {
      t_ivf[i].clear();
    }

    for (std::size_t i = 0; i < data_num; ++i) {
      t_ivf[data_bucket[i]].push_back(i);
    }

    for (std::size_t i = 0; i < 2; ++i) {
      t_centroid[i] = t_data[t_ivf[i][0]];
      for (const auto &j : t_ivf[i]) {
        utils::Add<float>(t_data[j].data(), t_centroid[i].data(), data_dim);
      }
      float norm = utils::NormSqrT<float>(t_centroid[i].data(), &data_dim);
      for (std::size_t j = 0; j < data_dim; ++j) {
        t_centroid[i][j] /= norm;
      }
    }
  }

#pragma omp parallel for
  for (std::size_t i = 0; i < data_num; ++i) {
    float min_cos = -1.0;
    for (std::size_t j = 0; j < 2; ++j) {
      float dist = utils::InnerProduct(t_data[i].data(), t_centroid[j].data(),
                                       &data_dim);
      if (dist > min_cos) {
        data_bucket[i] = j;
        min_cos = dist;
      }
    }
  }
#pragma omp parallel for
  for (std::size_t i = 0; i < 2; ++i) {
    t_ivf[i].clear();
  }

  for (std::size_t i = 0; i < data_num; ++i) {
    t_ivf[data_bucket[i]].push_back(i);
  }
  return t_ivf[0].size() > t_ivf[1].size() ? t_ivf[0].size() : t_ivf[1].size();
}
} // namespace utils