#pragma once

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>

#include <cassert>
#include <cmath>
#include <unistd.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <fmt/core.h>
#include <fmt/chrono.h>
#include <fmt/format.h>

#include "utils/timer.h"

namespace utils{

std::string get_data_name(const std::string &kDatasetPath);

void read_fvecs(
    const std::string& filename, 
    float*& data, 
    std::size_t& num,
    std::size_t& dim);

void read_ivecs(
    const std::string& filename, 
    unsigned*& data, 
    std::size_t& num,
    std::size_t& dim);

// int read_bin(const int n, const int d, const std::string fname, float *data);

int read_bin_data(                  // read bin data from disk
    const int   n,                            // number of data points
    const int   d,                            // data dimension
    const char *fname,                  // address of data set
    float *min_coord,                   // min coordinates (return)
    float *max_coord,                   // max coordinates (return)
    float *data                         // data (return)$
);

int read_bin_data_and_normalize(    // read bin data & normalize
    int   n,                            // number of data points
    int   d,                            // data dimension
    const char *fname,                  // address of data set
    float *data                        // data (return)
);

template<typename T>
void save_res(const std::string& filename, const std::vector<std::vector<T>> &res) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    if (!out) {
        std::cerr << "Open File " << filename << " Fail" << std::endl;
        exit(1);
    }
    for (std::size_t i=0; i<res.size(); ++i) {
        std::size_t r_size = res[i].size() * sizeof(T);
        out.write((char *)(res[i].data()), r_size);
    }
    // assert(final_graph_.size() == nd_);
    // unsigned GK = (unsigned) final_graph_[0].size();
    // for (unsigned i = 0; i < nd_; i++) {
    //     out.write((char *) &GK, sizeof(unsigned));
    //     out.write((char *) final_graph_[i].data(), GK * sizeof(unsigned));
    // }
    out.close();
}

template<typename T>
void write_binary(std::ostream &out, const T &podRef) {
    out.write((char *) &podRef, sizeof(T));
}

template<typename T>
void read_binary(std::istream &in, T &podRef) {
    in.read((char *) &podRef, sizeof(T));
}

template<typename T>
void read_data(const std::string kFilePath,
               const std::size_t kRow,
               const std::size_t kCol,
               T* &data) {
    fmt::println("Read Data from: {}", kFilePath);
    std::ifstream fin(kFilePath, std::ios::binary);
    if (!fin) {
        std::cerr << "Open File " << kFilePath << " Fail" << std::endl;
        exit(1);
    }
    int npts, dim;
    fin.read((char *)&npts, sizeof(int));
    fin.read((char *)&dim, sizeof(int));
    std::size_t data_size = kRow * kCol;
    data = new T[data_size];
    fin.read((char *) data, data_size * sizeof(T));
    fin.close();
}

void read_data(size_t &num,
               size_t &dim,
               const std::string& kFrontPath,
               const std::string& kDataset,
               const std::string& kDataType,
               float*& data);

int read_bin(const int n,
             const int d,
             const std::string& kFilePath,
             float*& data);

void read_bin(const unsigned num,
              const unsigned dim,
              const std::string &kFilePath,
              float **data);

void read_bin(const size_t n,
              const size_t d,
              const std::string& fname,
              std::vector<std::vector<float>>& data);

void read_p2h_data(const unsigned num,
                   const unsigned dim,
                   const std::string &kFilePath,
                   float*& data);

void read_p2h_data(const unsigned num,
                   const unsigned dim,
                   const std::string &kFilePath,
                   float** data);

void load_knng(
    const std::string kGraphPath,
    std::vector<std::vector<unsigned>> &graph);

void load_merge_knng(
    const std::string kGraphPath,
    std::vector<std::vector<unsigned>> &graph
);


void load_pure_hnsw(
    const std::string& kGraphPath,
    std::vector<std::vector<unsigned>>& graph
);

void save_graph(
    const std::string &kGraphPath,
    const std::vector<std::vector<unsigned>> &graph
);

void load_graph(
    const std::string &kGraphPath,
    std::vector<std::vector<unsigned>> &graph
);

void get_union(
    const std::vector<unsigned> &vec1,
    const std::vector<unsigned> &vec2,
    std::vector<unsigned> &res);
}