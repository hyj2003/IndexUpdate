#include "utils/io_utils.h"
#include "fmt/core.h"
#include <cstdlib>
#include <fstream>
#include <tuple>
#include <vector>
#include <set>

namespace utils {
void read_data(size_t &num,
               size_t &dim,
               const std::string& kFrontPath,
               const std::string& kDataset,
               const std::string& kDataType,
               float*& data) {
    const std::string kDataPath = fmt::format("{}/{}.{}", kFrontPath, kDataset, kDataType);
    // fmt::println("Read data from {}", kDataPath);
    if (kDataType=="fvecs") {
        read_fvecs(kDataPath, data, num, dim);
    } else if (kDataType=="bin") {
        read_bin(num, dim, kDataPath, data);
    } else if (kDataType=="ds") {
        read_p2h_data(num, dim, kDataPath, data);
    }
}

// void read_query(size_t &num,
//                 size_t &dim,
//                 const std::string& kFrontPath,
//                 const std::string& kDataset,
//                 const std::string& kQueryType,
//                 float*& query) {
//     const std::string kQueryPath = fmt::format("{}/{}.{}", kFrontPath, kDataset, kQueryType);
//     // fmt::println("Read data from {}", kDataPath);
//     if (kQueryType=="fvecs") {
//         read_fvecs(kQueryPath, query, num, dim);
//     } else if (kQueryType=="oq") {
//         read_bin(num, dim, kDataPath, data);
//     } else if (kDataType=="bs") {
//         read_p2h_data(num, dim, kDataPath, data);
//     }   
// }

std::string get_data_name(const std::string &kDatasetPath) {
    int pos = kDatasetPath.rfind('/');
    return kDatasetPath.substr(pos + 1);
}

// void read_fvecs(const std::string filename, float*& data, unsigned& num,unsigned& dim) {
void read_fvecs(
    const std::string& filename, 
    float*& data, 
    std::size_t& num,
    std::size_t& dim)
{
    fmt::println("load data from {}", filename);
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
    unsigned t_dim;
    in.read((char*)&t_dim,4);
    dim = t_dim;
    in.seekg(0,std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim+1) / 4);
    fmt::println("data dimension: {}, data number: {}", dim, num);
    data = new float[num * dim * sizeof(float)];
    
    in.seekg(0,std::ios::beg);
    for(size_t i = 0; i < num; i++){
      in.seekg(4,std::ios::cur);
      in.read((char*)(data+i*dim),dim*4);
    }
    in.close();
}

void read_ivecs(
    const std::string& filename, 
    unsigned*& data, 
    std::size_t& num,
    std::size_t& dim)
{
    fmt::println("load data from {}", filename);
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
    unsigned t_dim;
    in.read((char*)&t_dim,4);
    dim = t_dim;
    in.seekg(0,std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim+1) / 4);
    fmt::println("data dimension: {}, data number: {}", dim, num);
    data = new unsigned[num * dim * sizeof(unsigned)];
    
    in.seekg(0,std::ios::beg);
    for(size_t i = 0; i < num; i++){
      in.seekg(4,std::ios::cur);
      in.read((char*)(data+i*dim),dim*4);
    }
    in.close();
}

void read_bin(const size_t n,
              const size_t d,
              const std::string& fname,
              std::vector<std::vector<float>>& data) {
    fmt::println("Read: {}, row: {}, col: {}", fname, n, d);

    data.resize(n);

    std::ifstream in(fname, std::ios::binary);
    if (!in.is_open()){
        fmt::println("Open File {} Fail", fname);
        exit(1);
    }
    int npts, dim;
    in.read((char *)&npts, sizeof(int));
    in.read((char *)&dim, sizeof(int));
    for (size_t i=0; i<n; ++i) {
        data[i].resize(d);
        in.read((char *)(data[i].data()), d * sizeof(float));
    }
    in.close();
    fmt::println("Read Done");
}

int read_bin(const int n,
             const int d,
             const std::string& kFilePath,
             float*& data) {
    fmt::println("Read: {}, row: {}, col: {}", kFilePath, n, d);

    data = new float[n * d];

    fmt::println("Read {}", kFilePath);
    std::ifstream in(kFilePath, std::ios::binary);
    if (!in.is_open()) {
        fmt::println("Open File {} Fail", kFilePath);
        exit(1);
    }
    // in.seekg(0, std::ios::end);
    // std::cout << in.tellg() << std::endl;
    // in.seekg(0, std::ios::beg);
    int npts, dim;
    in.read((char *)&npts, sizeof(int));
    in.read((char *)&dim, sizeof(int));
    for (size_t i=0; i<n; ++i) {
        in.read((char *) &data[i*d], d * sizeof(float));
    }
    in.close();
    fmt::println("Read Done");

    return 0;
}

void read_bin(const unsigned num,
              const unsigned dim,
              const std::string &kFilePath,
              float **data) {
    fmt::println("Read {}", kFilePath);
    std::ifstream in(kFilePath, std::ios::binary);
    if (!in.is_open()) {
        fmt::println("Open File {} Fail", kFilePath);
        exit(1);
    }
    int npts, ndim;
    in.read((char *)&npts, sizeof(int));
    in.read((char *)&ndim, sizeof(int));
    data = new float*[num];
    for (size_t i=0; i<num; ++i) {
        data[i] = new float[dim];
        in.read((char *) data[i], dim * sizeof(float));

    }
    in.close();
    fmt::println("Read Done");
}

void read_p2h_data(const unsigned num,
                   const unsigned dim,
                   const std::string &kFilePath,
                   float*& data) {
    fmt::println("Load data from {}", kFilePath);
    fmt::println("DataNum: {}, DataDim: {}", num, dim);

    std::ifstream in(kFilePath, std::ios::binary);
    if (!in.is_open()) {
        fmt::println("Open File {} Fail", kFilePath);
        exit(1);
    }

    data = new float[num * dim];

    for (size_t i=0; i<num; ++i) {
        in.read((char *)(&data[i*dim]), dim * sizeof(float));
        float tmp;
        in.read((char *)(&tmp), sizeof(float));
        if (tmp != 1.0f) {
            fmt::println("Error: {}", tmp);
            exit(1);
        }
    }
    in.close();
}

void read_p2h_data(const unsigned num,
                   const unsigned dim,
                   const std::string &kFilePath,
                   float** data) {
    std::ifstream in(kFilePath, std::ios::binary);
    if (!in.is_open()) {
        fmt::println("Open File {} Fail", kFilePath);
        exit(1);
    }

    data = new float*[num];
    for (size_t i=0; i<num; ++i) {
        data[i] = new float[dim];
        in.read((char *) data[i], dim * sizeof(float));
        float tmp;
        in.read((char *)&tmp, sizeof(float));
    }
    in.close();
}

int read_bin_data(                  // read bin data from disk
    const int   n,                  // number of data points
    const int   d,                  // data dimension
    const char *fname,              // address of data set
    float *min_coord,               // min coordinates (return)
    float *max_coord,               // max coordinates (return)
    float *data                     // data (return)
) {
    // gettimeofday(&g_start_time, NULL);
    // utils::Timer<std::chrono::microseconds> timer;
    // timer.reset();

    // read bin data
    FILE *fp = fopen(fname, "rb");
    if (!fp) { printf("Could not open %s\n", fname); return 1; }

    int i = 0;
    while (!feof(fp) && i < n) {
        uint64_t shift = (uint64_t) i*(d+1);
        std::ignore = fread(&data[shift], sizeof(float), d, fp);
        data[shift+d] = 1.0f;
        ++i;
    }
    assert(i == n);
    fclose(fp);

    // shift data by the position of center
    for (int i = 0; i < n; ++i) {
        const float *point = (const float*) &data[(uint64_t)i*(d+1)];
        for (int j = 0; j < d; ++j) {
            if (i == 0 || point[j] < min_coord[j]) min_coord[j] = point[j];
            if (i == 0 || point[j] > max_coord[j]) max_coord[j] = point[j];
        }
    }

    // calc the data center
    float *center = new float[d];
    for (int i = 0; i < d; ++i) center[i] = (min_coord[i]+max_coord[i])/2.0f;

    // shift the data by the center & find the max l2-norm to the center
    float max_norm = -1.0f, norm = -1.0f, val = -1.0f;
    for (int i = 0; i < n; ++i) {
        float *point = &data[(uint64_t)i*(d+1)];
        // shift the data by the center
        norm = 0.0f;
        for (int j = 0; j < d; ++j) {
            val = point[j] - center[j];
            point[j] = val; norm += val * val;
        }
        norm = sqrt(norm);
        // find the max l2-norm to the center
        if (max_norm < norm) max_norm = norm;
    }

    // max normalization: rescale the data by the max l2-norm
    for (int i = 0; i < n; ++i) {
        float *point = &data[(uint64_t)i*(d+1)];
        for (int j = 0; j < d; ++j) {
            val = point[j] / max_norm;
            point[j] = val;
            if (i == 0 || val < min_coord[j]) min_coord[j] = val;
            if (i == 0 || val > max_coord[j]) max_coord[j] = val;
        }
    }
    for (int j = 0; j < d; ++j) {
        printf("min[%d]=%f, max[%d]=%f\n", j, min_coord[j], j, max_coord[j]);
    }
    delete[] center;

    // gettimeofday(&g_end_time, NULL);
    // float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
    //     (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    // float running_time = timer.getElapsedTime().count() / 1.0e6;

    // printf("Read Data: %f Seconds\n", running_time);

    return 0;
}


int read_bin_data_and_normalize(    // read bin data & normalize
    int   n,                        // number of data points
    int   d,                        // data dimension
    const char *fname,              // address of data set
    float *data                     // data (return)
) {
    // gettimeofday(&g_start_time, NULL);
    // utils::Timer<std::chrono::microseconds> timer;
    // timer.reset();

    // read bin data
    FILE *fp = fopen(fname, "rb");
    if (!fp) { printf("Could not open %s\n", fname); return 1; }

    int i = 0;
    while (!feof(fp) && i < n) {
        uint64_t shift = (uint64_t) i*(d+1);
        std::ignore = fread(&data[shift], sizeof(float), d, fp);
        data[shift+d] = 1.0f;
        ++i;
    }
    assert(i == n);
    fclose(fp);

    // calc the min & max coordinates for d dimensions
    float *min_coord = new float[d];
    float *max_coord = new float[d];

    for (int i = 0; i < n; ++i) {
        const float *point = (const float*) &data[(uint64_t)i*(d+1)];
        for (int j = 0; j < d; ++j) {
            if (i == 0 || point[j] < min_coord[j]) min_coord[j] = point[j];
            if (i == 0 || point[j] > max_coord[j]) max_coord[j] = point[j];
        }
    }

    // calc the data center
    float *center = new float[d];
    for (int i = 0; i < d; ++i) center[i] = (min_coord[i]+max_coord[i])/2.0f;

    float norm = -1.0f, val = -1.0f;
    for (int i = 0; i < n; ++i) {
        float *point = &data[(uint64_t)i*(d+1)];
        // shift data by the center & calc the l2-norm to the center
        norm = 0.0f;
        for (int j = 0; j < d; ++j) {
            val = point[j] - center[j];
            point[j] = val; norm += val * val;
        }
        norm = sqrt(norm);
        // normalization
        for (int j = 0; j < d; ++j) point[j] /= norm;
    }
    // release space
    delete[] max_coord;
    delete[] min_coord;
    delete[] center;

    // gettimeofday(&g_end_time, NULL);
    // float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
    //     (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;

    // float running_time = timer.getElapsedTime().count() / 1.0e6;

    // printf("Read & Normalize Data: %f Seconds\n", running_time);

    return 0;
}

void load_knng(
    const std::string kGraphPath,
    std::vector<std::vector<unsigned>> &graph
) {
    std::ifstream in(kGraphPath, std::ios::binary);
    unsigned k;
    in.read((char*)&k,4);
    std::cout << "k: " << k << std::endl;
    in.seekg(0,std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    size_t num = fsize / ((size_t)k + 1) / 4;
    in.seekg(0,std::ios::beg);

    graph.resize(num);
    for(size_t i = 0; i < num; i++){
      in.seekg(4,std::ios::cur);
      graph[i].resize(k);
      graph[i].reserve(k);
      in.read((char*)graph[i].data(), k * sizeof(unsigned));
    }
    in.close();
}

void load_merge_knng(
    const std::string kGraphPath,
    std::vector<std::vector<unsigned>> &graph
) {
    std::ifstream input(kGraphPath, std::ios::binary);
    if(!input.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
    fmt::println("Load merge knng from {}", kGraphPath);

    unsigned data_num;
    // input.read((char*)&data_num, sizeof(data_num));
    utils::read_binary(input, data_num);
    fmt::println("data_num: {}", data_num);
    graph.resize(data_num);
    for (size_t i=0; i<data_num; ++i) {
        unsigned size;
        input.read((char*)&size, sizeof(size));
        graph[i].resize(size);
        graph[i].reserve(size);
        input.read((char*)graph[i].data(), size * sizeof(unsigned));
    }
    input.close();
    fmt::println("Load Over");
}


void load_pure_hnsw(
    const std::string& kGraphPath,
    std::vector<std::vector<unsigned>>& graph
) {
    std::ifstream input(kGraphPath, std::ios::binary);
    fmt::println("Load pure hnsw from {}", kGraphPath);

    size_t data_num;
    input.read((char *)&data_num, sizeof(data_num));

    unsigned ep;
    input.read((char *)&ep, sizeof(ep));

    graph.resize(data_num);
    graph.reserve(data_num);
    for (size_t i=0; i<data_num; ++i) {
        unsigned size;
        input.read((char *)&size, sizeof(size));
        graph[i].resize(size);
        graph[i].reserve(size);
        input.read((char*)graph[i].data(), size * sizeof(unsigned));
    }

    input.close();
}


void save_graph(
    const std::string &kGraphPath, 
    const std::vector<std::vector<unsigned int>> &graph) {

    std::ofstream out(kGraphPath, std::ios::binary | std::ios::out);
    if(!out.is_open()){
        fmt::println("open file error");
        exit(-1);
    }

    const size_t data_num = graph.size();
    out.write((char*)&data_num, sizeof(data_num));

    for (size_t i=0; i<data_num; ++i) {
        const unsigned size = graph[i].size();
        out.write((char*)&size, sizeof(size));
        out.write((char*)graph[i].data(), size * sizeof(unsigned));
    }

    out.close();
    fmt::println("Save Graph in: {}", kGraphPath);
}

void load_graph(
    const std::string &kGraphPath,
    std::vector<std::vector<unsigned int>> &graph) {

    std::ifstream in(kGraphPath, std::ios::binary | std::ios::out);
    if(!in.is_open()){
        fmt::println("open file error");
        exit(-1);
    }
    fmt::println("Load Graph from {}", kGraphPath);
    
    size_t data_num;
    in.read((char *)&data_num, sizeof(data_num));

    graph.resize(data_num);
    graph.reserve(data_num);
    for (size_t i=0; i<data_num; ++i) {
        unsigned size;
        in.read((char *)&size, sizeof(size));
        graph[i].resize(size);
        graph[i].reserve(size);
        in.read((char*)graph[i].data(), size * sizeof(unsigned));
    }
    in.close();
}

void get_union(
    const std::vector<unsigned> &vec1,
    const std::vector<unsigned> &vec2,
    std::vector<unsigned> &res) 
    {
    std::set<unsigned> set1(vec1.begin(), vec1.end());
    std::set<unsigned> set2(vec2.begin(), vec2.end());

    std::set<unsigned> union_set;
    union_set.insert(set1.begin(), set1.end());
    union_set.insert(set2.begin(), set2.end());

    res.assign(union_set.begin(), union_set.end());
}
}