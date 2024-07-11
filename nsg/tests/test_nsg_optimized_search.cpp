//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <chrono>
#include <string>

void load_data(char* filename, float*& data, unsigned& num,
               unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  // std::cout<<"data dimension: "<<dim<<std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[(size_t)num * (size_t)dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}

void save_result(const char* filename,
                 std::vector<std::vector<unsigned> >& results) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++) {
    unsigned GK = (unsigned)results[i].size();
    out.write((char*)&GK, sizeof(unsigned));
    out.write((char*)results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}
template<typename T>
void read_data(const std::string kFilePath, T* &data) {
    std::cout << "Read Data from: " << kFilePath << std::endl;
    std::ifstream fin(kFilePath, std::ios::binary);
    if (!fin) {
        std::cerr << "Open File " << kFilePath << " Fail" << std::endl;
        exit(1);
    }
    int npts, dim;
    fin.read((char *)&npts, sizeof(int));
    fin.read((char *)&dim, sizeof(int));
    std::size_t data_size = npts * dim;
    data = new T[data_size];
    fin.read((char *) data, data_size * sizeof(T));
    fin.close();
}
float get_recall_by_id(const std::size_t kQueryNum, const std::size_t kGtSize, const unsigned *kGtIds, 
                      std::vector<std::vector<unsigned> > &res) {
  float correct_num = 0;
  std::size_t r_size = res[0].size();
  for (std::size_t q=0; q<kQueryNum; ++q) {
    std::vector<bool> flag(r_size, true);
    for (; !res[q].empty(); res[q].pop_back()) {
      const auto& r=res[q].back();
      for (std::size_t i=0; i<r_size; ++i) {
        if (flag[i] && r == kGtIds[i]) {
          ++correct_num;
          flag[i] = false;
          break;
        }
      }
    }
    kGtIds += kGtSize;
  }
  return correct_num / (kQueryNum*r_size);
}
int main(int argc, char** argv) {
  if (argc != 7) {
    std::cout << argv[0]
              << " data_file query_file nsg_path search_L search_K result_path"
              << std::endl;
    exit(-1);
  }
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);
  float* query_load = NULL;
  unsigned query_num, query_dim;
  load_data(argv[2], query_load, query_num, query_dim);
  assert(dim == query_dim);

  unsigned L = (unsigned)atoi(argv[4]);
  unsigned K = (unsigned)atoi(argv[5]);

  if (L < K) {
    std::cout << "search_L cannot be smaller than search_K!" << std::endl;
    exit(-1);
  }

  data_load = efanna2e::data_align(data_load, points_num, dim);//one must
  // align the data before build query_load = efanna2e::data_align(query_load,
  // query_num, query_dim);
  efanna2e::IndexNSG index(dim, points_num, efanna2e::FAST_L2, nullptr);
  index.Load(argv[3]);
  index.OptimizeGraph(data_load);

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);
  paras.Set<unsigned>("P_search", L);
  unsigned *ids;
  read_data(argv[6], ids);
  std::vector<std::vector<unsigned> > res(query_num);
  for (unsigned i = 0; i < query_num; i++) res[i].resize(K);

  auto s = std::chrono::high_resolution_clock::now();
  for (unsigned i = 0; i < query_num; i++) {
    index.SearchWithOptGraph(query_load + i * dim, K, paras, res[i].data());
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  std::cout << "search time: " << diff.count() << "\n";

  // save_result(argv[6], res);
  std::cout << "Recall: " << get_recall_by_id(query_num, 100, ids, res) << std::endl;

  return 0;
}
