//#include <distances.h>
//#include <indexing.h>
//
#include "omp.h"

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"

template<typename T>
void create_data(int argc, char **argv) {
    std::string data_file = std::string(argv[2]);
    std::string R_disk = std::string(argv[4]);
    std::string L_disk = std::string(argv[5]);
    std::string B_disk = std::string(argv[6]);
    std::string M_disk = std::string(argv[7]);
    std::string Threads = std::string(argv[8]);
    int kvecs = std::atoi(argv[9]);
    std::string R_mem = std::string(argv[10]);
    std::string L_mem = std::string(argv[11]);
    int mem_num = std::atoi(argv[12]);
    std::string save_folder = std::string(argv[13]);

    std::string disk_save_path = save_folder + "disk_data";
    std::string mem_save_path = save_folder + "mem_data_";

    std::string params_disk = R_disk + " " + L_disk + " " + B_disk + " " + M_disk + " " + Threads;

    size_t npts, dim;
    std::cout << data_file << std::endl;
    diskann::get_bin_metadata(data_file, npts, dim);
    std::cout << "new T " << npts << " " << dim << " " << (_u64)npts * dim << std::endl;
    T *data = new T[(_u64)npts * dim];
    std::cout << "end new" << std::endl;
    int const_1 = 1;
    diskann::load_bin<T>(data_file, data, npts, dim);
    {
        std::ofstream data_writer(disk_save_path.c_str(), std::ios::binary);
        data_writer.write((char *) &kvecs, sizeof(uint32_t));
        data_writer.write((char *) &dim, sizeof(uint32_t));
        data_writer.write((char *) data, kvecs * dim * sizeof(T));
        data_writer.close();
    }
    int num_per_mem = (npts - kvecs + mem_num - 1) / mem_num;
    for (int i = kvecs, o = 0; i < (int)npts; i += num_per_mem, o++) {
        mem_save_path += '0' + o;
        // std::string tags_save_path = mem_save_path + ".tags";
        std::ofstream data_writer((mem_save_path + ".index.data").c_str(), std::ios::binary);
        // std::ofstream tags_writer(tags_save_path.c_str(), std::ios::binary);
        mem_save_path.erase(mem_save_path.end() - 1);
        T *d = data + i;
        int n = std::min((int)npts - i, num_per_mem);
        data_writer.write((char *) &n, sizeof(uint32_t));
        data_writer.write((char *) &dim, sizeof(uint32_t));
        data_writer.write((char *) d, n * dim * sizeof(T));
        data_writer.close();

        // uint32_t *tags = new uint32_t[n];
        // for (int j = 0; j < n; j++) {
        //     tags[j] = i + j;
        // }
        // tags_writer.write((char *) &n, sizeof(uint32_t));
        // tags_writer.write((char *) &const_1, sizeof(uint32_t));
        // tags_writer.write((char *) tags, n * sizeof(uint32_t));
        // tags_writer.close();
    }

    std::string disk_index_path = save_folder + "disk_index";
    std::string mem_index_path = save_folder + "mem_index_";
    {
        std::cout << "Starting building disk index ... ";
        diskann::build_disk_index_with_tags<T, uint32_t>(
        disk_save_path.c_str(), disk_index_path.c_str(), params_disk.c_str(), diskann::Metric::L2, kvecs);
        std::cout << "done." << std::endl;
    }
    // mem_save_path += '0';
    // std::cout << mem_save_path << std::endl;
    for (int i = kvecs, o = 0; i < (int)npts; i += num_per_mem, o++) {
        // std::cout << "!!" << std::endl;
        // mem_save_path.append(1, '0' + o);
        mem_save_path += '0' + o;
        // std::cout << "!!" << std::endl;
        std::cout << "Starting building index in " << mem_save_path << " ... ";
        // std::cout << "??" << std::endl;
        // std::string tags_save_path = mem_save_path + ".tags";
        diskann::Parameters paras;
        paras.Set<unsigned>("R", std::atoi(argv[10]));
        paras.Set<unsigned>("L", std::atoi(argv[11]));
        paras.Set<unsigned>(
            "C", 750);  // maximum candidate set size during pruning procedure
        paras.Set<float>("alpha", 1.2);
        paras.Set<bool>("saturate_graph", 0);
        paras.Set<unsigned>("num_threads", std::atoi(argv[8]));
        _u64 data_num, data_dim;
        diskann::get_bin_metadata(mem_save_path + ".index.data", data_num, data_dim);
        diskann::Index<T, uint32_t> index(diskann::L2, data_dim, data_num, 0, true);
        std::vector<uint32_t> tags;
        int n = std::min((int)npts - i, num_per_mem);
        for (int j = 0; j < n; j++) {
            tags.push_back(i + j);
        }
        auto              s = std::chrono::high_resolution_clock::now();
        index.build((mem_save_path + ".index.data").c_str(), data_num, paras, tags);
        index.check_graph_quality(20000, paras);
        std::chrono::duration<double> diff =
            std::chrono::high_resolution_clock::now() - s;

        std::cout << "Indexing time: " << diff.count() << "\n";
        index.save((mem_save_path + ".index").c_str(), true, true);
        mem_save_path.erase(mem_save_path.end() - 1);
    }
    {
        std::cout << "Starting generating delete_list ...";
        std::string delete_list = save_folder + "delete_list";
        int del_num = std::atoi(argv[14]);
        std::vector<uint32_t> del;
        for (int i = 0; i < (int)npts; i++) del.push_back(i);
        std::mt19937 rnd(19260817);
        std::shuffle(del.begin(), del.end(), rnd);
        std::ofstream del_writer(delete_list.c_str(), std::ios::binary);
        del_writer.write((char *) &del_num, sizeof(uint32_t));
        del_writer.write((char *) &const_1, sizeof(uint32_t));
        del_writer.write((char *) del.data(), del_num * sizeof(uint32_t));
        del_writer.close();
        std::cout << " done." << std::endl;
    }
}
int main(int argc, char** argv) {
  if (argc != 15) {
    std::cout << "Usage: " << argv[0]
              << "  [data_type<float/int8/uint8>]  [data_file.bin]  "
                 "[index_prefix_path]  "
                 "[R_disk]  [L_disk]  [B_disk]  [M_disk]  [T]  [kvecs]  [R_mem]  [L_mem]  [mem_num]  [save_folder]  [del_num]. "
                 "See README for more information on "
                 "parameters."
              << std::endl;
  } else {
    if (std::string(argv[1]) == std::string("float"))
      create_data<float>(argc, argv);
    else if (std::string(argv[1]) == std::string("int8"))
      create_data<int8_t>(argc, argv);
    else if (std::string(argv[1]) == std::string("uint8"))
      create_data<uint8_t>(argc, argv);
    else
      std::cout << "Error. wrong file type" << std::endl;
  }
  return 0;
}
