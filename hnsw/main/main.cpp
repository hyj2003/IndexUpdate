#include "../hnswlib/include/hnswlib.h"
#include "utils/io_utils.h"
#include "utils/timer.h"
#include "utils/dist_func.h"
#include <omp.h>
int main(int argc, char **argv) {
    const std::size_t kDataDim = atoi(argv[1]);
    const std::size_t kDataNum = atoi(argv[2]);
    const std::string kFrontPath = argv[3];
    const std::string kDataset = argv[4];

    const unsigned kHnswM = atoi(argv[5]);
    const unsigned kHnswEf = atoi(argv[6]);

    const std::size_t kQueryDim = kDataDim;

    float *data_ptr = new float[kDataNum * kDataDim];
    const std::string kDataPath = kFrontPath + kDataset + ".bin";
    utils::read_bin(kDataNum, kDataDim, kDataPath, data_ptr);
    utils::Timer<std::chrono::microseconds> timer;
    timer.reset();
    auto space = new hnswlib::L2Space(kDataDim);

    hnswlib::HierarchicalNSW<float> *hnsw = new hnswlib::HierarchicalNSW<float>(space, kDataNum, kHnswM, kHnswEf);
    hnsw->addPoint(&data_ptr[0], 0);
    omp_set_num_threads(32);
#pragma omp parallel for schedule(dynamic, 32)
    for (std::size_t i=1; i<kDataNum; ++i) {
        hnsw->addPoint(&data_ptr[i*kDataDim], i);
    }
    timer.end();
    std::cout << kDataset << " HNSW Build Time: " << (float)timer.getElapsedTime().count() / 1.0e3 << " ms" << std::endl;

    const std::string kIndexPath = kFrontPath + kDataset + ".hnsw";
    std::cout << "index path: " << kIndexPath << std::endl;
    hnsw->saveIndex(kIndexPath);
//     for (int i = 0; i < 10; i++) {
//         const std::string kIndexPath = kFrontPath + kDataset + "-" + std::to_string(i) + ".hnsw";
//         std::cout << "index path: " << kIndexPath << std::endl;
//         hnsw[i]->saveIndex(kIndexPath);
//     }
    delete[] data_ptr;
    return 0;
}