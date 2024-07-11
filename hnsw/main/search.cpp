#include "../hnswlib/include/hnswlib.h"
#include "utils/io_utils.h"
#include "utils/timer.h"
#include "utils/dist_func.h"
#include "utils/recall_utils.h"

int main(int argc, char **argv) {
    const std::string kFrontPath = argv[1];
    const std::string kDataset = argv[2];

    const std::size_t kDataDim = atoi(argv[3]);
    const std::size_t kDataNum = atoi(argv[4]);
    const std::size_t kQueryNum = atoi(argv[5]);


    const std::string kIndexPath = argv[6];
    const unsigned kHnswEf = atoi(argv[7]);
    const unsigned kTopK = atoi(argv[8]);
    const std::size_t kGtSize = atoi(argv[9]);

    const std::size_t kQueryDim = kDataDim;

    float *data_ptr = new float[kDataNum * kDataDim];
    float *query_ptr = new float[kQueryNum * kQueryDim];
    const std::string kDataPath = kFrontPath + kDataset + ".bin";
    const std::string kQueryPath = kFrontPath + kDataset + ".q";
    utils::read_bin(kDataNum, kDataDim, kDataPath, data_ptr);
    utils::read_bin(kQueryNum, kQueryDim, kQueryPath, query_ptr);
    auto space = new hnswlib::L2Space(kDataDim);
    hnswlib::HierarchicalNSW<float> *hnsw = new hnswlib::HierarchicalNSW<float>(space, kIndexPath);
    utils::Timer<std::chrono::microseconds> timer;
    std::vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>> hnsw_res(kQueryNum);
    timer.reset();
    // for (int i = 0; i < kDataDim; i++)
    //     printf("%0.5f ", query_ptr[i]);
    // putchar('\n');
// #pragma omp parallel for
    hnsw->setEf(kHnswEf);
    for (std::size_t i=0; i<kQueryNum; ++i) {
        // auto q = hnsw->searchKnnCntSQ8(&query_ptr[i*kQueryDim], 2 * kTopK, cal_cnt);
        // auto &p = hnsw_res[i];
        // while (!q.empty()) {
        //     int id = q.top().second; q.pop();
        //     // printf("%d\n", id);
        //     p.emplace(fabs(utils::InnerProductFloatSSE(query_ptr + i * kQueryDim, data_ptr + id * kQueryDim, &kDataDim)), id);
        //     if ((int)p.size() > kTopK) p.pop();
        // }
        hnsw_res[i] = hnsw->searchKnn(&query_ptr[i*kQueryDim], kTopK);
    }
    timer.end();
    std::cout << kDataset << " HNSW Search Time: " << (float)timer.getElapsedTime().count() / 1e3 / kQueryNum 
              << " ms" << std::endl;

    unsigned *gt_id = nullptr;
    const std::string kGtPath = kFrontPath + kDataset + ".gt";
    utils::read_data(kGtPath, kQueryNum, kGtSize, gt_id);
    float recall = utils::get_recall_by_id(kQueryNum, kGtSize, gt_id, hnsw_res);

    std::cout << kDataset << " Recall: " << recall << std::endl;

    std::cout << kDataset << " Avg Hop: " << (float)hnsw->metric_hops / kQueryNum << std::endl;
    std::cout << kDataset << " Avg HopCal: " << (float) hnsw->metric_distance_computations / kQueryNum << std::endl;
    // float avg_hop = 0;
    // for (int t = 0; t < 10; t++) {
    //     avg_hop += (float)hnsw[t]->metric_hops / kQueryNum;
    // }
    // std::cout << kDataset << " Avg Hop: " << avg_hop << std::endl;
    // float avg_hopcal = 0;
    // for (int t = 0; t < 10; t++) {
    //     avg_hop += (float) hnsw[t]->metric_distance_computations / kQueryNum;
    // }
    // std::cout << kDataset << " Avg HopCal: " << avg_hop << std::endl;
    // std::cout << kDataset << " Avg RealCal: " << (float) cal_cnt / kQueryNum << std::endl;
    delete[] query_ptr;
    delete[] data_ptr;
    return 0;
}