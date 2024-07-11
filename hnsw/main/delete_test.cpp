#include "../hnswlib/include/hnswlib.h"
#include "utils/io_utils.h"
#include "utils/timer.h"
#include "utils/dist_func.h"
#include <omp.h>
std::size_t kDataDim;
std::size_t kDataNum;
std::string kFrontPath;
std::string kDataset;
unsigned kHnswM;
unsigned kHnswConsEf;
int kQueryNum, kTopK;
unsigned kHnswEf;
float *data_ptr, *queries;
int start;
hnswlib::HierarchicalNSW<float> *hnsw;
hnswlib::BruteforceSearch<float> *brute;
std::vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>> hnsw_res, gt_res;
void init() {
    utils::Timer<std::chrono::microseconds> timer;
    timer.reset();
    auto space_hnsw = new hnswlib::L2Space(kDataDim);
    auto space_brute = new hnswlib::L2Space(kDataDim);
    hnsw = new hnswlib::HierarchicalNSW<float>(space_hnsw, kDataNum, kHnswM, kHnswConsEf, 100, false);
    brute = new hnswlib::BruteforceSearch<float>(space_brute, kDataNum);
    hnsw->setEf(kHnswEf);
    hnsw->addPoint(&data_ptr[0], 0);
    // omp_set_num_threads(32);
#pragma omp parallel for schedule(dynamic, 96)
    for (std::size_t i=1; i<start; ++i) {
        hnsw->addPoint(&data_ptr[i*kDataDim], i);
    }
    timer.end();
    std::cout << kDataset << " HNSW Build Time: " << (float)timer.getElapsedTime().count() / 1.0e3 << " ms" << std::endl;
    const std::string kIndexPath = kFrontPath + kDataset + ".hnsw";
    std::cout << "index path: " << kIndexPath << std::endl;
    hnsw->saveIndex(kIndexPath);
#pragma omp parallel for schedule(dynamic, 96)
    for (std::size_t i=0; i<start; ++i) {
        brute->addPoint(&data_ptr[i*kDataDim], i);
    }
}
std::vector<double> search_time, insert_time;
void search(int cycle) {
    hnsw_res.resize(kQueryNum);
    utils::Timer<std::chrono::microseconds> timer;
    timer.reset();
    for (int i = 0; i < kQueryNum; i++) {
        hnsw_res[i] = hnsw->searchKnn(&queries[i*kDataDim], kTopK);
    }
    timer.end();
    std::cout << "Cycle " << cycle << " Search Time: " << (float)timer.getElapsedTime().count() / 1.0e3 << " ms" << std::endl;
    if (cycle >= 0)
        search_time.push_back((double)timer.getElapsedTime().count() / 1.0e3);
}
void compute_recall(int cycle) {
    std::cout << "Cycle " << cycle << " recall is computing ..." << std::endl;
    gt_res.resize(kQueryNum);
#pragma omp parallel for schedule(dynamic, 96)
    for (int i = 0; i < kQueryNum; i++) {
        // std::cout << i << std::endl;
        gt_res[i] = brute->searchKnn(&queries[i*kDataDim], kTopK);
    }
    std::cout << "GT is generated." << std::endl;
    int cnt = 0;
// #pragma omp parallel for schedule(dynamic, 96)
    for (int i = 0; i < kQueryNum; i++) {
        int r_size = gt_res[i].size();
        std::vector<bool> flag(r_size, true);
        std::vector<int> ids;
        while (!gt_res[i].empty()) {
            ids.emplace_back(gt_res[i].top().second);
            gt_res[i].pop();
        }
        while (!hnsw_res[i].empty()) {
            const auto& r= hnsw_res[i].top(); hnsw_res[i].pop();
            for (int i = 0; i < r_size; ++i) {
                if (flag[i] && r.second == ids[i]) {
                // #pragma omp atomic
                    ++cnt;
                    flag[i] = false;
                    break;
                }
            }
        }
    }
    std::cout << "Recall: " << (double)cnt / (kTopK * kQueryNum) << std::endl;
}
int main(int argc, char **argv) {
    kDataDim = atoi(argv[1]);
    kDataNum = atoi(argv[2]);
    kFrontPath = argv[3];
    kDataset = argv[4];

    kHnswM = atoi(argv[5]);
    kHnswConsEf = atoi(argv[6]);

    kQueryNum = atoi(argv[7]);
    kTopK = atoi(argv[8]);
    kHnswEf = atoi(argv[9]);

    const int delta = atoi(argv[10]);
    const int total_cyc = atoi(argv[11]);

    data_ptr = new float[kDataNum * kDataDim];
    const std::string kDataPath = kFrontPath + kDataset + ".bin";
    const std::string kQueryPath = kFrontPath + kDataset + ".q";
    utils::read_bin(kDataNum, kDataDim, kDataPath, data_ptr);
    utils::read_bin(kQueryNum, kDataDim, kQueryPath, queries);

    
    std::mt19937 rnd(19260817);
    std::vector<bool> del(kDataNum, false);
    start = kDataNum / 2;
    int pos = start;
    init();
    search(-1);
    for (int cyc = 0; cyc < total_cyc; cyc++) {
        search(cyc);
        compute_recall(cyc);
        for (int i = 0; i < delta; i++) {
            int id = rnd() % pos;
            while (del[id]) id = rnd() % pos;
            del[id] = true;
            hnsw->markDelete(id);
            brute->removePoint(id);
        }
        // puts("!!!");
        utils::Timer<std::chrono::microseconds> timer;
        timer.reset();
        omp_set_num_threads(32);
#pragma omp parallel for schedule(dynamic, 32)
        for (int i = pos; i < pos + delta; i++) {
            hnsw->addPoint(&data_ptr[i*kDataDim], i);
        }
        omp_set_num_threads(96);
        timer.end();
        std::cout << "Insert latency: " << (float)timer.getElapsedTime().count() / 1.0e3 << " ms\n" << std::endl;
        insert_time.push_back((double)timer.getElapsedTime().count() / 1.0e3);
        for (int i = pos; i < pos + delta; i++) {
            brute->addPoint(&data_ptr[i*kDataDim], i);
        }
        pos += delta;
    }
    std::cout << "[";
    for (int i = 0; i < (int)search_time.size(); i++) {
        std::cout << search_time[i];
        if (i != (int)search_time.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "[";
    for (int i = 0; i < (int)insert_time.size(); i++) {
        std::cout << insert_time[i];
        if (i != (int)insert_time.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    return 0;
}