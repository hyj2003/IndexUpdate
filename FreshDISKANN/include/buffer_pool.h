#include <shared_mutex>

#include "tsl/robin_map.h"
#include "aligned_file_reader.h"
#include "concurrent_queue.h"
#include "neighbor.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq_table.h"
#include "pq_flash_index.h"
#include "utils.h"

class BufferPool {
 public:
  void read(IOContext &ctx) {
    std::vector<AlignedRead> read_reqs;
  }
 private:
  std::shared_ptr<AlignedFileReader> reader = nullptr;
} ;