#pragma once
#ifndef _WINDOWS

#include "aligned_file_reader.h"

class LinuxAlignedFileReader : public AlignedFileReader {
 private:
  uint64_t     file_sz;
  FileHandle   file_desc;
  io_context_t bad_ctx = (io_context_t) -1;

 public:
  LinuxAlignedFileReader();
  ~LinuxAlignedFileReader();

  IOContext &get_ctx();

  // register thread-id for a context
  void register_thread();

  // de-register thread-id for a context
  void deregister_thread();

  void deregister_all_threads();

  // Open & close ops
  // Blocking calls
  void open(const std::string &fname, bool enable_writes, bool enable_create);
  void close();

  // process batch of aligned requests in parallel
  // NOTE :: blocking call
  void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
            bool async = false);

  void sequential_write(AlignedRead &write_req, IOContext &ctx);

  int submit_reqs(std::vector<AlignedRead> &read_reqs, IOContext &ctx);
  void get_events(IOContext &ctx, int n_ops);
};

#endif
