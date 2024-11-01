#include "mpi/vasilev_s_nearest_neighbor_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <vector>

std::vector<int> vasilev_s_nearest_neighbor_elements_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(0, 1000);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}

bool vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsSequentialMPI::pre_processing() {
  internal_order_test();
  input_.resize(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  min_diff_ = std::numeric_limits<int>::max();
  index1_ = -1;
  index2_ = -1;
  return true;
}

bool vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsSequentialMPI::validation() {
  internal_order_test();
  if (taskData->inputs_count.empty() || taskData->inputs_count[0] < 2) {
    return false;
  }
  return taskData->outputs_count[0] == 3;
}

bool vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsSequentialMPI::run() {
  internal_order_test();
  for (size_t i = 0; i < input_.size() - 1; ++i) {
    int diff = std::abs(input_[i + 1] - input_[i]);
    if (diff < min_diff_) {
      min_diff_ = diff;
      index1_ = static_cast<int>(i);
      index2_ = static_cast<int>(i + 1);
    }
  }
  return true;
}

bool vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsSequentialMPI::post_processing() {
  internal_order_test();
  int* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
  output_ptr[0] = min_diff_;
  output_ptr[1] = index1_;
  output_ptr[2] = index2_;
  return true;
}

bool vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsParallelMPI::pre_processing() {
  internal_order_test();
  int total_size = 0;

  if (world.rank() == 0) {
    input_.resize(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    total_size = static_cast<int>(input_.size());
  }

  boost::mpi::broadcast(world, total_size, 0);

  int base_size = total_size / world.size();
  int remainder = total_size % world.size();
  std::vector<int> send_counts(world.size(), base_size);
  std::vector<int> displacements(world.size(), 0);

  for (int i = 0; i < remainder; ++i) {
    send_counts[i] += 1;
  }
  std::partial_sum(send_counts.begin(), send_counts.end() - 1, displacements.begin() + 1);

  int local_size = send_counts[world.rank()];
  int start_idx = displacements[world.rank()];
  int end_idx = start_idx + local_size;

  if (start_idx > 0) {
    start_idx -= 1;
    local_size += 1;
  }
  if (end_idx < total_size) {
    local_size += 1;
  }
  local_input_.resize(local_size);

  if (world.rank() == 0) {
    std::vector<int> extended_input;
    for (int i = 0; i < world.size(); ++i) {
      int ext_start = displacements[i];
      int ext_end = displacements[i] + send_counts[i];
      if (ext_start > 0) ext_start -= 1;
      if (ext_end < total_size) ext_end += 1;
      extended_input.insert(extended_input.end(), input_.begin() + ext_start, input_.begin() + ext_end);
    }
    boost::mpi::scatterv(world, extended_input, send_counts, displacements, local_input_.data(), local_size, 0);
  } else {
    boost::mpi::scatterv(world, input_, send_counts, displacements, local_input_.data(), local_size, 0);
  }

  local_offset_ = start_idx;

  min_diff_ = std::numeric_limits<int>::max();
  index1_ = -1;
  index2_ = -1;
  return true;
}

bool vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsParallelMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs_count.empty() || taskData->inputs_count[0] < 2) {
      return false;
    }
    return taskData->outputs_count[0] == 3;
  }
  return true;
}

bool vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsParallelMPI::run() {
  internal_order_test();
  for (size_t i = 0; i < local_input_.size() - 1; ++i) {
    int diff = std::abs(local_input_[i + 1] - local_input_[i]);
    if (diff < min_diff_) {
      min_diff_ = diff;
      index1_ = static_cast<int>(i) + local_offset_;
      index2_ = static_cast<int>(i + 1) + local_offset_;
    }
  }

  LocalResult local_result = {min_diff_, index1_, index2_};

  std::vector<LocalResult> all_results;
  if (world.rank() == 0) {
    all_results.resize(world.size());
  }

  boost::mpi::gather(world, local_result, all_results, 0);

  if (world.rank() == 0) {
    min_diff_ = all_results[0].min_diff;
    index1_ = all_results[0].index1;
    index2_ = all_results[0].index2;
    for (size_t i = 1; i < all_results.size(); ++i) {
      if (all_results[i].min_diff < min_diff_) {
        min_diff_ = all_results[i].min_diff;
        index1_ = all_results[i].index1;
        index2_ = all_results[i].index2;
      }
    }
  }

  return true;
}

bool vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsParallelMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
    output_ptr[0] = min_diff_;
    output_ptr[1] = index1_;
    output_ptr[2] = index2_;
  }
  return true;
}
