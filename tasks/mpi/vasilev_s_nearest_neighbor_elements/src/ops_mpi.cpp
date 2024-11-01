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
  int local_size = base_size + (world.rank() < remainder ? 1 : 0);
  int start_idx = world.rank() * base_size + std::min(world.rank(), remainder);
  int end_idx = start_idx + local_size;

  if (start_idx > 0) {
    start_idx -= 1;
  }
  if (end_idx < total_size) {
    end_idx += 1;
  }
  int adjusted_local_size = end_idx - start_idx;
  local_input_.resize(adjusted_local_size);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); ++proc) {
      int proc_base_size = total_size / world.size();
      int proc_remainder = total_size % world.size();
      int proc_local_size = proc_base_size + (proc < proc_remainder ? 1 : 0);
      int proc_start_idx = proc * proc_base_size + std::min(proc, proc_remainder);
      int proc_end_idx = proc_start_idx + proc_local_size;

      if (proc_start_idx > 0) {
        proc_start_idx -= 1;
      }
      if (proc_end_idx < total_size) {
        proc_end_idx += 1;
      }
      int proc_adjusted_size = proc_end_idx - proc_start_idx;

      world.send(proc, 0, &input_[proc_start_idx], proc_adjusted_size);
    }
    std::copy(input_.begin() + start_idx, input_.begin() + end_idx, local_input_.begin());
  } else {
    world.recv(0, 0, (local_input_).data(), adjusted_local_size);
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
