#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vasilev_s_nearest_neighbor_elements_mpi {

struct LocalResult {
  int min_diff;
  int index1;
  int index2;

  // Функция сериализации для Boost.Serialization
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar& min_diff;
    ar& index1;
    ar& index2;
  }
};

std::vector<int> getRandomVector(int sz);

class FindClosestNeighborsSequentialMPI : public ppc::core::Task {
 public:
  explicit FindClosestNeighborsSequentialMPI(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int min_diff_{};
  int index1_{};
  int index2_{};
};

class FindClosestNeighborsParallelMPI : public ppc::core::Task {
 public:
  explicit FindClosestNeighborsParallelMPI(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> local_input_;
  int min_diff_{};
  int index1_{};
  int index2_{};
  int local_offset_{};
  boost::mpi::communicator world;
};

}  // namespace vasilev_s_nearest_neighbor_elements_mpi
