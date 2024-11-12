#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/vasilev_s_striped_horizontal_scheme/include/ops_mpi.hpp"

namespace vasilev_s_striped_horizontal_scheme_mpi {

std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(0, 1000);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}

std::vector<int> getRandomMatrix(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(0, 1000);
  std::vector<int> matrix(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = dist(gen);
  }
  return matrix;
}
}  // namespace vasilev_s_striped_horizontal_scheme_mpi

TEST(vasilev_s_striped_horizontal_scheme_mpi, calculate_distribution_num_proc_greater_than_rows) {
  int rows = 3;
  int cols = 4;
  int num_proc = 5;
  std::vector<int> sizes;
  std::vector<int> displs;

  vasilev_s_striped_horizontal_scheme_mpi::calculate_distribution(rows, cols, num_proc, sizes, displs);

  ASSERT_EQ(static_cast<int>(sizes.size()), num_proc);
  ASSERT_EQ(static_cast<int>(displs.size()), num_proc);

  for (int i = 0; i < num_proc; ++i) {
    if (i < rows) {
      EXPECT_EQ(sizes[i], cols);
      EXPECT_EQ(displs[i], i * cols);
    } else {
      EXPECT_EQ(sizes[i], 0);
      EXPECT_EQ(displs[i], -1);
    }
  }
}

TEST(vasilev_s_striped_horizontal_scheme_mpi, calculate_distribution_num_proc_less_than_rows_with_remainder) {
  int rows = 10;
  int cols = 3;
  int num_proc = 4;
  std::vector<int> sizes;
  std::vector<int> displs;

  vasilev_s_striped_horizontal_scheme_mpi::calculate_distribution(rows, cols, num_proc, sizes, displs);

  ASSERT_EQ(static_cast<int>(sizes.size()), num_proc);
  ASSERT_EQ(static_cast<int>(displs.size()), num_proc);

  int expected_sizes[] = {9, 9, 6, 6};
  int expected_displs[] = {0, 9, 18, 24};

  for (int i = 0; i < num_proc; ++i) {
    EXPECT_EQ(sizes[i], expected_sizes[i]);
    EXPECT_EQ(displs[i], expected_displs[i]);
  }
}

TEST(vasilev_s_striped_horizontal_scheme_mpi, calculate_distribution_num_proc_less_than_rows_no_remainder) {
  int rows = 8;
  int cols = 2;
  int num_proc = 4;
  std::vector<int> sizes;
  std::vector<int> displs;

  vasilev_s_striped_horizontal_scheme_mpi::calculate_distribution(rows, cols, num_proc, sizes, displs);

  ASSERT_EQ(static_cast<int>(sizes.size()), num_proc);
  ASSERT_EQ(static_cast<int>(displs.size()), num_proc);

  int expected_sizes[] = {4, 4, 4, 4};
  int expected_displs[] = {0, 4, 8, 12};

  for (int i = 0; i < num_proc; ++i) {
    EXPECT_EQ(sizes[i], expected_sizes[i]);
    EXPECT_EQ(displs[i], expected_displs[i]);
  }
}

TEST(vasilev_s_striped_horizontal_scheme_mpi, small_matrix_vector_test) {
  boost::mpi::communicator world;

  std::vector<int> global_matrix;
  std::vector<int> global_vector;
  std::vector<int> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_rows = 4;

  if (world.rank() == 0) {
    global_matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    global_vector = {1, 0, -1};

    global_result.resize(num_rows, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel =
      std::make_shared<vasilev_s_striped_horizontal_scheme_mpi::StripedHorizontalSchemeParallelMPI>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<int> seq_result(global_result.size(), 0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataSeq->inputs_count.emplace_back(global_vector.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    auto taskSequential =
        std::make_shared<vasilev_s_striped_horizontal_scheme_mpi::StripedHorizontalSchemeSequentialMPI>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    ASSERT_EQ(global_result.size(), seq_result.size());
    for (size_t i = 0; i < global_result.size(); ++i) {
      EXPECT_EQ(global_result[i], seq_result[i]);
    }
  }
}
