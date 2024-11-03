#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/vasilev_s_nearest_neighbor_elements/include/ops_mpi.hpp"

TEST(vasilev_s_nearest_neighbor_elements_mpi, test_small_vector) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_result(3, 0);  // min_diff, index1, index2

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = {5, 3, 8, 7, 2};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsParallelMPI task_parallel(taskDataPar);
  ASSERT_EQ(task_parallel.validation(), true);
  task_parallel.pre_processing();
  task_parallel.run();
  task_parallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_result(3, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_result.data()));
    taskDataSeq->outputs_count.emplace_back(expected_result.size());

    vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsSequentialMPI task_sequential(taskDataSeq);
    ASSERT_EQ(task_sequential.validation(), true);
    task_sequential.pre_processing();
    task_sequential.run();
    task_sequential.post_processing();

    ASSERT_EQ(global_result[0], expected_result[0]);  // min_diff
    ASSERT_EQ(global_result[1], expected_result[1]);  // index1
    ASSERT_EQ(global_result[2], expected_result[2]);  // index2
  }
}

TEST(vasilev_s_nearest_neighbor_elements_mpi, test_random_vector) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_result(3, 0);  // min_diff, index1, index2

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 1000;
    global_vec = vasilev_s_nearest_neighbor_elements_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsParallelMPI task_parallel(taskDataPar);
  ASSERT_EQ(task_parallel.validation(), true);
  task_parallel.pre_processing();
  task_parallel.run();
  task_parallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_result(3, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_result.data()));
    taskDataSeq->outputs_count.emplace_back(expected_result.size());

    vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsSequentialMPI task_sequential(taskDataSeq);
    ASSERT_EQ(task_sequential.validation(), true);
    task_sequential.pre_processing();
    task_sequential.run();
    task_sequential.post_processing();

    ASSERT_EQ(global_result[0], expected_result[0]);  // min_diff
    ASSERT_EQ(global_result[1], expected_result[1]);  // index1
    ASSERT_EQ(global_result[2], expected_result[2]);  // index2
  }
}

TEST(vasilev_s_nearest_neighbor_elements_mpi, test_equal_elements) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_result(3, 0);  // min_diff, index1, index2

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = {7, 7, 7, 7, 7};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsParallelMPI task_parallel(taskDataPar);
  ASSERT_EQ(task_parallel.validation(), true);
  task_parallel.pre_processing();
  task_parallel.run();
  task_parallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_result(3, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_result.data()));
    taskDataSeq->outputs_count.emplace_back(expected_result.size());

    vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsSequentialMPI task_sequential(taskDataSeq);
    task_sequential.validation();
    task_sequential.pre_processing();
    task_sequential.run();
    task_sequential.post_processing();

    ASSERT_EQ(global_result[0], expected_result[0]);  // min_diff
    ASSERT_EQ(global_result[1], expected_result[1]);  // index1
    ASSERT_EQ(global_result[2], expected_result[2]);  // index2
  }
}

TEST(vasilev_s_nearest_neighbor_elements_mpi, test_negative_numbers) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_result(3, 0);  // min_diff, index1, index2

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = {-10, -20, -15, -30, -25};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsParallelMPI task_parallel(taskDataPar);
  task_parallel.validation();
  task_parallel.pre_processing();
  task_parallel.run();
  task_parallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_result(3, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_result.data()));
    taskDataSeq->outputs_count.emplace_back(expected_result.size());

    vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsSequentialMPI task_sequential(taskDataSeq);
    ASSERT_EQ(task_sequential.validation(), true);
    task_sequential.pre_processing();
    task_sequential.run();
    task_sequential.post_processing();

    ASSERT_EQ(global_result[0], expected_result[0]);  // min_diff
    ASSERT_EQ(global_result[1], expected_result[1]);  // index1
    ASSERT_EQ(global_result[2], expected_result[2]);  // index2
  }
}

TEST(LocalResultTest, OperatorLessThan) {
  vasilev_s_nearest_neighbor_elements_mpi::LocalResult a{5, 10, 11};
  vasilev_s_nearest_neighbor_elements_mpi::LocalResult b{10, 5, 6};
  vasilev_s_nearest_neighbor_elements_mpi::LocalResult c{5, 9, 10};

  EXPECT_TRUE(a < b);
  EXPECT_FALSE(b < a);
  EXPECT_FALSE(a < c);
  EXPECT_TRUE(c < a);
}
