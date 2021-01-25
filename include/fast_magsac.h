#pragma once

#include <limits>
#include <chrono>
#include <memory>
#include "model.h"
#include "model_score.h"
#include "sampler.h"
#include "statistics.h"
#include "uniform_sampler.h"
#include <math.h> 
#include "gamma_values.h"
#include "GCoptimization.h"
#include <iostream>
#include "grid_neighborhood_graph.h"

#ifdef _WIN32 
	#include <ppl.h>
#endif

template <class DatumType, class ModelEstimator>
class FASTMAGSAC  
{
public:
	enum Version { 
		// The original version of MAGSAC. It works well, however, can be quite slow in many cases.
		MAGSAC_ORIGINAL, 
		// The recently proposed MAGSAC++ algorithm which keeps the accuracy of the original MAGSAC but is often orders of magnitude faster.
		MAGSAC_PLUS_PLUS }; 

	FASTMAGSAC(const Version magsac_version_ = Version::MAGSAC_PLUS_PLUS) :
		time_limit(std::numeric_limits<double>::max()), // 
		desired_fps(-1),
		iteration_limit(std::numeric_limits<size_t>::max()),
		maximum_threshold(10.0),
		apply_post_processing(true),
		mininum_iteration_number(50),
		partition_number(5),
		core_number(1),
		number_of_irwls_iters(1),
		interrupting_threshold(5.0),
		last_iteration_number(0),
		log_confidence(0),
		point_number(0),
		magsac_version(magsac_version_)
	{ 
	}

	~FASTMAGSAC() {}

	// A function to run MAGSAC.
	bool run(
		const cv::Mat &points_, // The input data points
		const double confidence_, // The required confidence in the results
		ModelEstimator& estimator_, // The model estimator
		gcransac::sampler::Sampler<cv::Mat, size_t> &sampler_, // The sampler used
		gcransac::Model &obtained_model_, // The estimated model parameters
		int &iteration_number_, // The number of iterations done
		ModelScore &model_score_); // The score of the estimated model

	bool run(
		const cv::Mat &points_, // The input data points
		const double confidence_, // The required confidence in the results
		ModelEstimator& estimator_, // The model estimator
		gcransac::sampler::Sampler<cv::Mat, size_t> &sampler_, // The sampler used
		gcransac::Model &obtained_model_, // The estimated model parameters
		int &iteration_number_, // The number of iterations done
		ModelScore &model_score_,
	    const gcransac::neighborhood::GridNeighborhoodGraph *neighborhood_graph_);
		
	// A function to set the maximum inlier-outlier threshold 
	void setMaximumThreshold(const double maximum_threshold_) 
	{
		maximum_threshold = maximum_threshold_;
	}

	// A function to set the inlier-outlier threshold used for speeding up the procedure
	// and for determining the required number of iterations.
	void setReferenceThreshold(const double threshold_)
	{
		interrupting_threshold = threshold_;
	}

	double getReferenceThreshold()
	{
		return interrupting_threshold;
	}

	// Setting the flag determining if post-processing is needed
	void applyPostProcessing(bool value_) 
	{
		apply_post_processing = value_;
	}

	// A function to set the maximum number of iterations
	void setIterationLimit(size_t iteration_limit_)
	{
		iteration_limit = iteration_limit_;
	}

	// A function to set the minimum number of iterations
	void setMinimumIterationNumber(size_t mininum_iteration_number_)
	{
		mininum_iteration_number = mininum_iteration_number_;
	}

	// A function to set the number of cores used in the original MAGSAC algorithm.
	// In MAGSAC++, it is not used. Note that when multiple MAGSACs run in parallel,
	// it is beneficial to keep the core number one for each independent MAGSAC.
	// Otherwise, the threads will act weirdly.
	void setCoreNumber(size_t core_number_)
	{
		if (magsac_version == MAGSAC_PLUS_PLUS)
			fprintf(stderr, "Setting the core number for MAGSAC++ is deprecated.");
		core_number = core_number_;
	}

	// Setting the number of partitions used in the original MAGSAC algorithm
	// to speed up the procedure. In MAGSAC++, this parameter is not used.
	void setPartitionNumber(size_t partition_number_)
	{
		if (magsac_version == MAGSAC_PLUS_PLUS)
			fprintf(stderr, "Setting the partition number for MAGSAC++ is deprecated.");
		partition_number = partition_number_;
	}

	// A function to set a desired minimum frames-per-second (FPS) value.
	void setFPS(int fps_) 
	{ 
		desired_fps = fps_; // The required FPS.
		// The time limit which the FPS implies
		time_limit = fps_ <= 0 ? 
			std::numeric_limits<double>::max() : 
			1.0 / fps_;
	}

	// get the weight based on residuel
	double getWeightFromRes(double res, double threshold, int type);

	// The post-processing algorithm applying sigma-consensus to the input model once.
	bool postProcessing(
		const cv::Mat &points, // All data points
		const gcransac::Model &so_far_the_best_model, // The input model to be improved
		gcransac::Model &output_model, // The improved model parameters
		ModelScore &output_score, // The score of the improved model
		const ModelEstimator &estimator); // The model estimator

	// The function determining the quality/score of a model using the original MAGSAC
	// criterion. Note that this function is significantly slower than the quality
	// function of MAGSAC++.
	void getModelQuality(
		const cv::Mat& points_, // All data points
		const gcransac::Model& model_, // The input model
		const ModelEstimator& estimator_, // The model estimator
		double& marginalized_iteration_number_, // The required number of iterations marginalized over the noise scale
		double& score_); // The score/quality of the model

	// The function determining the quality/score of a 
	// model using the MAGSAC++ criterion.
	void getModelQualityPlusPlus(
		const cv::Mat &points_, // All data points
		const gcransac::Model &model_, // The model parameter
		const ModelEstimator &estimator_, // The model estimator class
		double &score_, // The score to be calculated
		const double &previous_best_score_); // The score of the previous so-far-the-best model

	// The function determining the quality/score of a 
	// model using the MAGSAC++ criterion.
	void getModelQualityPlusPlus(
	const cv::Mat &points_, // All data points
	const gcransac::Model &model_, // The model parameter
	const ModelEstimator &estimator_, // The model estimator class
	double &score_, // The score to be calculated
	int &inlier_number_, //
	const double &previous_best_score_, int type, std::vector<double> &resduals_);

	// Returns a labeling w.r.t. the current model and point set using graph-cut
	void labeling(const cv::Mat &points_, // The input data points
		size_t neighbor_number_, // The neighbor number in the graph
		const gcransac::Model &model_, // The current model_
		const ModelEstimator &estimator_, // The model estimator
		double lambda_, // The weight for the spatial coherence term
		double threshold_, // The threshold_ for the inlier-outlier decision
		std::vector<bool> &inliers_labels_, // The resulting inlier set
		double &energy_,
		const gcransac::neighborhood::GridNeighborhoodGraph *neighborhood_graph_); // The resulting energy

	void labeling(const cv::Mat &points_, // The input data points
		std::vector<double> &residuals_, // The resulting inlier set
		double threshold_, // The threshold_ for the inlier-outlier decision
		std::vector<size_t> &inliers_, // The resulting inlier set
		std::vector<double> &weights_, // The resulting inlier set
		const gcransac::neighborhood::GridNeighborhoodGraph *neighborhood_graph_);

	// Return the constant reference of the scoring function
	const gcransac::utils::RANSACStatistics &getRansacStatistics() const { return statistics; }

	float splitResdauls(std::vector<double> resduals_, double maximum_threshold_, int split_number_,
						std::vector<int> &resdual_splits_, int &inllier_num_);
	float getWeightFromResiduals(std::vector<double> residuals_, double maximum_threshold_,
                                                          int split_number_, std::vector<int> &residual_splits_, int &inllier_num_,
														  std::vector<double> &weights_);

    size_t number_of_irwls_iters;
	gcransac::utils::RANSACStatistics statistics;

protected:
	Version magsac_version; // The version of MAGSAC used
	size_t iteration_limit; // Maximum number of iterations allowed
	size_t mininum_iteration_number; // Minimum number of iteration before terminating
	double maximum_threshold; // The maximum sigma value
	size_t core_number; // Number of core used in sigma-consensus
	double time_limit; // A time limit after the algorithm is interrupted
	int desired_fps; // The desired FPS (TODO: not tested with MAGSAC)
	bool apply_post_processing; // Decides if the post-processing step should be applied
	int point_number; // The current point number
	int last_iteration_number; // The iteration number implied by the last run of sigma-consensus
	double log_confidence; // The logarithm of the required confidence
	size_t partition_number; // Number of partitions used to speed up sigma-consensus
	double interrupting_threshold; // A threshold to speed up MAGSAC by interrupting the sigma-consensus procedure whenever there is no chance of being better than the previous so-far-the-best model

	bool sigmaConsensus(
		const cv::Mat& points_,
		const gcransac::Model& model_,
		gcransac::Model& refined_model_,
		ModelScore& score_,
		const ModelEstimator& estimator_,
		const ModelScore& best_score_);

	bool sigmaConsensusPlusPlus(
		const cv::Mat& points_,
		const gcransac::Model& model_,
		gcransac::Model& refined_model_,
		ModelScore& score_,
		const ModelEstimator& estimator_,
		const ModelScore& best_score_,
		const gcransac::neighborhood::GridNeighborhoodGraph *neighborhood_graph_);

	bool sigmaConsensusPlusPlus(
		const cv::Mat &points_,
		const gcransac::Model& model_,
		gcransac::Model& refined_model_,
		ModelScore &score_,
		const ModelEstimator &estimator_,
		const ModelScore &best_score_,
		std::vector<double> sum_weights_,
		gcransac::sampler::Sampler<cv::Mat, size_t> &sampler_);
};

template <class DatumType, class ModelEstimator>
void FASTMAGSAC<DatumType, ModelEstimator>::labeling(const cv::Mat &points_, // The input data points
		std::vector<double> &residuals_, // The resulting inlier set
		double threshold_, // The threshold_ for the inlier-outlier decision
		std::vector<size_t> &inliers_, // The resulting inlier set
		std::vector<double> &weights_, // The resulting inlier set
		const gcransac::neighborhood::GridNeighborhoodGraph *neighborhood_graph_)
		{
			const int &point_number = points_.rows;

			for (auto point_idx = 0; point_idx < point_number; ++point_idx)
			{
				if(residuals_[point_idx]>threshold_) continue;


				inliers_.push_back(point_idx);

				int neighbor_positive_cnt=0, neighbor_negetive_cnt=0;
				for (const size_t &actual_neighbor_idx : neighborhood_graph_->getNeighbors(point_idx))
				{
					if (actual_neighbor_idx == point_idx || actual_neighbor_idx < 0)
						continue;

						if(residuals_[actual_neighbor_idx]>threshold_)
						{
							neighbor_negetive_cnt++;
						}
						else
						{
							neighbor_positive_cnt++;
						}
				}

				if(neighbor_negetive_cnt+neighbor_positive_cnt)
				{
					weights_.push_back((neighbor_positive_cnt+1.0)/(neighbor_negetive_cnt+neighbor_positive_cnt+1.0));
				}
				else
				{
					weights_.push_back(0.6);
				}

			}
		}

// Returns a labeling w.r.t. the current model and point set using graph-cut
template <class DatumType, class ModelEstimator>
void FASTMAGSAC<DatumType, ModelEstimator>::labeling(const cv::Mat &points_, // The input data points
		size_t neighbor_number_, // The neighbor number in the graph
		const gcransac::Model &model_, // The current model_
		const ModelEstimator &estimator_, // The model estimator
		double lambda_, // The weight for the spatial coherence term
		double threshold_, // The threshold_ for the inlier-outlier decision
		std::vector<bool> &inliers_labels_, // The resulting inlier set
		double &energy_,
		const gcransac::neighborhood::GridNeighborhoodGraph *neighborhood_graph_)
		{
			const int &point_number = points_.rows;

			// Initializing the problem graph for the graph-cut algorithm.
			Energy<double, double, double> *problem_graph =
				new Energy<double, double, double>(point_number, // The number of vertices
					point_number*point_number, // The number of edges
					NULL);

			// Add a vertex for each point
			for (auto i = 0; i < point_number; ++i)
				problem_graph->add_node();

			// The distance and energy for each point
			std::vector<double> distance_per_threshold;
			distance_per_threshold.reserve(point_number);
			double tmp_squared_distance,
				tmp_energy;
			const double squared_truncated_threshold = threshold_;
			const double one_minus_lambda = 1.0 - lambda_;

			// Estimate the vertex capacities
			for (size_t i = 0; i < point_number; ++i)
			{
				// Calculating the point-to-model squared residual
				tmp_squared_distance = estimator_.squaredResidual(points_.row(i),
					model_.descriptor);
				// Storing the residual divided by the squared threshold 
				distance_per_threshold.emplace_back(
					std::clamp(tmp_squared_distance / squared_truncated_threshold, 0.0, 1.0));
				// Calculating the implied unary energy
				tmp_energy = 1 - distance_per_threshold.back();

				// Adding the unary energy to the graph
				if (tmp_squared_distance <= squared_truncated_threshold)
					problem_graph->add_term1(i, one_minus_lambda * tmp_energy, 0);
				else
					problem_graph->add_term1(i, 0, one_minus_lambda * (1 - tmp_energy));
			}

			for (auto point_idx = 0; point_idx < point_number; ++point_idx)
			{
				float energy1 = distance_per_threshold[point_idx]; // Truncated quadratic cost
				if(energy1>0.9999) continue;

				// Iterate through  all neighbors
				bool isBad = false;
				for (const size_t &actual_neighbor_idx : neighborhood_graph_->getNeighbors(point_idx))
				{

					if (actual_neighbor_idx == point_idx || actual_neighbor_idx < 0)
						continue;

					float energy2 = distance_per_threshold[actual_neighbor_idx]; // Truncated quadratic cost
					if(abs(energy2-energy1)>0.3) {
						isBad =true; break;
					}
				}

				if(isBad==false) continue;

				printf("%.3f: ", energy1);
				for (const size_t &actual_neighbor_idx : neighborhood_graph_->getNeighbors(point_idx))
				{
					if (actual_neighbor_idx == point_idx || actual_neighbor_idx < 0)
						continue;

					float energy2 = distance_per_threshold[actual_neighbor_idx]; // Truncated quadratic cost
					printf("%.3f ", energy2);
				}
				printf("\n");
			}

			std::vector<std::vector<int>> used_edges(point_number, std::vector<int>(point_number, 0));

			if (lambda_ > 0)
			{
				double energy1, energy2, energy_sum;
				double e00, e11 = 0; // Unused: e01 = 1.0, e10 = 1.0,

				// Iterate through all points and set their edges
				for (auto point_idx = 0; point_idx < point_number; ++point_idx)
				{
					energy1 = distance_per_threshold[point_idx]; // Truncated quadratic cost

					// Iterate through  all neighbors
					for (const size_t &actual_neighbor_idx : neighborhood_graph_->getNeighbors(point_idx))
					{
						if (actual_neighbor_idx == point_idx)
							continue;

						if (actual_neighbor_idx == point_idx || actual_neighbor_idx < 0)
							continue;

						if (used_edges[actual_neighbor_idx][point_idx] == 1 ||
							used_edges[point_idx][actual_neighbor_idx] == 1)
							continue;

						used_edges[actual_neighbor_idx][point_idx] = 1;
						used_edges[point_idx][actual_neighbor_idx] = 1;

						energy2 = distance_per_threshold[actual_neighbor_idx]; // Truncated quadratic cost
						energy_sum = energy1 + energy2;

						e00 = 0.5 * energy_sum;

						constexpr double e01_plus_e10 = 2.0; // e01 + e10 = 2
						if (e00 + e11 > e01_plus_e10)
							printf("Non-submodular expansion term detected; smooth costs must be a metric for expansion\n");

						// problem_graph->add_term2(point_idx, // The current point's index
						// 	actual_neighbor_idx, // The current neighbor's index
						// 	e00 * lambda_,
						// 	lambda_, // = e01 * lambda
						// 	lambda_, // = e10 * lambda
						// 	e11 * lambda_);
													problem_graph->add_term2(point_idx, // The current point's index
							actual_neighbor_idx, // The current neighbor's index
							0 * lambda_,
							lambda_, // = e01 * lambda
							lambda_, // = e10 * lambda
							0 * lambda_);
					}
				}
			}


			// Run the standard st-graph-cut algorithm
			problem_graph->minimize();

			// Select the inliers, i.e., the points labeled as SINK.
			inliers_labels_.resize(points_.rows);
			for (auto point_idx = 0; point_idx < points_.rows; ++point_idx)
			{
				if (problem_graph->what_segment(point_idx) == Graph<double, double, double>::SINK)
				{
					inliers_labels_[point_idx] = true;
				}
				else
				{
					inliers_labels_[point_idx] = false;
				}
			}
			// Clean the memory
			delete problem_graph;
		}

template <class DatumType, class ModelEstimator>
bool FASTMAGSAC<DatumType, ModelEstimator>::run(
	const cv::Mat& points_,
	const double confidence_,
	ModelEstimator& estimator_,
	gcransac::sampler::Sampler<cv::Mat, size_t> &sampler_,
	gcransac::Model& obtained_model_,
	int& iteration_number_,
	ModelScore &model_score_)
{
	// Initialize variables
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measuring: start and end times
	std::chrono::duration<double> elapsed_seconds; // Variables for time measuring: elapsed time
	log_confidence = log(1.0 - confidence_); // The logarithm of 1 - confidence
	point_number = points_.rows; // Number of points
	const int sample_size = estimator_.sampleSize(); // The sample size required for the estimation
	size_t max_iteration = iteration_limit; // The maximum number of iterations initialized to the iteration limit
	int iteration = 0; // Current number of iterations
	gcransac::Model so_far_the_best_model; // Current best model
	ModelScore so_far_the_best_score; // The score of the current best model
	std::unique_ptr<size_t[]> minimal_sample(new size_t[sample_size]); // The sample used for the estimation

	std::vector<double> sum_weights(point_number);
	so_far_the_best_score.score = -maximum_threshold;

	std::vector<size_t> pool(points_.rows);
	for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
	{
		pool[point_idx] = point_idx;
		sum_weights[point_idx] = 0.0;
	}
	
	if (points_.rows < sample_size)
	{	
		fprintf(stderr, "There are not enough points for applying robust estimation. Minimum is %d; while %d are given.\n", 
			sample_size, points_.rows);
		return false;
	}

	// Set the start time variable if there is some time limit set
	if (desired_fps > -1)
		start = std::chrono::system_clock::now();

	constexpr size_t max_unsuccessful_model_generations = 50;

	// Main MAGSAC iteration
	while (mininum_iteration_number > iteration ||
		iteration < max_iteration)
	{
		// Increase the current iteration number
		++iteration;
				
		// Sample a minimal subset
		std::vector<gcransac::Model> models; // The set of estimated models
		size_t unsuccessful_model_generations = 0; // The number of unsuccessful model generations
		// Try to select a minimal sample and estimate the implied model parameters
		while (++unsuccessful_model_generations < max_unsuccessful_model_generations)
		{
			// Get a minimal sample randomly
			if (!sampler_.sample(pool, // The index pool from which the minimal sample can be selected
				minimal_sample.get(), // The minimal sample
				sample_size)) // The size of a minimal sample
				continue;

			// Check if the selected sample is valid before estimating the model
			// parameters which usually takes more time. 
			if (!estimator_.isValidSample(points_, // All points
				minimal_sample.get())) // The current sample
				continue;

			// Estimate the model from the minimal sample
 			if (estimator_.estimateModel(points_, // All data points
				minimal_sample.get(), // The selected minimal sample
				&models)) // The estimated models
				break; 
		}

		// If the method was not able to generate any usable models, break the cycle.
		iteration += unsuccessful_model_generations - 1;
		statistics.iteration_number = iteration;

		// Select the so-far-the-best from the estimated models
		for (const auto &model : models)
		{
			ModelScore score; // The score of the current model
			gcransac::Model refined_model; // The refined model parameters

			// Apply sigma-consensus to refine the model parameters by marginalizing over the noise level sigma
			bool success;
			if (magsac_version == Version::MAGSAC_ORIGINAL)
				success = sigmaConsensus(points_,
					model,
					refined_model,
					score,
					estimator_,
					so_far_the_best_score);
			else
				success = sigmaConsensusPlusPlus(points_,
					model,
					refined_model,
					score,
					estimator_,
					so_far_the_best_score,
					sum_weights, 
					sampler_);

			// Continue if the model was rejected
			// if (!success || score.score == -1)
			if (!success)
				continue;

			// Save the iteration number when the current model is found
			score.iteration = iteration;
						
			// Update the best model parameters if needed
			if (so_far_the_best_score < score)
			// if (so_far_the_best_score.inlier_number < score.inlier_number)
			{
				statistics.better_models++;
				so_far_the_best_model = refined_model; // Update the best model parameters
				so_far_the_best_score = score; // Update the best model's score
				max_iteration = MIN(max_iteration, last_iteration_number); // Update the max iteration number, but do not allow to increase
			}
		}

		// Update the time parameters if a time limit is set
		if (desired_fps > -1)
		{
			end = std::chrono::system_clock::now();
			elapsed_seconds = end - start;

			// Interrupt if the time limit is exceeded
			if (elapsed_seconds.count() > time_limit)
				break;
		}
	}
	
	// Apply sigma-consensus as a post processing step if needed and the estimated model is valid
	if (apply_post_processing)
	{
		// TODO
	}
	
	obtained_model_ = so_far_the_best_model;
	iteration_number_ = iteration;
	model_score_ = so_far_the_best_score;

	return so_far_the_best_score.score > 0;
}


template <class DatumType, class ModelEstimator>
bool FASTMAGSAC<DatumType, ModelEstimator>::run(
	const cv::Mat& points_,
	const double confidence_,
	ModelEstimator& estimator_,
	gcransac::sampler::Sampler<cv::Mat, size_t> &sampler_,
	gcransac::Model& obtained_model_,
	int& iteration_number_,
	ModelScore &model_score_,
	const gcransac::neighborhood::GridNeighborhoodGraph *neighborhood_graph_)
{
	// Initialize variables
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measuring: start and end times
	std::chrono::duration<double> elapsed_seconds; // Variables for time measuring: elapsed time
	log_confidence = log(1.0 - confidence_); // The logarithm of 1 - confidence
	point_number = points_.rows; // Number of points
	const int sample_size = estimator_.sampleSize(); // The sample size required for the estimation
	size_t max_iteration = iteration_limit; // The maximum number of iterations initialized to the iteration limit
	int iteration = 0; // Current number of iterations
	gcransac::Model so_far_the_best_model; // Current best model
	ModelScore so_far_the_best_score; // The score of the current best model
	std::unique_ptr<size_t[]> minimal_sample(new size_t[sample_size]); // The sample used for the estimation

	std::vector<double> sum_weights(point_number);
	so_far_the_best_score.score = -maximum_threshold;

	std::vector<size_t> pool(points_.rows);
	for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
	{
		pool[point_idx] = point_idx;
		sum_weights[point_idx] = 0.0;
	}
	
	if (points_.rows < sample_size)
	{	
		fprintf(stderr, "There are not enough points for applying robust estimation. Minimum is %d; while %d are given.\n", 
			sample_size, points_.rows);
		return false;
	}

	// Set the start time variable if there is some time limit set
	if (desired_fps > -1)
		start = std::chrono::system_clock::now();

	constexpr size_t max_unsuccessful_model_generations = 50;

	// Main MAGSAC iteration
	while (mininum_iteration_number > iteration ||
		iteration < max_iteration)
	{
		// Increase the current iteration number
		++iteration;
				
		// Sample a minimal subset
		std::vector<gcransac::Model> models; // The set of estimated models
		size_t unsuccessful_model_generations = 0; // The number of unsuccessful model generations
		// Try to select a minimal sample and estimate the implied model parameters
		while (++unsuccessful_model_generations < max_unsuccessful_model_generations)
		{
			// Get a minimal sample randomly
			if (!sampler_.sample(pool, // The index pool from which the minimal sample can be selected
				minimal_sample.get(), // The minimal sample
				sample_size)) // The size of a minimal sample
				continue;

			// Check if the selected sample is valid before estimating the model
			// parameters which usually takes more time. 
			if (!estimator_.isValidSample(points_, // All points
				minimal_sample.get())) // The current sample
				continue;

			// Estimate the model from the minimal sample
 			if (estimator_.estimateModel(points_, // All data points
				minimal_sample.get(), // The selected minimal sample
				&models)) // The estimated models
				break; 
		}

		// If the method was not able to generate any usable models, break the cycle.
		iteration += unsuccessful_model_generations - 1;
		statistics.iteration_number = iteration;

		// Select the so-far-the-best from the estimated models
		for (const auto &model : models)
		{
			ModelScore score; // The score of the current model
			gcransac::Model refined_model; // The refined model parameters

			// Apply sigma-consensus to refine the model parameters by marginalizing over the noise level sigma
			bool success;
			if (magsac_version == Version::MAGSAC_ORIGINAL)
				success = sigmaConsensus(points_,
					model,
					refined_model,
					score,
					estimator_,
					so_far_the_best_score);
			else
				success = sigmaConsensusPlusPlus(points_,
					model,
					refined_model,
					score,
					estimator_,
					so_far_the_best_score, neighborhood_graph_);

			// Continue if the model was rejected
			// if (!success || score.score == -1)
			if (!success)
				continue;

			// Save the iteration number when the current model is found
			score.iteration = iteration;
						
			// Update the best model parameters if needed
			if (so_far_the_best_score < score)
			// if (so_far_the_best_score.inlier_number < score.inlier_number)
			{
				statistics.better_models++;
				so_far_the_best_model = refined_model; // Update the best model parameters
				so_far_the_best_score = score; // Update the best model's score
				max_iteration = MIN(max_iteration, last_iteration_number); // Update the max iteration number, but do not allow to increase
			}
		}

		// Update the time parameters if a time limit is set
		if (desired_fps > -1)
		{
			end = std::chrono::system_clock::now();
			elapsed_seconds = end - start;

			// Interrupt if the time limit is exceeded
			if (elapsed_seconds.count() > time_limit)
				break;
		}
	}
	
	// Apply sigma-consensus as a post processing step if needed and the estimated model is valid
	if (apply_post_processing)
	{
		// TODO
	}
	
	obtained_model_ = so_far_the_best_model;
	iteration_number_ = iteration;
	model_score_ = so_far_the_best_score;

	return so_far_the_best_score.score > 0;
}

template <class DatumType, class ModelEstimator>
bool FASTMAGSAC<DatumType, ModelEstimator>::postProcessing(
	const cv::Mat &points_,
	const gcransac::Model &model_,
	gcransac::Model &refined_model_,
	ModelScore &refined_score_,
	const ModelEstimator &estimator_)
{
	fprintf(stderr, "Sigma-consensus++ is not implemented yet as post-processing.\n");
	return false;
}


template <class DatumType, class ModelEstimator>
bool FASTMAGSAC<DatumType, ModelEstimator>::sigmaConsensus(
	const cv::Mat &points_,
	const gcransac::Model& model_,
	gcransac::Model& refined_model_,
	ModelScore &score_,
	const ModelEstimator &estimator_,
	const ModelScore &best_score_)
{
	// Set up the parameters
	constexpr double L = 1.05;
	constexpr double k = ModelEstimator::getSigmaQuantile();
	constexpr double threshold_to_sigma_multiplier = 1.0 / k;
	constexpr size_t sample_size = estimator_.sampleSize();
	static auto comparator = [](std::pair<double, int> left, std::pair<double, int> right) { return left.first < right.first; };
	const int point_number = points_.rows;
	double current_maximum_sigma = this->maximum_threshold;

	// Calculating the residuals
	std::vector< std::pair<double, size_t> > all_residuals;
	all_residuals.reserve(point_number);

	// If it is not the first run, consider the previous best and interrupt the validation when there is no chance of being better
	if (best_score_.inlier_number > 0)
	{
		// Number of inliers which should be exceeded
		int points_remaining = best_score_.inlier_number;
		int inlier_number = 0;

		// Collect the points which are closer than the threshold which the maximum sigma implies
		for (int point_idx = 0; point_idx < point_number; ++point_idx)
		{
			// Calculate the residual of the current point
			const double residual = estimator_.residual(points_.row(point_idx), model_);
			if (current_maximum_sigma > residual)
			{
				// Store the residual of the current point and its index
				all_residuals.emplace_back(std::make_pair(residual, point_idx));

				// Count points which are closer than a reference threshold to speed up the procedure
				if (residual < interrupting_threshold)
				{
					--points_remaining;  inlier_number++;
				}
			}

			// Interrupt if there is no chance of being better
			// TODO: replace this part by SPRT test
			// if (point_number - point_idx < points_remaining)
				// return false;
		}

		// Store the number of really close inliers just to speed up the procedure
		// by interrupting the next verifications.
		// score_.inlier_number = best_score_.inlier_number - points_remaining;
		score_.inlier_number = inlier_number;
		if(inlier_number<best_score_.inlier_number) return false;
	}
	else
	{
		// The number of really close points
		size_t points_close = 0;

		// Collect the points which are closer than the threshold which the maximum sigma implies
		for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
		{
			// Calculate the residual of the current point
			const double residual = estimator_.residual(points_.row(point_idx), model_);
			if (current_maximum_sigma > residual)
			{
				// Store the residual of the current point and its index
				all_residuals.emplace_back(std::make_pair(residual, point_idx));

				// Count points which are closer than a reference threshold to speed up the procedure
				if (residual < interrupting_threshold)
					++points_close;
			}
		}

		// Store the number of really close inliers just to speed up the procedure
		// by interrupting the next verifications.
		score_.inlier_number = points_close;
	}

	std::vector<gcransac::Model> sigma_models;
	std::vector<size_t> sigma_inliers;
	std::vector<double> final_weights;
	
	// The number of possible inliers
	const size_t possible_inlier_number = all_residuals.size();

	// Sort the residuals in ascending order
	std::sort(all_residuals.begin(), all_residuals.end(), comparator);

	// The maximum threshold is set to be slightly bigger than the distance of the
	// farthest possible inlier.
	current_maximum_sigma =
		all_residuals.back().first + std::numeric_limits<double>::epsilon();

	const double sigma_step = current_maximum_sigma / partition_number;

	last_iteration_number = 10000;

	score_.score = 0;

	// The weights calculated by each parallel process
	std::vector<std::vector<double>> point_weights_par(partition_number, std::vector<double>(possible_inlier_number, 0));

	// If OpenMP is used, calculate things in parallel
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(core_number)
	for (int partition_idx = 0; partition_idx < partition_number; ++partition_idx)
	{
		// The maximum sigma value in the current partition
		const double max_sigma = (partition_idx + 1) * sigma_step;

		// Find the last element which has smaller distance than 'max_threshold'
		// Since the vector is ordered binary search can be used to find that particular element.
		const auto &last_element = std::upper_bound(all_residuals.begin(), all_residuals.end(), std::make_pair(max_sigma, 0), comparator);
		const size_t sigma_inlier_number = last_element - all_residuals.begin();

		// Put the indices into a vector
		std::vector<size_t> sigma_inliers;
		sigma_inliers.reserve(sigma_inlier_number);

		// Store the points which are closer than the current sigma limit
		for (size_t relative_point_idx = 0; relative_point_idx < sigma_inlier_number; ++relative_point_idx)
			sigma_inliers.emplace_back(all_residuals[relative_point_idx].second);

		// Check if there are enough inliers to fit a model
		if (sigma_inliers.size() > sample_size)
		{
			// Estimating the model which the current set of inliers imply
			std::vector<gcransac::Model> sigma_models;
			estimator_.estimateModelNonminimal(points_,
				&(sigma_inliers)[0],
				sigma_inlier_number,
				&sigma_models);

			// If the estimation was successful calculate the implied probabilities
			if (sigma_models.size() == 1)
			{
				const double max_sigma_squared_2 = 2 * max_sigma * max_sigma;
				double residual_i_2, // The residual of the i-th point
					probability_i; // The probability of the i-th point

				// Iterate through all points to estimate the related probabilities
				for (size_t relative_point_idx = 0; relative_point_idx < sigma_inliers.size(); ++relative_point_idx)
				{
					// TODO: Replace with Chi-square instead of normal distribution
					const size_t &point_idx = sigma_inliers[relative_point_idx];

					// Calculate the residual of the current point
					residual_i_2 = estimator_.squaredResidual(points_.row(point_idx),
						sigma_models[0]);

					// Calculate the probability of the i-th point assuming Gaussian distribution
					// TODO: replace by Chi-square distribution
					probability_i = exp(-residual_i_2 / max_sigma_squared_2);

					// Store the probability of the i-th point coming from the current partition
					point_weights_par[partition_idx][relative_point_idx] += probability_i;


				}
			}
		}
	}
#else
	fprintf(stderr, "Not implemented yet.\n");
#endif

	// The weights used for the final weighted least-squares fitting
	final_weights.reserve(possible_inlier_number);

	// Collect all points which has higher probability of being inlier than zero
	sigma_inliers.reserve(possible_inlier_number);
	for (size_t point_idx = 0; point_idx < possible_inlier_number; ++point_idx)
	{
		// Calculate the weight of the current point
		double weight = 0.0;
		for (size_t partition_idx = 0; partition_idx < partition_number; ++partition_idx)
			weight += point_weights_par[partition_idx][point_idx];

		// If the weight is approx. zero, continue.
		if (weight < std::numeric_limits<double>::epsilon())
			continue;

		// Store the index and weight of the current point
		sigma_inliers.emplace_back(all_residuals[point_idx].second);
		final_weights.emplace_back(weight);
	}

	// If there are fewer inliers than the size of the minimal sample interupt the procedure
	if (sigma_inliers.size() < sample_size)
		return false;

	// Estimate the model parameters using weighted least-squares fitting
	if (!estimator_.estimateModelNonminimal(
		points_, // All input points
		&(sigma_inliers)[0], // Points which have higher than 0 probability of being inlier
		static_cast<int>(sigma_inliers.size()), // Number of possible inliers
		&sigma_models, // Estimated models
		&(final_weights)[0])) // Weights of points 
		return false;

	bool is_model_updated = false;
	
	if (sigma_models.size() == 1 && // If only a single model is estimated
		estimator_.isValidModel(sigma_models.back(),
			points_,
			sigma_inliers,
			&(sigma_inliers)[0],
			interrupting_threshold,
			is_model_updated)) // and it is valid
	{
		// Return the refined model
		refined_model_ = sigma_models.back();

		// Calculate the score of the model and the implied iteration number
		double marginalized_iteration_number;
		getModelQuality(points_, // All the input points
			refined_model_, // The estimated model
			estimator_, // The estimator
			marginalized_iteration_number, // The marginalized inlier ratio
			score_.score); // The marginalized score

		if (marginalized_iteration_number < 0 || std::isnan(marginalized_iteration_number))
			last_iteration_number = std::numeric_limits<int>::max();
		else
			last_iteration_number = static_cast<int>(round(marginalized_iteration_number));
		return true;
	}
	return false;
}

template <class DatumType, class ModelEstimator>
bool FASTMAGSAC<DatumType, ModelEstimator>::sigmaConsensusPlusPlus(
	const cv::Mat &points_,
	const gcransac::Model& model_,
	gcransac::Model& refined_model_,
	ModelScore &score_,
	const ModelEstimator &estimator_,
	const ModelScore &best_score_,
	std::vector<double> sum_weights_,
	gcransac::sampler::Sampler<cv::Mat, size_t> &sampler_)
{
	constexpr size_t sample_size = estimator_.nonMinimalSampleSize();
	const int point_number = points_.rows;
	// The manually set maximum inlier-outlier threshold
	double current_maximum_sigma = this->maximum_threshold;
	// Calculating the pairs of (residual, point index).
	std::vector< std::pair<double, size_t> > residuals;
	// Occupy the maximum required memory to avoid doing it later.
	residuals.reserve(point_number);

	const int weight_type = 2;
	interrupting_threshold = (current_maximum_sigma);

	int points_remaining = best_score_.inlier_number;
	int inlier_number = 0;
	double score = 0;
	std::vector<double> resdual_all;

	getModelQualityPlusPlus(points_, // All the input points
	model_, // The estimated model
	estimator_, // The estimator
	score, // The marginalized score
	inlier_number,
	best_score_.score, 0, resdual_all);

	score_.inlier_number = inlier_number;
	score_.init_score = score;
	// if(inlier_number<best_score_.inlier_number || inlier_number<sample_size)
	if(score<best_score_.init_score || inlier_number<estimator_.nonMinimalSampleSize())
	{
		return false;
	} 

	statistics.accepted_models++;

	// printf("local_inlier_th: %f %d\n", local_inlier_th, inlier_number);

	// Collect the points which are closer than the threshold which the maximum sigma implies
	// std::vector<double>  local_weights_1(point_number),  local_weights_2(point_number);
	// for (int point_idx = 0; point_idx < point_number; ++point_idx)
	// {
	// 	// sum_weights_[point_idx] = getWeightFromRes(resdual_all[point_idx], local_inlier_th, weight_type);
	// 	sum_weights_[point_idx] = getWeightFromRes(resdual_all[point_idx], interrupting_threshold, weight_type);
	// 	// local_weights_1[point_idx] = getWeightFromRes(resdual_all[point_idx], interrupting_threshold*0.8, weight_type);
	// 	// local_weights_2[point_idx] = getWeightFromRes(resdual_all[point_idx], interrupting_threshold*1.5, weight_type);
	// }

	// Points used in the weighted least-squares fitting
	std::vector<size_t> inliers;
	// The current sample used in the inner RANSAC
	// const auto &inlier_limit = estimator_.inlierLimit();
	const auto &inlier_limit = sample_size*200;
  	std::unique_ptr<size_t[]> current_sample(new size_t[inlier_limit]);
	for (int point_idx = 0; point_idx < point_number; ++point_idx)
	{
		if(resdual_all[point_idx] < this->maximum_threshold)
		{
			inliers.push_back(point_idx);
		}
	}


	// Initialize the polished model with the initial one
	gcransac::Model polished_model = model_;
	// A flag to determine if the initial model has been updated
	bool updated = false;
	bool model_valid = false;
	bool is_model_updated = false;
	std::vector<double> refine_residuals;
	int inlier_num;
	double max_score = -1;            // The current best score
	gcransac::Model best_model;           // The current best model
	std::vector<gcransac::Model> models; 

	// if (estimator_.estimateModelNonminimal(
	// 		points_,           // All input points
	// 		&(all_inliers)[0], // Points which have higher than 0 probability of being inlier
	// 		static_cast<int>(all_inliers.size()), // Number of possible inliers
	// 		&sum_models,                          // Estimated models
	// 		&(sum_weights_)[0]))                  // Weights of points
	// {
	// 	if (estimator_.isValidModel(sum_models[0], points_, all_inliers, &(all_inliers[0]),
	// 								interrupting_threshold, is_sum_model_updated)) {
	// 		sum_model_valid = true;

	// 		refined_model_ = sum_models[0];

	// 		getModelQualityPlusPlus(points_, // All the input points
	// 		refined_model_, // The estimated model
	// 		estimator_, // The estimator
	// 		local_score1, // The marginalized score
	// 		inlier_num,
	// 		best_score_.score, 0, refine_residuals); // The score of the previous so-far-the-best model

	// 		score_.score = local_score1;

	// 		// Update the iteration number
	// 	last_iteration_number = log_confidence / log(1.0 - std::pow(static_cast<double>(score_.inlier_number) / point_number, sample_size));
	// 	// last_iteration_number = 3 * log_confidence / log(1.0 - std::pow(static_cast<double>(inlier_num) / point_number, sample_size));
		
	// 	// std::vector<int> resdual_splits;
	// 	// printf("inlier_num:%d  score:%f\n", inlier_num, score_.score);
	// 	// printf("better: %d \n", best_score_.score<score_.score);

	// 	// splitResdauls(refine_residuals, 32, 16, resdual_splits);

	// 	}
	// }

	// Apply the local optimization
	for (unsigned int i = 0; i < 10; ++i) {
		// Reset the model vector
		models.clear();
		if (inlier_limit < inliers.size()) // If there are more inliers available than the minimum
											// number, sample randomly.
		{

		// Get a minimal sample randomly
		if (!sampler_.sample(inliers,              // The index pool from which the minimal sample can be selected
								current_sample.get(), // The minimal sample
								inlier_limit))     // The size of a minimal sample
			continue;

		// Apply least-squares model fitting to the selected points.
		// If it fails, continue the for cycle and, thus, the sampling.
		if (!estimator_.estimateModelNonminimal(points_,              // The input data points
												current_sample.get(), // The selected sample
												inlier_limit,      // The size of the sample
												&models))             // The estimated model parameter
			continue;
		} else if (sample_size < inliers.size()) // If there are enough inliers to estimate the
												// model, use all of them
		{
		// Apply least-squares model fitting to the selected points.
		// If it fails, break the for cycle since we have used all inliers for
		// this step.
		if (!estimator_.estimateModelNonminimal(points_,        // The input data points
												&inliers[0],    // The selected sample
												inliers.size(), // The size of the sample
												&models))       // The estimated model parameter
			break;
		} else {
		break;
		}

		// if (estimator_.isValidModel(models[0], points_, inliers, &(inliers[0]), interrupting_threshold,
		// 							is_model_updated)) 
		{

		double local_score;
		getModelQualityPlusPlus(points_,     // All the input points
								models[0],   // The estimated model
								estimator_,  // The estimator
								local_score, // The marginalized score
								inlier_num, best_score_.score, 0,
								refine_residuals); // The score of the previous so-far-the-best model

			if (local_score > max_score) {
				updated = true;

				refined_model_ = models[0];
				max_score = local_score;
				// Update the iteration number
				last_iteration_number =
					log_confidence /
					log(1.0 - std::pow(static_cast<double>(score_.inlier_number) / point_number, sample_size));
			}
		}
	}

        if(updated)
	{
		score_.score = max_score;
		return true;
	}

	return false;
}


template <class DatumType, class ModelEstimator>
bool FASTMAGSAC<DatumType, ModelEstimator>::sigmaConsensusPlusPlus(
	const cv::Mat &points_,
	const gcransac::Model& model_,
	gcransac::Model& refined_model_,
	ModelScore &score_,
	const ModelEstimator &estimator_,
	const ModelScore &best_score_,
	const gcransac::neighborhood::GridNeighborhoodGraph *neighborhood_graph_)
{
	constexpr size_t sample_size = estimator_.nonMinimalSampleSize();
	const int point_number = points_.rows;
	// The manually set maximum inlier-outlier threshold
	double current_maximum_sigma = this->maximum_threshold;
	// Calculating the pairs of (residual, point index).
	std::vector< std::pair<double, size_t> > residuals;
	// Occupy the maximum required memory to avoid doing it later.
	residuals.reserve(point_number);

	const int weight_type = 2;
	interrupting_threshold = (current_maximum_sigma);

	int points_remaining = best_score_.inlier_number;
	int inlier_number = 0;
	double score = 0;
	std::vector<double> residuals_init;

	getModelQualityPlusPlus(points_, // All the input points
	model_, // The estimated model
	estimator_, // The estimator
	score, // The marginalized score
	inlier_number,
	best_score_.score, 0, residuals_init);

	score_.inlier_number = inlier_number;
	score_.init_score = score;
	// if(inlier_number<best_score_.inlier_number || inlier_number<sample_size)
	if(score<best_score_.init_score || inlier_number<estimator_.nonMinimalSampleSize())
	{
		return false;
	} 

	statistics.accepted_models++;

	// printf("local_inlier_th: %f %d\n", local_inlier_th, inlier_number);

	// Collect the points which are closer than the threshold which the maximum sigma implies
	// std::vector<double>  local_weights_1(point_number),  local_weights_2(point_number);
	// for (int point_idx = 0; point_idx < point_number; ++point_idx)
	// {
	// 	// sum_weights_[point_idx] = getWeightFromRes(resdual_all[point_idx], local_inlier_th, weight_type);
	// 	sum_weights_[point_idx] = getWeightFromRes(resdual_all[point_idx], interrupting_threshold, weight_type);
	// 	// local_weights_1[point_idx] = getWeightFromRes(resdual_all[point_idx], interrupting_threshold*0.8, weight_type);
	// 	// local_weights_2[point_idx] = getWeightFromRes(resdual_all[point_idx], interrupting_threshold*1.5, weight_type);
	// }

	// gragh-cut labeling
	// std::vector<bool> inliers_labels;
	// double energy;
	// labeling(points_, 0, model_, estimator_, 0.975, 
	// 	interrupting_threshold, inliers_labels, energy, neighborhood_graph_);

	// Points used in the weighted least-squares fitting
	// std::vector<size_t> inliers;
	// std::vector<double> local_weights;
	// const auto &inlier_limit = sample_size*2;
  	// std::unique_ptr<size_t[]> current_sample(new size_t[inlier_limit]);
	// for (int point_idx = 0; point_idx < point_number; ++point_idx)
	// {
	// 	if(residuals_init[point_idx] < this->maximum_threshold)
	// 	{
	// 		inliers.push_back(point_idx);

	// 		if(inliers_labels[point_idx])
	// 		{
	// 			local_weights.push_back(1.0);
	// 		}
	// 		else
	// 		{
	// 			local_weights.push_back(0.0);
	// 		}
	// 	}
	// }

		// 
		std::vector<size_t> inliers;
		std::vector<double> local_weights;
		labeling(points_, residuals_init, 
		interrupting_threshold, inliers, 
		local_weights, neighborhood_graph_);

	// for(int i=0; i<local_weights.size(); i++)
	// {
	// 	if(local_weights[i]<0.8)
	// 	printf("%.2f %.2f|| ", residuals_init[inliers[i]], local_weights[i]);
	// }		
	// printf("\n");

	// printf("%d %d \n", inliers.size(), local_weights.size());

	// Initialize the polished model with the initial one
	gcransac::Model polished_model = model_;
	// A flag to determine if the initial model has been updated
	bool updated = false;
	bool model_valid = false;
	bool is_model_updated = false;
	std::vector<double> refine_residuals;
	int inlier_num;
	double max_score = -1;            // The current best score
	gcransac::Model best_model;           // The current best model
	std::vector<gcransac::Model> models; 

	// Apply the local optimization
	for (unsigned int i = 0; i < 1; ++i) {
		// Reset the model vector
		models.clear();
		 if (sample_size < inliers.size()) // If there are enough inliers to estimate the
												// model, use all of them
		{
		// Apply least-squares model fitting to the selected points.
		// If it fails, break the for cycle since we have used all inliers for
		// this step.
		if (!estimator_.estimateModelNonminimal(points_,        // The input data points
												&(inliers)[0],    // The selected sample
												inliers.size(), // The size of the sample
												&models, 
											    &(local_weights)[0]))       // The estimated model parameter
			break;
		} else {
		break;
		}

		// if (estimator_.isValidModel(models[0], points_, inliers, &(inliers[0]), interrupting_threshold,
		// 							is_model_updated)) 
		{

		double local_score;
		getModelQualityPlusPlus(points_,     // All the input points
								models[0],   // The estimated model
								estimator_,  // The estimator
								local_score, // The marginalized score
								inlier_num, best_score_.score, 0,
								refine_residuals); // The score of the previous so-far-the-best model

			if (local_score > max_score) {
				updated = true;

				refined_model_ = models[0];
				max_score = local_score;
				// Update the iteration number
				last_iteration_number =
					log_confidence /
					log(1.0 - std::pow(static_cast<double>(score_.inlier_number) / point_number, sample_size));
			}
		}
	}

        if(updated)
	{
		score_.score = max_score;
		return true;
	}

	return false;
}



// template <class DatumType, class ModelEstimator>
// bool FASTMAGSAC<DatumType, ModelEstimator>::sigmaConsensusPlusPlus(
// 	const cv::Mat &points_,
// 	const gcransac::Model& model_,
// 	gcransac::Model& refined_model_,
// 	ModelScore &score_,
// 	const ModelEstimator &estimator_,
// 	const ModelScore &best_score_,
// 	std::vector<double> sum_weights_)
// {
// 	// The degrees of freedom of the data from which the model is estimated.
// 	// E.g., for models coming from point correspondences (x1,y1,x2,y2), it is 4.
// 	constexpr size_t degrees_of_freedom = ModelEstimator::getDegreesOfFreedom();
// 	// A 0.99 quantile of the Chi^2-distribution to convert sigma values to residuals
// 	constexpr double k = ModelEstimator::getSigmaQuantile();
// 	// A multiplier to convert residual values to sigmas
// 	constexpr double threshold_to_sigma_multiplier = 1.0 / k;
// 	// Calculating k^2 / 2 which will be used for the estimation and, 
// 	// due to being constant, it is better to calculate it a priori.
// 	constexpr double squared_k_per_2 = k * k / 2.0;
// 	// Calculating (DoF - 1) / 2 which will be used for the estimation and, 
// 	// due to being constant, it is better to calculate it a priori.
// 	constexpr double dof_minus_one_per_two = (degrees_of_freedom - 1.0) / 2.0;
// 	// TODO: check
// 	constexpr double C = ModelEstimator::getC();
// 	// The size of a minimal sample used for the estimation
// 	constexpr size_t sample_size = estimator_.sampleSize();
// 	// Calculating 2^(DoF - 1) which will be used for the estimation and, 
// 	// due to being constant, it is better to calculate it a priori.
// 	static const double two_ad_dof = std::pow(2.0, dof_minus_one_per_two);
// 	// Calculating C * 2^(DoF - 1) which will be used for the estimation and, 
// 	// due to being constant, it is better to calculate it a priori.
// 	static const double C_times_two_ad_dof = C * two_ad_dof;
// 	// Calculating the gamma value of (DoF - 1) / 2 which will be used for the estimation and, 
// 	// due to being constant, it is better to calculate it a priori.
// 	static const double gamma_value = tgamma(dof_minus_one_per_two);
// 	// Calculating the upper incomplete gamma value of (DoF - 1) / 2 with k^2 / 2.
// 	constexpr double gamma_k = ModelEstimator::getUpperIncompleteGammaOfK();
// 	// Calculating the lower incomplete gamma value of (DoF - 1) / 2 which will be used for the estimation and, 
// 	// due to being constant, it is better to calculate it a priori.
// 	static const double gamma_difference = gamma_value - gamma_k;
// 	// The number of points provided
// 	const int point_number = points_.rows;
// 	// The manually set maximum inlier-outlier threshold
// 	double current_maximum_sigma = this->maximum_threshold;
// 	// Calculating the pairs of (residual, point index).
// 	std::vector< std::pair<double, size_t> > residuals;
// 	// Occupy the maximum required memory to avoid doing it later.
// 	residuals.reserve(point_number);

// 	const int weight_type = 2;
// 	// interrupting_threshold = sqrt(current_maximum_sigma)*1.5;
// 	interrupting_threshold = (current_maximum_sigma);

// 	// If it is not the first run, consider the previous best and interrupt the validation when there is no chance of being better
// 	// if (best_score_.inlier_number > 0)
// 		// Number of points close to the previous so-far-the-best model. 
// 		// This model should have more inliers.
// 		int points_remaining = best_score_.inlier_number;
// 		int inlier_number = 0;
// 		double score = 0;
// 		std::vector<double> resdual_all;

// 		getModelQualityPlusPlus(points_, // All the input points
// 		model_, // The estimated model
// 		estimator_, // The estimator
// 		score, // The marginalized score
// 		inlier_number,
// 		best_score_.score, 0, resdual_all);

// 		score_.inlier_number = inlier_number;
// 		score_.init_score = score;
// 		if(inlier_number<best_score_.inlier_number || inlier_number<estimator_.nonMinimalSampleSize())
// 		// if(score<best_score_.init_score || inlier_number<estimator_.nonMinimalSampleSize())
// 		{
// 			return false;
// 		} 
	
// 		statistics.accepted_models++;

// 		std::vector<int> resdual_splits;
// 		// printf("ite:%d inlier%d\n", statistics.iteration_number, score_.inlier_number);
// 		// float local_inlier_th = splitResdauls(resdual_all, 40, 40, resdual_splits, inlier_number);
// 		// float local_inlier_th = splitResdauls(resdual_all, 8, 20, resdual_splits, inlier_number);
// 		float local_inlier_th = splitResdauls(resdual_all, 8, 16, resdual_splits, inlier_number);

// 		// printf("local_inlier_th: %f %d\n", local_inlier_th, inlier_number);

// 		// local_inlier_th = getWeightFromResiduals(resdual_all, 6, 16, resdual_splits, inlier_number, sum_weights_);

// 		// Collect the points which are closer than the threshold which the maximum sigma implies
// 		std::vector<double>  local_weights_1(point_number),  local_weights_2(point_number);
// 		for (int point_idx = 0; point_idx < point_number; ++point_idx)
// 		{
// 			// sum_weights_[point_idx] = getWeightFromRes(resdual_all[point_idx], local_inlier_th, weight_type);
// 			sum_weights_[point_idx] = getWeightFromRes(resdual_all[point_idx], interrupting_threshold, weight_type);
// 			// local_weights_1[point_idx] = getWeightFromRes(resdual_all[point_idx], interrupting_threshold*0.8, weight_type);
// 			// local_weights_2[point_idx] = getWeightFromRes(resdual_all[point_idx], interrupting_threshold*1.5, weight_type);
// 		}


// 	// Models fit by weighted least-squares fitting
// 	std::vector<gcransac::Model> sigma_models;
// 	// Points used in the weighted least-squares fitting
// 	std::vector<size_t> sigma_inliers;
// 	// Weights used in the the weighted least-squares fitting
// 	std::vector<double> sigma_weights;
// 	// Number of points considered in the fitting
// 	const size_t possible_inlier_number = residuals.size();
// 	// Occupy the memory to avoid doing it inside the calculation possibly multiple times
// 	sigma_inliers.reserve(possible_inlier_number);
// 	// Occupy the memory to avoid doing it inside the calculation possibly multiple times
// 	sigma_weights.reserve(possible_inlier_number);

// 	// The current sample used in the inner RANSAC
// 	const auto &inlier_limit = estimator_.inlierLimit();
//   	std::unique_ptr<size_t[]> current_sample(new size_t[inlier_limit]);

// 	for (int point_idx = 0; point_idx < point_number; ++point_idx)
// 	{
// 		if(resdual_all[point_idx] < this->maximum_threshold)
// 		{
// 			sigma_inliers.push_back(point_idx);
// 		}
// 	}


// 	// Initialize the polished model with the initial one
// 	gcransac::Model polished_model = model_;
// 	// A flag to determine if the initial model has been updated
// 	bool updated = false;

// 	///
// 	std::vector<size_t> all_inliers(point_number);
// 	for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
// 	{
// 		all_inliers[point_idx] = point_idx;
// 	}

// 	std::vector<gcransac::Model> sum_models;
// 	bool sum_model_valid = false;
// 	bool is_sum_model_updated = false;
// 	ModelScore  sum_score;
// 	std::vector<double> refine_residuals;
// 	int inlier_num;
// 	double local_score1=-1, local_score2=-1, local_score3=-1;

// 	if (estimator_.estimateModelNonminimal(
// 			points_,           // All input points
// 			&(all_inliers)[0], // Points which have higher than 0 probability of being inlier
// 			static_cast<int>(all_inliers.size()), // Number of possible inliers
// 			&sum_models,                          // Estimated models
// 			&(sum_weights_)[0]))                  // Weights of points
// 	{
// 		if (estimator_.isValidModel(sum_models[0], points_, all_inliers, &(all_inliers[0]),
// 									interrupting_threshold, is_sum_model_updated)) {
// 			sum_model_valid = true;

// 			refined_model_ = sum_models[0];

// 			getModelQualityPlusPlus(points_, // All the input points
// 			refined_model_, // The estimated model
// 			estimator_, // The estimator
// 			local_score1, // The marginalized score
// 			inlier_num,
// 			best_score_.score, 0, refine_residuals); // The score of the previous so-far-the-best model

// 			score_.score = local_score1;

// 			// Update the iteration number
// 		last_iteration_number = log_confidence / log(1.0 - std::pow(static_cast<double>(score_.inlier_number) / point_number, sample_size));
// 		// last_iteration_number = 3 * log_confidence / log(1.0 - std::pow(static_cast<double>(inlier_num) / point_number, sample_size));
		
// 		// std::vector<int> resdual_splits;
// 		// printf("inlier_num:%d  score:%f\n", inlier_num, score_.score);
// 		// printf("better: %d \n", best_score_.score<score_.score);

// 		// splitResdauls(refine_residuals, 32, 16, resdual_splits);

// 		}
// 	}

// 	// if (estimator_.estimateModelNonminimal(
// 	// 		points_,           // All input points
// 	// 		&(all_inliers)[0], // Points which have higher than 0 probability of being inlier
// 	// 		static_cast<int>(all_inliers.size()), // Number of possible inliers
// 	// 		&sum_models,                          // Estimated models
// 	// 		&(local_weights_1)[0]))                  // Weights of points
// 	// {
// 	// 	if (estimator_.isValidModel(sum_models[0], points_, all_inliers, &(all_inliers[0]),
// 	// 								interrupting_threshold, is_sum_model_updated)) {
// 	// 		sum_model_valid = true;

// 	// 		getModelQualityPlusPlus(points_, // All the input points
// 	// 		sum_models[0], // The estimated model
// 	// 		estimator_, // The estimator
// 	// 		local_score2, // The marginalized score
// 	// 		inlier_num,
// 	// 		best_score_.score, 0, refine_residuals); // The score of the previous so-far-the-best model

// 	// 		if(local_score2 > score_.score)
// 	// 		{
// 	// 			refined_model_ = sum_models[0];
// 	// 			score_.score = local_score2;
// 	// 		} 
// 	// 		// Update the iteration number
// 	// 	last_iteration_number = log_confidence / log(1.0 - std::pow(static_cast<double>(score_.inlier_number) / point_number, sample_size));
		
// 	// 	// std::vector<int> resdual_splits;
// 	// 	// printf("inlier_num:%d  score:%f\n", inlier_num, score_.score);
// 	// 	// printf("better: %d \n", best_score_.score<score_.score);
// 	// 	}
// 	// }


// 	// 	if (estimator_.estimateModelNonminimal(
// 	// 		points_,           // All input points
// 	// 		&(all_inliers)[0], // Points which have higher than 0 probability of being inlier
// 	// 		static_cast<int>(all_inliers.size()), // Number of possible inliers
// 	// 		&sum_models,                          // Estimated models
// 	// 		&(local_weights_2)[0]))                  // Weights of points
// 	// {
// 	// 	if (estimator_.isValidModel(sum_models[0], points_, all_inliers, &(all_inliers[0]),
// 	// 								interrupting_threshold, is_sum_model_updated)) {
// 	// 		sum_model_valid = true;

// 	// 		getModelQualityPlusPlus(points_, // All the input points
// 	// 		sum_models[0], // The estimated model
// 	// 		estimator_, // The estimator
// 	// 		local_score3, // The marginalized score
// 	// 		inlier_num,
// 	// 		best_score_.score, 0, refine_residuals); // The score of the previous so-far-the-best model

// 	// 		if(local_score3 > score_.score)
// 	// 		{
// 	// 			refined_model_ = sum_models[0];
// 	// 			score_.score = local_score3;
// 	// 		} 
// 	// 		// Update the iteration number
// 	// 	last_iteration_number = log_confidence / log(1.0 - std::pow(static_cast<double>(score_.inlier_number) / point_number, sample_size));
		
// 	// 	// std::vector<int> resdual_splits;
// 	// 	// printf("inlier_num:%d  score:%f\n", inlier_num, score_.score);
// 	// 	// printf("better: %d \n", best_score_.score<score_.score);
// 	// 	}
// 	// }

// 	if(score_.score>0) return true;

// 	return false;
// }

template <class DatumType, class ModelEstimator>
void FASTMAGSAC<DatumType, ModelEstimator>::getModelQualityPlusPlus(
	const cv::Mat &points_, // All data points
	const gcransac::Model &model_, // The model parameter
	const ModelEstimator &estimator_, // The model estimator class
	double &score_, // The score to be calculated
	const double &previous_best_score_) // The score of the previous so-far-the-best model 
{
	// The degrees of freedom of the data from which the model is estimated.
	// E.g., for models coming from point correspondences (x1,y1,x2,y2), it is 4.
	constexpr size_t degrees_of_freedom = ModelEstimator::getDegreesOfFreedom();
	// A 0.99 quantile of the Chi^2-distribution to convert sigma values to residuals
	constexpr double k = ModelEstimator::getSigmaQuantile();
	// A multiplier to convert residual values to sigmas
	constexpr double threshold_to_sigma_multiplier = 1.0 / k;
	// Calculating k^2 / 2 which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	constexpr double squared_k_per_2 = k * k / 2.0;
	// Calculating (DoF - 1) / 2 which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	constexpr double dof_minus_one_per_two = (degrees_of_freedom - 1.0) / 2.0;
	// Calculating (DoF + 1) / 2 which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	constexpr double dof_plus_one_per_two = (degrees_of_freedom + 1.0) / 2.0;
	// TODO: check
	constexpr double C = 0.25;
	// Calculating 2^(DoF - 1) which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	static const double two_ad_dof_minus_one = std::pow(2.0, dof_minus_one_per_two);
	// Calculating 2^(DoF + 1) which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	static const double two_ad_dof_plus_one = std::pow(2.0, dof_plus_one_per_two);
	// Calculate the gamma value of k
	constexpr double gamma_value_of_k = ModelEstimator::getUpperIncompleteGammaOfK();
	// Calculate the lower incomplete gamma value of k
	constexpr double lower_gamma_value_of_k = ModelEstimator::getLowerIncompleteGammaOfK();
	// The number of points provided
	const int point_number = points_.rows;
	// The previous best loss
	const double previous_best_loss = 1.0 / previous_best_score_;
	// Convert the maximum threshold to a sigma value
	const double maximum_sigma = threshold_to_sigma_multiplier * maximum_threshold;
	// Calculate the squared maximum sigma
	const double maximum_sigma_2 = maximum_sigma * maximum_sigma;
	// Calculate \sigma_{max}^2 / 2
	const double maximum_sigma_2_per_2 = maximum_sigma_2 / 2.0;
	// Calculate 2 * \sigma_{max}^2
	const double maximum_sigma_2_times_2 = maximum_sigma_2 * 2.0;
	// Calculate the loss implied by an outlier
	const double outlier_loss = maximum_sigma * two_ad_dof_minus_one  * lower_gamma_value_of_k;
	// Calculating 2^(DoF + 1) / \sigma_{max} which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	const double two_ad_dof_plus_one_per_maximum_sigma = two_ad_dof_plus_one / maximum_sigma;
	// The loss which a point implies
	double loss = 0.0,
		// The total loss regarding the current model
		total_loss = 0.0;

	// Iterate through all points to calculate the implied loss
	for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
	{
		// Calculate the residual of the current point
		const double residual =
			estimator_.residualForScoring(points_.row(point_idx), model_.descriptor);

		// If the residual is smaller than the maximum threshold, consider it outlier
		// and add the loss implied to the total loss.
		if (maximum_threshold < residual)
			loss = outlier_loss;
		else // Otherwise, consider the point inlier, and calculate the implied loss
		{
			// Calculate the squared residual
			const double squared_residual = residual * residual;
			// Divide the residual by the 2 * \sigma^2
			const double squared_residual_per_sigma = squared_residual / maximum_sigma_2_times_2;
			// Get the position of the gamma value in the lookup table
			size_t x = round(precision_of_stored_incomplete_gammas * squared_residual_per_sigma);
			// If the sought gamma value is not stored in the lookup, return the closest element
			if (stored_incomplete_gamma_number < x)
				x = stored_incomplete_gamma_number;

			// Calculate the loss implied by the current point
			loss = maximum_sigma_2_per_2 * stored_lower_incomplete_gamma_values[x] +
				squared_residual / 4.0 * (stored_complete_gamma_values[x] -
					gamma_value_of_k);
			loss = loss * two_ad_dof_plus_one_per_maximum_sigma;
		}

		// Update the total loss
		total_loss += loss;

		// Break the validation if there is no chance of being better than the previous
		// so-far-the-best model.
		if (previous_best_loss < total_loss)
			break;
	}

	// Calculate the score of the model from the total loss
	score_ =  1.0 / total_loss;
}



template <class DatumType, class ModelEstimator>
void FASTMAGSAC<DatumType, ModelEstimator>::getModelQualityPlusPlus(
	const cv::Mat &points_, // All data points
	const gcransac::Model &model_, // The model parameter
	const ModelEstimator &estimator_, // The model estimator class
	double &score_, // The score to be calculated
	int &inlier_number_, //
	const double &previous_best_score_, int type, std::vector<double> &resduals_) // The score of the previous so-far-the-best model 
{
	// The degrees of freedom of the data from which the model is estimated.
	// E.g., for models coming from point correspondences (x1,y1,x2,y2), it is 4.
	constexpr size_t degrees_of_freedom = ModelEstimator::getDegreesOfFreedom();
	// A 0.99 quantile of the Chi^2-distribution to convert sigma values to residuals
	constexpr double k = ModelEstimator::getSigmaQuantile();
	// A multiplier to convert residual values to sigmas
	constexpr double threshold_to_sigma_multiplier = 1.0 / k;
	// Calculating k^2 / 2 which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	constexpr double squared_k_per_2 = k * k / 2.0;
	// Calculating (DoF - 1) / 2 which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	constexpr double dof_minus_one_per_two = (degrees_of_freedom - 1.0) / 2.0;
	// Calculating (DoF + 1) / 2 which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	constexpr double dof_plus_one_per_two = (degrees_of_freedom + 1.0) / 2.0;
	// TODO: check
	constexpr double C = 0.25;
	// Calculating 2^(DoF - 1) which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	static const double two_ad_dof_minus_one = std::pow(2.0, dof_minus_one_per_two);
	// Calculating 2^(DoF + 1) which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	static const double two_ad_dof_plus_one = std::pow(2.0, dof_plus_one_per_two);
	// Calculate the gamma value of k
	constexpr double gamma_value_of_k = ModelEstimator::getUpperIncompleteGammaOfK();
	// Calculate the lower incomplete gamma value of k
	constexpr double lower_gamma_value_of_k = ModelEstimator::getLowerIncompleteGammaOfK();
	// The number of points provided
	const int point_number = points_.rows;
	// The previous best loss
	const double previous_best_loss = 1.0/previous_best_score_;
	// Convert the maximum threshold to a sigma value
	const double maximum_sigma = threshold_to_sigma_multiplier * maximum_threshold;
	// Calculate the squared maximum sigma
	const double maximum_sigma_2 = maximum_sigma * maximum_sigma;
	// Calculate \sigma_{max}^2 / 2
	const double maximum_sigma_2_per_2 = maximum_sigma_2 / 2.0;
	// Calculate 2 * \sigma_{max}^2
	const double maximum_sigma_2_times_2 = maximum_sigma_2 * 2.0;
	// Calculate the loss implied by an outlier
	const double outlier_loss = maximum_sigma * two_ad_dof_minus_one  * lower_gamma_value_of_k;
	// Calculating 2^(DoF + 1) / \sigma_{max} which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	const double two_ad_dof_plus_one_per_maximum_sigma = two_ad_dof_plus_one / maximum_sigma;
	// The loss which a point implies
	double loss = 0.0,
		// The total loss regarding the current model
		total_loss = 0.0;
	// const double interrupting_threshold = sqrt(maximum_threshold)*1.5;
	const double interrupting_threshold = sqrt(maximum_threshold);
	// const double interrupting_threshold = 5.0;

	// Iterate through all points to calculate the implied loss
	inlier_number_ = 0;
	resduals_.resize(point_number);
	for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
	{
		// Calculate the residual of the current point
		const double residual =
			estimator_.residualForScoring(points_.row(point_idx), model_.descriptor);
			// estimator_.squaredResidual(points_.row(point_idx), model_.descriptor);
		resduals_[point_idx] = residual;

		// If the residual is smaller than the maximum threshold, consider it outlier
		// and add the loss implied to the total loss.
		if (maximum_threshold < residual)
		{
			loss = (maximum_threshold)/point_number;
		}
		else // Otherwise, consider the point inlier, and calculate the implied loss
		{
			loss = (residual)/point_number;
			if (residual < interrupting_threshold)
			{
				inlier_number_++;
			}
		}

		// Update the total loss
		total_loss += loss;

		// Break the validation if there is no chance of being better than the previous
		// so-far-the-best model.
		// if (previous_best_loss < total_loss)
		// 	break;
	}

	// Calculate the score of the model from the total loss
	score_ =  1/total_loss;
}

template <class DatumType, class ModelEstimator>
void FASTMAGSAC<DatumType, ModelEstimator>::getModelQuality(
	const cv::Mat &points_, // All data points
	const gcransac::Model &model_, // The model parameter
	const ModelEstimator &estimator_, // The model estimator class
	double &marginalized_iteration_number_, // The marginalized iteration number to be calculated
	double &score_) // The score to be calculated
{
	// Set up the parameters
	constexpr size_t sample_size = estimator_.sampleSize();
	const size_t point_number = points_.rows;

	// Getting the inliers
	std::vector<std::pair<double, size_t>> all_residuals;
	all_residuals.reserve(point_number);

	double max_distance = 0;
	for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
	{
		// Calculate the residual of the current point
		const double residual =
			estimator_.residualForScoring(points_.row(point_idx), model_.descriptor);
		// If the residual is smaller than the maximum threshold, add it to the set of possible inliers
		if (maximum_threshold > residual)
		{
			max_distance = MAX(max_distance, residual);
			all_residuals.emplace_back(std::make_pair(residual, point_idx));
		}
	}

	// Set the maximum distance to be slightly bigger than that of the farthest possible inlier
	max_distance = max_distance +
		std::numeric_limits<double>::epsilon();

	// Number of possible inliers
	const size_t possible_inlier_number = all_residuals.size();

	// The extent of a partition
	const double threshold_step = max_distance / partition_number;

	// The maximum threshold considered in each partition
	std::vector<double> thresholds(partition_number);
	std::vector<double> thresholds_squared(partition_number);
	std::vector<double> thresholds_2_squared(partition_number);

	// Calculating the thresholds for each partition
	for (size_t i = 0; i < partition_number; ++i)
	{
		thresholds[i] = (i + 1) * threshold_step;
		thresholds_squared[i] = thresholds[i] * thresholds[i];
		thresholds_2_squared[i] = 2 * thresholds_squared[i];
	}

	double residual_i, // Residual of the i-th point
		residual_i_squared, // Squared residual of the i-th poin 
		probability_i; // Probability of the i-th point given the model

	std::vector<double> inliers(partition_number, 0), // RANSAC score for each partition
		probabilities(partition_number, 1); // Probabilities for each partition
	for (size_t point_idx = 0; point_idx < possible_inlier_number; ++point_idx)
	{
		residual_i = all_residuals[point_idx].first;
		residual_i_squared = residual_i * residual_i;

		for (size_t i = 0; i < partition_number; ++i)
		{
			if (residual_i < thresholds[i])
			{
				probability_i = 1.0 - residual_i_squared / thresholds_squared[i];
				++inliers[i];
				probabilities[i] += probability_i;
			}
		}
	}

	score_ = 0;
	marginalized_iteration_number_ = 0.0;
	for (auto i = 0; i < partition_number; ++i)
	{
		score_ += probabilities[i];
		marginalized_iteration_number_ += log_confidence / log(1.0 - std::pow(inliers[i] / point_number, sample_size));
	}
	marginalized_iteration_number_ = marginalized_iteration_number_ / partition_number;
}


// get the weight based on residuel
template <class DatumType, class ModelEstimator>
double FASTMAGSAC<DatumType, ModelEstimator>::getWeightFromRes(double residual, double threshold, int type)
{
	switch (type)
	{
	case 0:
		if(residual<threshold)
			return (1/(pow(residual+1, 0.5))) + 0.1;
		else
			return 0;
	case 1:
		if(residual<threshold)
			return (1/((residual+1)));
		else
			return 0;
	case 2:
		if(residual<threshold)
			return 0.8+((exp(-residual*residual/this->maximum_threshold/this->maximum_threshold/2)));
		else
			return 0;
	case 3:
		if(residual<threshold)
		{
			double diff = -residual*residual+threshold*threshold;
			return (abs(diff));
		}
		else
			return 0;
	case 4:
		if(residual<threshold)
		{
			return (1.0);
		}
		else if(residual > this->maximum_threshold)
		{
			return 0;
		}
		else
		{
			return 0;
		}
	case 5:
		if(residual<threshold)
			return 0.8+((exp(-residual/this->maximum_threshold)));
		else
			return 0;
		
	default:
		return 0;
	}
}


template <class DatumType, class ModelEstimator>
float FASTMAGSAC<DatumType, ModelEstimator>::splitResdauls(std::vector<double> residuals_, double maximum_threshold_,
                                                          int split_number_, std::vector<int> &residual_splits_, int &inllier_num_) {
  residual_splits_.clear();
  if (maximum_threshold_ < 0 || split_number_ < 2)
    return;

  double maximum_threshold_step = maximum_threshold_ / split_number_;
  float local_inlier_th = maximum_threshold_;
  residual_splits_.resize(split_number_ + 1);

  // generate histogram of residual
  for (int point_idx = 0; point_idx < residuals_.size(); ++point_idx) {

    int idx = int(residuals_[point_idx] / maximum_threshold_step);
    if ((idx < 0))
      continue;
    if ((idx > split_number_))
      residual_splits_[split_number_];
    else
      residual_splits_[idx]++;
  }

  int cur_max_split_cnt = 0, i;
  inllier_num_ = 0;

  if (0) {
    for (i = 0; i < residual_splits_.size() - 3; i++) {
      printf("%d\t", residual_splits_[i]);

      inllier_num_ += residual_splits_[i];

      cur_max_split_cnt = cur_max_split_cnt > residual_splits_[i] ? cur_max_split_cnt : residual_splits_[i];
      float split_th = cur_max_split_cnt * 0.15;
    //   float split_th = cur_max_split_cnt * 0.08;
      // if (split_th > residual_splits_[i + 1] && split_th > residual_splits_[i + 2]) {
      if (split_th > residual_splits_[i + 1]) {
        local_inlier_th = (i + 1) * maximum_threshold_step;
          printf("||\t");
        break;
      }
    }

	i++;
	for(; i<residual_splits_.size(); i++)
	{
		printf("%d\t", residual_splits_[i]);
	}
	printf("\n");

  }
  else
  {
	  for (i = 0; i < residual_splits_.size() - 3; i++) {
      inllier_num_ += residual_splits_[i];

      cur_max_split_cnt = cur_max_split_cnt > residual_splits_[i] ? cur_max_split_cnt : residual_splits_[i];
      float split_th = cur_max_split_cnt * 0.15;
    //   float split_th = cur_max_split_cnt * 0.08;

    //   if (split_th > residual_splits_[i + 1] && split_th > residual_splits_[i + 2]) {
      if (split_th > residual_splits_[i + 1]) {
        local_inlier_th = (i + 1) * maximum_threshold_step;
        break;
      }
    }
  }
  
  return local_inlier_th;
}



template <class DatumType, class ModelEstimator>
float FASTMAGSAC<DatumType, ModelEstimator>::getWeightFromResiduals(std::vector<double> residuals_, double maximum_threshold_,
                                                          int split_number_, std::vector<int> &residual_splits_, int &inllier_num_,
														  std::vector<double> &weights_) {
  residual_splits_.clear();
  inllier_num_ = 0;
  if (maximum_threshold_ < 0 || split_number_ < 2)
    return 0;

  double maximum_threshold_step = maximum_threshold_ / split_number_;
  float local_inlier_th = maximum_threshold_;
  residual_splits_.resize(split_number_ + 1);
  std::vector<double> density_residuals(split_number_ + 1);

  // generate histogram of residual
  for (int point_idx = 0; point_idx < residuals_.size(); ++point_idx) {

    int idx = int(residuals_[point_idx] / maximum_threshold_step);

    if (idx < 0) continue;

    if (idx <= split_number_)
	{
      residual_splits_[idx]++;
	  inllier_num_++;
	}
  }

  int last_nonzero = -1;	
  for(int i=0; i<density_residuals.size(); i++)
  {
	  if(residual_splits_[i]>0)
	  {
		//   density_residuals[i] = residual_splits_[i]*1.0/(i-last_nonzero);
		  density_residuals[i] = residual_splits_[i]*1.0;
		  last_nonzero = i;
		//   printf("%d:%0.4f\t", residual_splits_[i], density_residuals[i]);
	  }
  }
//   printf("\n");


  // get the for histogram of residual
  for (int point_idx = 0; point_idx < residuals_.size(); ++point_idx) {

    int idx = int(residuals_[point_idx] / maximum_threshold_step);

    if (idx < 0 || idx > split_number_)
	{
		weights_[point_idx] = 0.0;
	} 
	else
	{
		weights_[point_idx] = (sqrt(density_residuals[idx]));
	}
  }
  
  return local_inlier_th;
}



/*
template <class DatumType, class ModelEstimator>
float FASTMAGSAC<DatumType, ModelEstimator>::splitResdauls(std::vector<double> residuals_, double maximum_threshold_,
                                                          int split_number_, std::vector<int> &residual_splits_, int &inllier_num_) {
  residual_splits_.clear();
  if (maximum_threshold_ < 0 || split_number_ < 2)
    return;

  double maximum_threshold_step = maximum_threshold_ / split_number_;
  float local_inlier_th = maximum_threshold_;
  residual_splits_.resize(split_number_ + 1);

  for (int point_idx = 0; point_idx < residuals_.size(); ++point_idx) {

    int idx = int(residuals_[point_idx] / maximum_threshold_step);
    if ((idx < 0))
      continue;
    if ((idx > split_number_))
      residual_splits_[split_number_];
    else
      residual_splits_[idx]++;
  }

  int cur_max_split_cnt = 0, i;
  inllier_num_ = 0;

  if (0) {
    for (i = 0; i < residual_splits_.size() - 3; i++) {
      printf("%d\t", residual_splits_[i]);

      inllier_num_ += residual_splits_[i];

      cur_max_split_cnt = cur_max_split_cnt > residual_splits_[i] ? cur_max_split_cnt : residual_splits_[i];
      // float split_th = cur_max_split_cnt * 0.15;
      float split_th = cur_max_split_cnt * 0.15;
      // if (split_th > residual_splits_[i + 1] && split_th > residual_splits_[i + 2]) {
      if (split_th > residual_splits_[i + 1]) {
        local_inlier_th = (i + 1) * maximum_threshold_step;
          printf("||\t");
        break;
      }
    }

	i++;
	for(; i<residual_splits_.size(); i++)
	{
		printf("%d\t", residual_splits_[i]);
	}
	printf("\n");

  }
  else
  {
	  for (i = 0; i < residual_splits_.size() - 3; i++) {
      inllier_num_ += residual_splits_[i];

      cur_max_split_cnt = cur_max_split_cnt > residual_splits_[i] ? cur_max_split_cnt : residual_splits_[i];
      float split_th = cur_max_split_cnt * 0.15;
    //   float split_th = cur_max_split_cnt * 0.08;

      if (split_th > residual_splits_[i + 1] && split_th > residual_splits_[i + 2]) {
    //   if (split_th > residual_splits_[i + 1]) {
        local_inlier_th = (i + 1) * maximum_threshold_step;
        break;
      }
    }
  }
  
  return local_inlier_th;
}
*/
