#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include <cmath>
#include <time.h>
#include <iostream>
#include <ctime>
#include <chrono>
#include <random>
#include <Eigen/Dense>
#include "source.h"

using Eigen::MatrixXd;

extern "C" {
    // ([2, 3, 4, 5], 4)
    DLLEXPORT MLP* create_mlp_model(int* npl, int npl_size) {
        std::default_random_engine randomEngine(std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<float> distribution{ 0, 1 };

        MLP* mlp = new MLP;
        mlp->npl = npl;
        mlp->npl_size = npl_size;


        double*** w1 = new double**[npl_size];
        for (int l = 0; l < npl_size - 1; l++) {
            w1[l] = new double* [npl[l] + 1];
            for (int i = 0; i < npl[l]; i++) {
                w1[l][i] = new double[npl[l + 1]];
                for (int j = 0; j < npl[l + 1]; j++) {
                    w1[l][i][j] = distribution(randomEngine);
                }
            }
        }

        w1[npl_size - 1] = new double*[npl[npl_size - 1]];
        for (int i = 0; i < npl[npl_size - 1]; i++) {
            w1[npl_size - 1][i] = new double(distribution(randomEngine));
        }

        mlp->w = w1;

        return mlp;
    }

    DLLEXPORT void generate_nodes(MLP* mlp, double* inputs) {
        double** nodes = new double* [mlp->npl_size];

        nodes[0] = new double[mlp->npl[0] + 1];
        nodes[0][0] = 1;
        for (int i = 0; i < mlp->npl[0]; i++) {
            nodes[0][i + 1] = inputs[i];
        }

        for (int l = 1; l < mlp->npl_size; l++) {
            nodes[l] = new double[mlp->npl[l] + 1];
            nodes[l][0] = 1;
            for (int i = 0; i < mlp->npl[l]; i++) {
                //nodes[l][i + 1] = mlp->w[l][i]
                double sum = 0;
                for (int j = 0; j < mlp->npl[l - 1]; j++) {
                    double w =  mlp->w[l - 1][i][j] * nodes[l - 1][j];
                    sum += w;
                }

                nodes[l][i + 1] = tanh(sum);
            }
        }

        mlp->x = nodes;
    }

    DLLEXPORT void mlp_model_train_classification(MLP* model, double* dataset_inputs, int dataset_length, int inputs_size, double* dataset_expected_outputs, int outputs_size, int epoch, double alpha) {

    }














    DLLEXPORT double* linear_model_create(int input_dim) {
        std::srand(std::time(nullptr));

        auto array = new double[input_dim + 1];
        
        std::default_random_engine randomEngine(std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<float> distribution{ -1, 1 };

        for (int i = 0; i < input_dim + 1; i++) {
           // auto rd = (double)((std::rand() % RAND_MAX) / (double)RAND_MAX);
            auto rd = distribution(randomEngine);
            array[i] = rd;
        }

        return array;
    }

    DLLEXPORT double linear_model_predict_regression(double* model, double* inputs, int inputs_size) {
        double res = 0;
        //std::cout << "input = " << *inputs << std::endl;
        for (int i = 0; i < inputs_size; i++) {
            res += model[i+1] * inputs[i];
        }

//        std::cout << "res = " << res << " model " << model[0] << " return = " << res + model[0] << std::endl;
        return res + model[0];
    }

    DLLEXPORT double linear_model_predict_classification(double* model, double* inputs, int inputs_size) {
        return linear_model_predict_regression(model, inputs, inputs_size) >= 0 ? 1.0 : -1.0;
    }

    DLLEXPORT void linear_model_train_classification(double* model, double* dataset_inputs, int dataset_length, int inputs_size, double* dataset_expected_outputs, int outputs_size, int iterations_count, float alpha) {
        //rosen
        std::srand(std::time(nullptr));
        for (int i = 0; i < iterations_count; i++) {
            //srand(time(NULL));
            
            int k = floor(std::rand() % dataset_length);
            //std::cout << "picked = " << k << std::endl;
            int pos = k * inputs_size;
            //std::cout << "pos = " << pos << std::endl;
            double g_x_k = linear_model_predict_classification(model, &dataset_inputs[pos], inputs_size);
            double grad = alpha * (dataset_expected_outputs[k] - g_x_k);
            model[0] += grad;
            for (int j = 0; j < inputs_size; j++) {
                model[j + 1] += grad * dataset_inputs[pos + j];//(&dataset_inputs)[k][j];
            }
        }
    }

    DLLEXPORT void linear_model_train_regression(double* model, double* dataset_inputs, int dataset_length, int inputs_size, double* dataset_expected_outputs, int outputs_size) {
        //double adataset_inputs[6] = { 1,1,2,1,3,1 };

        MatrixXd xm(dataset_length, inputs_size + 1);
        MatrixXd ym(dataset_length, 1);

        for (int i = 0; i < dataset_length; ++i) {
            ym(i, 0) = dataset_expected_outputs[i];
            xm(i, 0) = 1;
            for (int j = 1; j < (inputs_size + 1); j++) {
                xm(i, j) = dataset_inputs[(i * inputs_size + (j - 1))];
            }
        }

        auto transpose = xm.transpose();
        auto xxm = transpose * xm;
        auto inverse = xxm.inverse();
        auto t2 = xm.transpose();
        auto big = inverse * t2;
        auto res = big * ym;

        //MatrixXd res = ((xm.transpose() * xm).inverse() * xm.transpose()) * ym;

        for (int i = 0; i < inputs_size + 1; i++) {
            model[i] = res(i, 0);
        }
    }

    DLLEXPORT void clearArray(double* array) {
        free(array);
    }
}