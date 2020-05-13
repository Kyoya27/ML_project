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
        std::uniform_real_distribution<float> distribution{ -1, 1 };

        MLP* mlp = new MLP;
        mlp->npl = npl;
        mlp->npl_size = npl_size;

        double*** w1 = new double**[npl_size];
        for (int l = 0; l < npl_size - 1; l++) {
            w1[l] = new double* [npl[l] + 1];
            for (int i = 0; i < npl[l] + 1; i++) {
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

        for (int l = 1; l < mlp->npl_size - 1; l++) {
            nodes[l] = new double[mlp->npl[l] + 1];
            nodes[l][0] = 1;
            for (int i = 0; i < mlp->npl[l]; i++) {
                double sum = 0;
                for (int j = 0; j < mlp->npl[l - 1] + 1; j++) {
                    sum += nodes[l - 1][j] * mlp->w[l - 1][j][i];
                }
                nodes[l][i + 1] = tanh(sum);
            }
        }

        nodes[mlp->npl_size - 1] = new double[mlp->npl[mlp->npl_size - 1]];
        for (int i = 0; i < mlp->npl[mlp->npl_size - 1]; i++) {
            double sum = 0;
            for (int j = 0; j < mlp->npl[mlp->npl_size - 2] + 1; j++) {
                sum += nodes[mlp->npl_size - 2][j] * mlp->w[mlp->npl_size - 2][j][i];
            }
            nodes[mlp->npl_size - 1][i] = tanh(sum);
        }

        mlp->x = nodes;
    }

    DLLEXPORT double mlp_model_predict_regression(MLP* mlp) {
        double somme = 0;
        for (int i = 0; i < mlp->npl[mlp->npl_size - 1]; i++) {

            somme += mlp->x[mlp->npl_size - 1][i] * mlp->w[mlp->npl_size - 1][i][0];
        }
        std::cout << somme << std::endl;
        return somme;
    }

    DLLEXPORT double mlp_model_predict_classification(MLP* mlp, double* inputs) {
        generate_nodes(mlp, inputs);
        return mlp_model_predict_regression(mlp) >= 0 ? 1.0 : -1.0;
    }

    DLLEXPORT void mlp_model_train_classification(MLP* mlp, double* dataset_inputs, int dataset_length, int inputs_size, double* dataset_expected_outputs, int outputs_size, double alpha) {
        //deltas dernière
        std::default_random_engine randomEngine(std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<float> distribution{ 0, 1 };
        auto trainingPosition = (int)floor(distribution(randomEngine) * 1) * (mlp->npl[0] + 1);
        generate_nodes(mlp, &(dataset_inputs[trainingPosition]));

        double** deltas = new double* [mlp->npl_size];
        deltas[mlp->npl_size - 1] = new double[mlp->npl_size - 1];

        for (int j = 0; j < mlp->npl[mlp->npl_size - 1]; j++) {
            deltas[mlp->npl_size - 1][j] = (1 - pow(mlp->x[mlp->npl_size - 1][j], 2)) * (mlp->x[mlp->npl_size - 1][j] - dataset_expected_outputs[j]);
        }

        for (int l = mlp->npl_size - 2; l >= 0; l--) {
            deltas[l] = new double[mlp->npl[l] + 1];
            for (int i = 0; i < mlp->npl[l] + 1; i++) {
                double somme = 0;
                for (int j = 0; j < mlp->npl[l + 1]; j++) {
                    somme += deltas[l + 1][j] * mlp->w[l][i][j];
                }
                deltas[l][i] = (1 - pow(mlp->x[l][i], 2)) * somme;
            }
        }

        for (int l = 0; l < mlp->npl_size -1; l++) {
            for (int i = 0; i < mlp->npl[l]; i++) {
                for (int j = 0; j < mlp->npl[l + 1]; j++) {
                    mlp->w[l][i][j] = mlp->w[l][i][j] - (alpha * mlp->x[l][i] * deltas[l+1][j]);
                }
            }
        }

        std::cout << "Delta :" << std::endl << "[" << std::endl;
        for (int l = 0; l < mlp->npl_size; l++) {
            std::cout << "\t[ ";
            for (int i = 0; i < mlp->npl[l]; i++) {
                std::cout << deltas[l][i] << " ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;


        mlp->deltas = deltas;
    }

    DLLEXPORT void mlp_model_train_regression(MLP* mlp, double* dataset_inputs, int dataset_length, int inputs_size, double* dataset_expected_outputs, int outputs_size, double alpha) {

        generate_nodes(mlp, dataset_inputs);

        //deltas dernière
        const int L = mlp->npl[mlp->npl_size - 1];
        double** deltas = new double* [mlp->npl_size];
        deltas[mlp->npl_size - 1] = new double[L + 1];

        for (int j = 0; j < L; j++) {
            deltas[mlp->npl_size - 1][j] = mlp->x[L][j] - dataset_expected_outputs[j];
        }

        for (int l = L - 1; l >= 0; l--) {
            deltas[l] = new double[mlp->npl[l] + 1];
            for (int i = 0; i < mlp->npl[l] + 1; i++) {
                double somme = 0;
                for (int j = 0; j < mlp->npl[l + 1] + 1; j++) { // peut etre pas + 1 ?
                    somme += deltas[l + 1][j] * mlp->w[l + 1][i][j];
                }
                deltas[l][i] = (1 - pow(mlp->x[l][i], 2)) * somme;
            }
        }

        for (int l = 1; l < mlp->npl_size; l++) {
            for (int i = 0; i < mlp->npl[l] + 1; i++) {
                for (int j = 0; j < mlp->npl[l + 1] + 1; j++) {
                    mlp->w[l][i][j] = mlp->w[l][i][j] - (alpha * mlp->x[l - 1][i] * deltas[l][j]);
                }
            }
        }


        mlp->deltas = deltas;
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

        for (int i = 0; i < inputs_size; i++) {
            res += model[i+1] * inputs[i];
        }

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

            int pos = k * inputs_size;

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