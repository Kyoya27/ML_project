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

using Eigen::MatrixXd;

extern "C" {

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

    DLLEXPORT void linear_model_train_regression(double* model, double* dataset_inputs, int dataset_length, int inputs_size, double* dataset_expected_outputs, int outputs_size, int iterations_count, float alpha) {
       //training poarams number = input size
        MatrixXd xm(dataset_length, inputs_size + 1);
        MatrixXd ym(dataset_length, 1);

        for (int i = 0; i < dataset_length; ++i) {
            ym(i, 0) = dataset_expected_outputs[i];
            xm(i, 0) = 1;
            for (int j = 1; j < (inputs_size + 1); j++) {
                xm(i, j) = dataset_inputs[(i * inputs_size + (j - 1))];
            }
        }

        MatrixXd res = ((xm.transpose() * xm).inverse() * xm.transpose()) * ym;

        for (int i = 0; i < inputs_size; i++) {
            model[i] = res(i, 0);
        }
    }
    DLLEXPORT void clearArray(double* array) {
        free(array);
    }
//    DLLEXPORT void linear_model_train_regression(double* model, double** dataset_inputs, int dataset_length, int inputs_size, double dataset_expected_outputs, int outputs_size, int iterations_count, float alpha) {
//        Eigen::MatrixXd eMatrix(dataset_length, dataset_length);
//        for (int i = 0; i < dataset_length; ++i)
//            eMatrix.row(i) = Eigen::VectorXd::Map(&dataset_inputs[i][0], dataset_length);
//        Eigen::MatrixXd pinv = eMatrix.completeOrthogonalDecomposition().pseudoInverse();
//        //Eigen::Matrix2d v = Eigen::Map<Eigen::MatrixXd>(model, 1, inputs_size + 1);
//        Eigen::VectorXd v(model, inputs_size);
//        //v = Eigen::Product(v, pinv);
//
////        v = v * 1;
//        
//  //      for (int i = 0; i <= inputs_size; i++) {
//  //          model[i] = v(0, i);
//    //    }
//
//        // std::cout << pinv;
//    }
}