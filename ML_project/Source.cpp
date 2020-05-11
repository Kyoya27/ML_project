#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include <cmath>
#include <time.h>
#include <iostream>
#include <Eigen/Core>

using Eigen::Matrix2d;


extern "C" {
    DLLEXPORT double* linear_model_create(int input_dim) {
        srand(time(NULL));
        double* array = new double[input_dim];
        
        for (int i = 0; i < input_dim; i++) {
            double rd = (double)(rand() / RAND_MAX);
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

    DLLEXPORT void linear_model_train_classification(double* model, double** dataset_inputs, int dataset_length, int inputs_size, double* dataset_expected_outputs, int outputs_size, int iterations_count, float alpha) {
        //rosen
        
        for (int i = 0; i < iterations_count; i++) {
            srand(time(NULL));
            int k = rand() % inputs_size;
            double g_x_k = linear_model_predict_classification(model, dataset_inputs[k], inputs_size);
            double grad = alpha * (dataset_expected_outputs[k] - g_x_k);
            model[0] += grad * 1;
            for (int j = 0; j < inputs_size; j++) {
                model[j + 1] += grad * dataset_inputs[k][j];
            }
        }
    }

    DLLEXPORT void linear_model_train_regression(double* model, double** dataset_inputs, int dataset_length, int inputs_size, double dataset_expected_outputs, int outputs_size, int iterations_count, float alpha) {
        //PseudoInverse
        double b0 = 0;
        double b1 = 0;

        for (int i = 0; i < iterations_count * inputs_size; i++) {
            int idx = i % inputs_size;
            double p = b0 + b1 * dataset_inputs[idx][0];
            double err = p - dataset_inputs[idx][1];
            b0 = b0 - alpha * err;
            b1 = b1 - alpha * err * dataset_inputs[idx][0];
            model[0] = b1;
            model[1] = b0;
            model[2] = err;
        }
    }
}