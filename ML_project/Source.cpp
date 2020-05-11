#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include <cmath>

extern "C" {
	DLLEXPORT int my_add(int x, int y) {
		return x + y;
	}
	DLLEXPORT int my_mul(int x, int y) {
		return x * y;
	}

	DLLEXPORT double*  linear_model_create(int input_dim) {
		return 0;
		double f = (double)rand() / RAND_MAX;
	}

	DLLEXPORT double  linear_model_predict_regression(double* model, double *inputs, int inputs_size) {
		return 0;
	}

	DLLEXPORT double  linear_model_predict_classification(double* model, double *inputs, int inputs_size) {
		return linear_model_predict_regression(model, inputs, inputs_size) >= 0 ? 1.0 : -1.0;
	}

	DLLEXPORT double  linear_model_train_classification(double* model, 
		double *dataset_inputs, int dataset_length , int inputs_size,
		double *dataset_expected_outputs, int outputs_size) {
		return 0;
	}

	DLLEXPORT void  linear_model_delete(double* model) {

	}
}