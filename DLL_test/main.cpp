#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include <iostream>
#include <time.h>

typedef struct MLP {
	int* npl;
	int npl_size;
	double*** w;
	double** x;
	double** deltas;
} MLP;

extern "C" {
	DLLEXPORT double* linear_model_create(int input_dim);
	DLLEXPORT double linear_model_predict_classification(double* model, double* inputs, int inputs_size);
	DLLEXPORT void linear_model_train_classification(
		double* model,
		double* dataset_inputs,
		int dataset_length,
		int inputs_size,
		double* dataset_expected_outputs,
		int outputs_size,
		int iterations_count,
		float alpha
	);
	DLLEXPORT void linear_model_train_regression(
		double* model,
		double* dataset_inputs,
		int dataset_length,
		int inputs_size,
		double* dataset_expected_outputs,
		int outputs_size
	);

	DLLEXPORT MLP* create_mlp_model(int* npl, int npl_size);
	DLLEXPORT void generate_nodes(MLP* mlp, double* inputs);
	DLLEXPORT double* mlp_model_predict_regression(MLP* mlp, double* inputs, bool isReg);
	DLLEXPORT double* mlp_model_predict_classification(MLP* mlp, double* inputs, bool isReg);
	DLLEXPORT void mlp_model_train_classification(MLP* mlp, double* dataset_inputs, int dataset_length, int inputs_size, double* dataset_expected_outputs, int outputs_size, int epoch, double alpha);
	DLLEXPORT void mlp_model_train_regression(MLP* mlp, double* dataset_inputs, int dataset_length, int inputs_size, double* dataset_expected_outputs, int outputs_size, int epoch, double alpha);
}

int main() {

	double blue_points[1][2] = {
		{0.35, 0.5}
	};

	double red_points[2][2] = {
		{0.6, 0.6},
		{0.55, 0.7}
	};

	double X[3][3] = {
		{1, 0.35, 0.5},
		{1, 0.6, 0.6},
		{1, 0.55, 0.7}
	};

	//double inputs[3][2]{
	//	{0.35, 0.5 },
	//	{0.6, 0.6 },
	//	{0.55, 0.7 }
	//};

	//double** inputs = new double*[3];
	//for (int i = 0; i < 3; i++) {
	//	inputs[i] = new double[2];
	//}
	//inputs[0][0] = 0.35;
	//inputs[0][1] = 0.5;
	//inputs[1][0] = 0.6;
	//inputs[1][1] = 0.6;
	//inputs[2][0] = 0.55;
	//inputs[2][1] = 0.7;


	


	

	/*double* model = linear_model_create(2);
	//double* model = new double[3];
	//model[0] = 0.519944;
	//model[1] = 0.00601215;
	//model[2] = 0.226081;
	for (int i = 0; i < 3; i++) {
		std::cout << "model =" << model[i] << " " << std::endl;
	}
	std::cout << linear_model_predict_classification(model, &(inputs[0]), 2);
	std::cout << linear_model_predict_classification(model, &(inputs[2]), 2);
	std::cout << linear_model_predict_classification(model, &(inputs[4]), 2);

	//linear_model_train_classification(model, inputs, 3, 2, Y, 3, 1000000, 0.01);
	linear_model_train_regression(model, inputs, 3, 2, Y, 3);

	std::cout << linear_model_predict_classification(model, &(inputs[0]), 2);
	std::cout << linear_model_predict_classification(model, &(inputs[2]), 2);
	std::cout << linear_model_predict_classification(model, &(inputs[4]), 2);
	std::cout << std::endl;

	double inParams[] = { 3.14, 1.12 };
	double expected[] = { 5.67 };
	for (int i = 0; i < 3; i++) {
		std::cout << model[i] << " ";
	}*/

	double* inputs = new double[8];
	inputs[0] = -1;
	inputs[1] = -1;
	inputs[2] = 1;
	inputs[3] = 1;
	inputs[4] = 1;
	inputs[5] = -1;
	inputs[6] = -1;
	inputs[7] = 1;

	int npl[] = { 2, 3, 1 };
	double Y[4] = { -1, -1, 1, 1};

	MLP* model = create_mlp_model(npl, 3);
	
	// MLP Classification
	//mlp_model_train_classification(model, inputs, 4, 2, Y, 4, 1000000, 0.001, false);
	// MLP Regression
	mlp_model_train_classification(model, inputs, 4, 2, Y, 4, 1000000, 0.001);
	std::cout << std::endl;
	std::cout << mlp_model_predict_classification(model, &(inputs[0]), false)[1] << std::endl;
	std::cout << mlp_model_predict_classification(model, &(inputs[2]), false)[1] << std::endl;
	std::cout << mlp_model_predict_classification(model, &(inputs[4]), false)[1] << std::endl;
	std::cout << mlp_model_predict_classification(model, &(inputs[6]), false)[1] << std::endl;

	return 0;
}