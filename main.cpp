#include <CL/cl.h>
#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <chrono>

const unsigned int SIZE = 2000000;
const int treadsNum = 4;
size_t localSize_gpu = 256;
size_t localSize_cpu = 256;

const char* saxpy_gpu = "__kernel void sa_gpu (              \n"
	"const int n,                                            \n"
	"float a,                                                \n"
	"__global float * x,                                     \n"
	"const int incx,                                         \n"
	"__global float * y,                                     \n"
	"const int incy                                          \n"
	") {                                                     \n"
	"int ind = get_global_id(0);                             \n"
	"if (ind < n)                                            \n"
	"    y[ind * incy] = y[ind * incy] + a * x[ind * incx];  \n"
"}";

const char* daxpy_gpu = "__kernel void da_gpu (              \n"
	"const int n,                                            \n"
	"double a,                                               \n"
	"__global double * x,                                    \n"
	"const int incx,                                         \n"
	"__global double * y,                                    \n"
	"const int incy                                          \n"
	") {                                                     \n"
	"int ind = get_global_id(0);                             \n"
	"if (ind < n)                                            \n"
	"    y[ind * incy] = y[ind * incy] + a * x[ind * incx];  \n"
"}";

void saxpy(int n, float a, float* x, int incx, float* y, int incy)
{
	for(int i = 0; i < n; i++)
	{
		y[i * incy] = y[i * incy] + a * x[i * incx];
	}
}

void daxpy(int n, double a, double* x, int incx, double* y, int incy)
{
	for(int i = 0; i < n; i++)
	{
		y[i * incy] = y[i * incy] + a * x[i * incx];
	}
}

void saxpy_omp(int n, float a, float* x, int incx, float* y, int incy)
{
	#pragma omp parallel for
	for(int i = 0; i < n; i++)
	{
		y[i * incy] = y[i * incy] + a * x[i * incx];
	}
}

void daxpy_omp(int n, double a, double* x, int incx, double* y, int incy)
{
	#pragma omp parallel for 
	for(int i = 0; i < n; i++)
	{
		y[i * incy] = y[i * incy] + a * x[i * incx];
	}
}

template <typename T>
bool checkResults(T* first, T* second, int n)
{
	bool check = false;
	double eps = 5e-3;
	for(int i = 0; i < n; i++)
	{
		if(fabs(first[i] - second[i]) < eps)
		{
			check = true;
		}
		else
		{
			check = false;
		}
	}
	return check;
}

int main ()
{
	std::cout << "SIZE = " << SIZE << std::endl;
	std::cout << "Thread num = " << treadsNum << std::endl;
	// -------------------------- ÑÐÀÂÍÅÍÈÅ ÐÅÇÓËÜÒÀÒÎÂ ÍÀ GPU VS CPU FLOAT --------------------------
	cl_int error = 0;

	cl_uint numPlatforms = 0;
	clGetPlatformIDs(0, NULL, &numPlatforms); 
	cl_platform_id platform = NULL;
	//cl_platform_id platform2 = NULL;

	if (0 < numPlatforms)
	{
		cl_platform_id* platforms = new cl_platform_id [numPlatforms];
		clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[1];
		//platform2 = platforms[0];

		char platform_name[128];
		//char platform_name2[128];
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platform_name, nullptr);
		//clGetPlatformInfo(platform2, CL_PLATFORM_NAME, 128, platform_name2, nullptr);
 		std::cout << "platform 1 = " << platform_name << std::endl;
		//std::cout << "platform 2 = " << platform_name2 << std::endl;

		delete[] platforms;
	}

	cl_context_properties properties [3] = { 
		CL_CONTEXT_PLATFORM, ( cl_context_properties ) platform, 0 };

	cl_context context = clCreateContextFromType (
		( NULL == platform ) ? NULL : properties,
		CL_DEVICE_TYPE_GPU,
		NULL,
		NULL,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create context from type failed: " << error << std::endl;
	}

	size_t size = 0;

	clGetContextInfo (
		context,
		CL_CONTEXT_DEVICES,
		0,
		NULL,
		&size );

	cl_device_id device;

	if (size > 0)
	{
		cl_device_id * devices = ( cl_device_id * ) alloca ( size );
		clGetContextInfo (
			context,
			CL_CONTEXT_DEVICES,
			size,
			devices,
			NULL );
		device = devices[0];
		
		char device_name[128];
		clGetDeviceInfo(device, CL_DEVICE_NAME, 128, device_name, nullptr);
		std::cout << device_name << std::endl;
	}

	cl_command_queue queue = clCreateCommandQueue(
		context,		
		device,
		CL_QUEUE_PROFILING_ENABLE,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create command queue with properties failed: " << error << std::endl;
	}

	size_t srclen[] = { strlen(saxpy_gpu) };

	cl_program program = clCreateProgramWithSource(
		context,
		1,
		&saxpy_gpu,
		srclen,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create program failed: " << error << std::endl;
	}

	error = clBuildProgram(
		program,
		1,
		&device,
		NULL,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Build program failed: " << error << std::endl;
	}

	cl_kernel kernel = clCreateKernel(program,
		"sa_gpu",
		&error);

	if (error != CL_SUCCESS) {
		std::cout << "Create kernel failed: " << error << std::endl;
	}

	size_t n = SIZE;
	size_t group = 0;

	float* data_x = (float*)_aligned_malloc(sizeof(float)*SIZE, 64); //[SIZE];
	float* data_y = (float*)_aligned_malloc(sizeof(float)*SIZE, 64); //[SIZE];
	float* data_x_cpu = (float*)_aligned_malloc(sizeof(float)*SIZE, 64); //[SIZE];
	float* data_y_cpu = (float*)_aligned_malloc(sizeof(float)*SIZE, 64); //[SIZE];
	float* data_x_fomp = (float*)_aligned_malloc(sizeof(float)*SIZE, 64); //[SIZE];
	float* data_y_fomp = (float*)_aligned_malloc(sizeof(float)*SIZE, 64); //[SIZE];
	float* data_x_fcpu = (float*)_aligned_malloc(sizeof(float)*SIZE, 64); //[SIZE];
	float* data_y_fcpu = (float*)_aligned_malloc(sizeof(float)*SIZE, 64); //[SIZE];
	float* results = (float*)_aligned_malloc(sizeof(float)*SIZE, 64); //[SIZE];
	float* results_fcpu = (float*)_aligned_malloc(sizeof(float)*SIZE, 64); //[SIZE];
	float a = (float)rand() / RAND_MAX;
	int incx = 1;
	int incy = 1;

	for (int i = 0; i < SIZE; i++) {
		float tmp_x = (float)rand() / RAND_MAX;
		data_x[i] = tmp_x;
		data_x_cpu[i] = tmp_x;
		data_x_fomp[i] = tmp_x;
		data_x_fcpu[i] = tmp_x;
		float tmp_y = (float)rand() / RAND_MAX;
		data_y[i] = tmp_y;
		data_y_cpu[i] = tmp_y;
		data_y_fomp[i] = tmp_y;
		data_y_fcpu[i] = tmp_y;
	}
	std::cout << "------------------ SAXPY ------------------" << std::endl;
	if (SIZE <= 10) {	
		for (int i = 0; i < SIZE; i++) {
			std::cout << "data_x[" << i << "] = " << data_x[i] << "\tdata_y[" << i << "] = " << data_y[i] << std::endl;
		}
		std::cout << "a = " << a << std::endl;
		std::cout << "incx = " << incx << std::endl;
		std::cout << "incy = " << incy << std::endl;
	}

	cl_mem x = clCreateBuffer (
		context,
		CL_MEM_READ_ONLY,
		sizeof(float) * SIZE,
		NULL,
		NULL);

	cl_mem y = clCreateBuffer (
		context,
		CL_MEM_READ_WRITE,
		sizeof(float) * SIZE,
		NULL,
		NULL);

	error = clEnqueueWriteBuffer (
		queue,
		x,
		CL_TRUE,
		0,
		sizeof(float) * SIZE,
		data_x,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_x failed: " << error << std::endl;
	}

	error = clEnqueueWriteBuffer (
		queue,
		y,
		CL_TRUE,
		0,
		sizeof(float) * SIZE,
		data_y,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_x failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel,
		0,
		sizeof(int),
		&n);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for n failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel,
		1,
		sizeof(float),
		&a);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for a failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel,
		2,
		sizeof(cl_mem),
		&x);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for x failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel,
		3,
		sizeof(int),
		&incx);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for incx failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel,
		4,
		sizeof(cl_mem),
		&y);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for y failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel,
		5,
		sizeof(int),
		&incy);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for incy failed: " << error << std::endl;
	}

	clGetKernelWorkGroupInfo (
		kernel,
		device,
		CL_KERNEL_WORK_GROUP_SIZE,
		sizeof (size_t),
		&group,
		NULL );

	cl_event evt;
	auto start_fgpu = std::chrono::steady_clock::now();

	error = clEnqueueNDRangeKernel (
		queue,
		kernel,
		1,
		NULL,
		&n,
		&localSize_gpu,
		0,
		NULL,
		&evt );

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue failed: " << error << std::endl;
	}

	clWaitForEvents(1, &evt);
	auto finish_fgpu = std::chrono::steady_clock::now();

	cl_ulong start = 0, end = 0;
    error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
    if (error != CL_SUCCESS) {
		std::cout << "Error getting start time: " << error <<std::endl;
    }
    error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
    if (error != CL_SUCCESS) {
		std::cout << "Error getting end time: " << error <<std::endl;
    }

	clEnqueueReadBuffer (
		queue,
		y,
		CL_TRUE,
		0,
		sizeof(float) * n,
		results,
		0,
		NULL,
		NULL);

	clock_t start_fcpu = clock();
	saxpy(n, a, data_x_cpu, 1, data_y_cpu, 1);
	clock_t finish_fcpu = clock();

	omp_set_num_threads(treadsNum);
	double start_fomp = omp_get_wtime();
	saxpy_omp(n, a, data_x_fomp, 1, data_y_fomp, 1);
	double finish_fomp = omp_get_wtime();

	if (SIZE <= 10) {
		std::cout << "n = " << n << ", a = " << a << ", incx = " << incx << ", incy = " << incy << std::endl;
		std::cout << "CPU RESULTS" << std::endl;
		for (int i = 0; i < n; i++) {
			std::cout << "result[" << i << "] = " << data_y_cpu[i] << std::endl;
		}
		std::cout << "OPENCL RESULTS" << std::endl;
		for (int i = 0; i < SIZE; i++) {
			std::cout << "result[" << i << "] = " << results[i] << std::endl;
		}
		std::cout << "OMP FLOAT RESULTS" << std::endl;
		for (int i = 0; i < n; i++) {
			std::cout << "result[" << i << "] = " << data_y_fomp[i] << std::endl;
		}
	}

	// -------------------------- Ñðàâíåíèå ðåçóëüòàòîâ CPU & GPU float --------------------------
	if(checkResults<float>(results, data_y_cpu, n))
	{
		std::cout << "Results (CPU & GPU) are equal." << std::endl;
	}
	else
	{
		std::cout << "Results (CPU & GPU) are different." << std::endl;
	}
	// -------------------------- Ñðàâíåíèå ðåçóëüòàòîâ CPU & OMP float --------------------------
	if(checkResults<float>(results, data_y_fomp, n))
	{
		std::cout << "Results (CPU & OMP) are equal." << std::endl;
	}
	else
	{
		std::cout << "Results (CPU & OMP) are different." << std::endl;
	}

	std::cout << "OMP time = " << (finish_fomp - start_fomp)*(1e+03) << "ms\t" << "GPU time = " 
		<< (cl_double)(end - start)*(cl_double)(1e-06) << "ms\t"  << "CPU time = " << (float)(finish_fcpu - start_fcpu) << "ms" << std::endl;

	// -------------------------- Îñâîáîæäåíèå ðåñóðñîâ --------------------------
	clReleaseMemObject(x);
	clReleaseMemObject(y);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);






	// -------------------------- ÑÐÀÂÍÅÍÈÅ ÐÅÇÓËÜÒÀÒÎÂ GPU VS CPU DOUBLE --------------------------
	error = 0;

	cl_context_properties properties_d [3] = { 
		CL_CONTEXT_PLATFORM, ( cl_context_properties ) platform, 0 };

	cl_context context_d = clCreateContextFromType (
		( NULL == platform ) ? NULL : properties_d,
		CL_DEVICE_TYPE_GPU,
		NULL,
		NULL,
		&error);

	if (error != CL_SUCCESS) {
		std::cout << "Create context from type failed: " << error << std::endl;
	}

	size = 0;

	clGetContextInfo (
		context_d,
		CL_CONTEXT_DEVICES,
		0,
		NULL,
		&size );

	cl_device_id device_d;

	if (size > 0)
	{
		cl_device_id * devices = ( cl_device_id * ) alloca ( size );
		clGetContextInfo (
			context_d,
			CL_CONTEXT_DEVICES,
			size,
			devices,
			NULL );
		device_d = devices[0];
		
		char device_name[128];
		clGetDeviceInfo(device_d, CL_DEVICE_NAME, 128, device_name, nullptr);
		std::cout << device_name << std::endl;
	}

	cl_command_queue queue_d = clCreateCommandQueue(
		context_d,		
		device_d,
		CL_QUEUE_PROFILING_ENABLE,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create command queue with properties failed: " << error << std::endl;
	}

	size_t srclen_d[] = { strlen(daxpy_gpu) };

	cl_program program_d = clCreateProgramWithSource(
		context_d,
		1,
		&daxpy_gpu,
		srclen_d,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create program failed: " << error << std::endl;
	}

	error = clBuildProgram(
		program_d,
		1,
		&device_d,
		NULL,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Build program failed: " << error << std::endl;
	}

	cl_kernel kernel_d = clCreateKernel(program_d,
		"da_gpu",
		&error);

	if (error != CL_SUCCESS) {
		std::cout << "Create kernel failed: " << error << std::endl;
	}

	n = SIZE;
	group = 0;

	double* data_x_d = (double*)_aligned_malloc(sizeof(double)*SIZE, 64); //[SIZE];
	double* data_y_d = (double*)_aligned_malloc(sizeof(double)*SIZE, 64); //[SIZE];
	double* data_x_cpu_d = (double*)_aligned_malloc(sizeof(double)*SIZE, 64); //[SIZE];
	double* data_y_cpu_d = (double*)_aligned_malloc(sizeof(double)*SIZE, 64); //[SIZE];
	double* data_x_domp = (double*)_aligned_malloc(sizeof(double)*SIZE, 64); //[SIZE];
	double* data_y_domp = (double*)_aligned_malloc(sizeof(double)*SIZE, 64); //[SIZE];
	double* results_d = (double*)_aligned_malloc(sizeof(double)*SIZE, 64); //[SIZE];
	double* data_x_dcpu = (double*)_aligned_malloc(sizeof(double)*SIZE, 64); //[SIZE];
	double* data_y_dcpu = (double*)_aligned_malloc(sizeof(double)*SIZE, 64); //[SIZE];
	double* results_dcpu = (double*)_aligned_malloc(sizeof(double)*SIZE, 64); //[SIZE];
	double a_d = (double)rand() / RAND_MAX;
	int incx_d = 1;
	int incy_d = 1;

	for (int i = 0; i < SIZE; i++) {
		double tmp_x = (double)rand() / RAND_MAX;
		data_x_d[i] = tmp_x;
		data_x_cpu_d[i] = tmp_x;
		data_x_domp[i] = tmp_x;
		double tmp_y = (double)rand() / RAND_MAX;
		data_y_d[i] = tmp_y;
		data_y_cpu_d[i] = tmp_y;
		data_y_domp[i] = tmp_y;
	}
	std::cout << "------------------ DAXPY ------------------" << std::endl;
		if (SIZE <= 10) {
		for (int i = 0; i < SIZE; i++) {
			std::cout << "data_x[" << i << "] = " << data_x_d[i] << "\tdata_y[" << i << "] = " << data_y_d[i] << std::endl;
		}
		std::cout << "a = " << a_d << std::endl;
		std::cout << "incx = " << incx_d << std::endl;
		std::cout << "incy = " << incy_d << std::endl;
	}

	cl_mem x_d = clCreateBuffer (
		context_d,
		CL_MEM_READ_ONLY,
		sizeof(double) * SIZE,
		NULL,
		NULL);

	cl_mem y_d = clCreateBuffer (
		context_d,
		CL_MEM_READ_WRITE,
		sizeof(double) * SIZE,
		NULL,
		NULL);

	error = clEnqueueWriteBuffer (
		queue_d,
		x_d,
		CL_TRUE,
		0,
		sizeof(double) * SIZE,
		data_x_d,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_x failed: " << error << std::endl;
	}

	error = clEnqueueWriteBuffer (
		queue_d,
		y_d,
		CL_TRUE,
		0,
		sizeof(double) * SIZE,
		data_y_d,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_x failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel_d,
		0,
		sizeof(int),
		&n);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for n failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel_d,
		1,
		sizeof(double),
		&a_d);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for a failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel_d,
		2,
		sizeof(cl_mem),
		&x_d);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for x failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel_d,
		3,
		sizeof(int),
		&incx_d);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for incx failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel_d,
		4,
		sizeof(cl_mem),
		&y_d);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for y failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel_d,
		5,
		sizeof(int),
		&incy_d);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for incy failed: " << error << std::endl;
	}

	clGetKernelWorkGroupInfo (
		kernel_d,
		device_d,
		CL_KERNEL_WORK_GROUP_SIZE,
		sizeof (size_t),
		&group,
		NULL );

	cl_event evt1;
	auto start_dgpu = std::chrono::steady_clock::now();

	error = clEnqueueNDRangeKernel (
		queue_d,
		kernel_d,
		1,
		NULL,
		&n,
		NULL,
		0,
		NULL,
		&evt1 );

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue failed: " << error << std::endl;
	}

	clWaitForEvents(1, &evt1);
	auto finish_dgpu = std::chrono::steady_clock::now();

	clEnqueueReadBuffer (
		queue_d,
		y_d,
		CL_TRUE,
		0,
		sizeof(double) * n,
		results_d,
		0,
		NULL,
		NULL);

	start = 0; end = 0;
    error = clGetEventProfilingInfo(evt1, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
    if (error != CL_SUCCESS) {
		std::cout << "Error getting start time: " << error <<std::endl;
    }
    error = clGetEventProfilingInfo(evt1, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
    if (error != CL_SUCCESS) {
		std::cout << "Error getting end time: " << error <<std::endl;
    }

	clock_t start_dcpu = clock();
	daxpy(n, a_d, data_x_cpu_d, 1, data_y_cpu_d, 1);
	clock_t finish_dcpu = clock();

	omp_set_num_threads(treadsNum);
	double start_domp = omp_get_wtime();
	daxpy_omp(n, a_d, data_x_domp, 1, data_y_domp, 1);
	double finish_domp = omp_get_wtime();

	if (SIZE <= 10) {
		std::cout << "n = " << n << ", a = " << a_d << ", incx = " << incx_d << ", incy = " << incy_d << std::endl;
		std::cout << "CPU RESULTS" << std::endl;
		for (int i = 0; i < n; i++) {
			std::cout << "result[" << i << "] = " << data_y_cpu_d[i] << std::endl;
		}
		std::cout << "OPENCL RESULTS" << std::endl;
		for (int i = 0; i < SIZE; i++) {
			std::cout << "result[" << i << "] = " << results_d[i] << std::endl;
		}
		std::cout << "OMP DOUBLE RESULTS" << std::endl;
		for (int i = 0; i < n; i++) {
			std::cout << "result[" << i << "] = " << data_y_domp[i] << std::endl;
		}
	}

	// -------------------------- Ñðàâíåíèå ðåçóëüòàòîâ (CPU & GPU) double --------------------------
	if(checkResults<double>(results_d, data_y_cpu_d, n))
	{
		std::cout << "Results (CPU & GPU) are equal." << std::endl;
	}
	else
	{
		std::cout << "Results (CPU & GPU) are different." << std::endl;
	}
	// -------------------------- Ñðàâíåíèå ðåçóëüòàòîâ (CPU & OMP) double --------------------------
	if(checkResults<double>(results_d, data_y_domp, n))
	{
		std::cout << "Results (CPU & OMP) are equal." << std::endl;
	}
	else
	{
		std::cout << "Results (CPU & OMP) are different." << std::endl;
	}
	std::cout << "OMP time = " << (finish_domp - start_domp)*(1e+03) << "ms\t" << "GPU time = " 
		<< (cl_double)(end - start)*(cl_double)(1e-06) << "ms\t" << "CPU time = " << (float)(finish_dcpu - start_dcpu) << "ms" << std::endl;

	clReleaseMemObject(x_d);
	clReleaseMemObject(y_d);
	clReleaseProgram(program_d);
	clReleaseKernel(kernel_d);
	clReleaseCommandQueue(queue_d);
	clReleaseContext(context_d);

	// ---------------------------------------- for CPU float ----------------------------------------

	cl_context context_fcpu = clCreateContextFromType (
		( NULL == platform ) ? NULL : properties,
		CL_DEVICE_TYPE_CPU,
		NULL,
		NULL,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create context from type failed: " << error << std::endl;
	}

	clGetContextInfo (
		context_fcpu,
		CL_CONTEXT_DEVICES,
		0,
		NULL,
		&size );

	cl_device_id device_fcpu;

	if (size > 0)
	{
		cl_device_id * devices = ( cl_device_id * ) alloca ( size );
		clGetContextInfo (
			context_fcpu,
			CL_CONTEXT_DEVICES,
			size,
			devices,
			NULL );
		device_fcpu = devices[0];
		
		char device_name[128];
		clGetDeviceInfo(device_fcpu, CL_DEVICE_NAME, 128, device_name, nullptr);
		std::cout << device_name << std::endl;
	}

	cl_command_queue queue_fcpu = clCreateCommandQueue(
		context_fcpu,		
		device_fcpu,
		CL_QUEUE_PROFILING_ENABLE,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create command queue with properties failed: " << error << std::endl;
	}

	size_t srclen_fcpu[] = { strlen(saxpy_gpu) };

	cl_program program_fcpu = clCreateProgramWithSource(
		context_fcpu,
		1,
		&saxpy_gpu,
		srclen_fcpu,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create program failed: " << error << std::endl;
	}

	error = clBuildProgram(
		program_fcpu,
		1,
		&device_fcpu,
		NULL,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Build program failed: " << error << std::endl;
	}

	cl_kernel kernel_fcpu = clCreateKernel(program_fcpu,
		"sa_gpu",
		&error);

	if (error != CL_SUCCESS) {
		std::cout << "Create kernel failed: " << error << std::endl;
	}

	group = 0;

	cl_mem x_fcpu = clCreateBuffer (
		context_fcpu,
		CL_MEM_READ_ONLY,
		sizeof(float) * SIZE,
		NULL,
		NULL);

	cl_mem y_fcpu = clCreateBuffer (
		context_fcpu,
		CL_MEM_READ_WRITE,
		sizeof(float) * SIZE,
		NULL,
		NULL);

	error = clEnqueueWriteBuffer (
		queue_fcpu,
		x_fcpu,
		CL_TRUE,
		0,
		sizeof(float) * SIZE,
		data_x_fcpu,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_x failed: " << error << std::endl;
	}

	error = clEnqueueWriteBuffer (
		queue_fcpu,
		y_fcpu,
		CL_TRUE,
		0,
		sizeof(float) * SIZE,
		data_y_fcpu,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_x failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel_fcpu,
		0,
		sizeof(int),
		&n);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for n failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel_fcpu,
		1,
		sizeof(float),
		&a);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for a failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel_fcpu,
		2,
		sizeof(cl_mem),
		&x_fcpu);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for x failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel_fcpu,
		3,
		sizeof(int),
		&incx);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for incx failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel_fcpu,
		4,
		sizeof(cl_mem),
		&y_fcpu);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for y failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel_fcpu,
		5,
		sizeof(int),
		&incy);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for incy failed: " << error << std::endl;
	}

	clGetKernelWorkGroupInfo (
		kernel_fcpu,
		device_fcpu,
		CL_KERNEL_WORK_GROUP_SIZE,
		sizeof (size_t),
		&group,
		NULL );

	cl_event evt_fcpu;

	error = clEnqueueNDRangeKernel (
		queue_fcpu,
		kernel_fcpu,
		1,
		NULL,
		&n,
		&localSize_cpu,
		0,
		NULL,
		&evt_fcpu );

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue failed: " << error << std::endl;
	}

	clWaitForEvents(1, &evt_fcpu);

	cl_ulong start2 = 0, end2 = 0;
    error = clGetEventProfilingInfo(evt_fcpu, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start2, nullptr);
    if (error != CL_SUCCESS) {
		std::cout << "Error getting start time: " << error <<std::endl;
    }
    error = clGetEventProfilingInfo(evt_fcpu, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end2, nullptr);
    if (error != CL_SUCCESS) {
		std::cout << "Error getting end time: " << error <<std::endl;
    }

	clEnqueueReadBuffer (
		queue_fcpu,
		y_fcpu,
		CL_TRUE,
		0,
		sizeof(float) * n,
		results_fcpu,
		0,
		NULL,
		NULL);

	std::cout << "CPU OpenCL time for SAXPY = " << (cl_double)(end2 - start2)*(cl_double)(1e-06) << "ms" << std::endl;

	// ---------------------------------------- for CPU double ----------------------------------------

	cl_context context_dcpu = clCreateContextFromType (
		( NULL == platform ) ? NULL : properties,
		CL_DEVICE_TYPE_CPU,
		NULL,
		NULL,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create context from type failed: " << error << std::endl;
	}

	clGetContextInfo (
		context_dcpu,
		CL_CONTEXT_DEVICES,
		0,
		NULL,
		&size );

	cl_device_id device_dcpu;

	if (size > 0)
	{
		cl_device_id * devices = ( cl_device_id * ) alloca ( size );
		clGetContextInfo (
			context_fcpu,
			CL_CONTEXT_DEVICES,
			size,
			devices,
			NULL );
		device_dcpu = devices[0];
		
		char device_name[128];
		clGetDeviceInfo(device_dcpu, CL_DEVICE_NAME, 128, device_name, nullptr);
		std::cout << device_name << std::endl;
	}

	cl_command_queue queue_dcpu = clCreateCommandQueue(
		context_dcpu,		
		device_dcpu,
		CL_QUEUE_PROFILING_ENABLE,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create command queue with properties failed: " << error << std::endl;
	}

	size_t srclen_dcpu[] = { strlen(daxpy_gpu) };

	cl_program program_dcpu = clCreateProgramWithSource(
		context_dcpu,
		1,
		&daxpy_gpu,
		srclen_dcpu,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create program failed: " << error << std::endl;
	}

	error = clBuildProgram(
		program_dcpu,
		1,
		&device_dcpu,
		NULL,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Build program failed: " << error << std::endl;
	}

	cl_kernel kernel_dcpu = clCreateKernel(program_dcpu,
		"da_gpu",
		&error);

	if (error != CL_SUCCESS) {
		std::cout << "Create kernel failed: " << error << std::endl;
	}

	group = 0;

	cl_mem x_dcpu = clCreateBuffer (
		context_dcpu,
		CL_MEM_READ_ONLY,
		sizeof(double) * SIZE,
		NULL,
		NULL);

	cl_mem y_dcpu = clCreateBuffer (
		context_dcpu,
		CL_MEM_READ_WRITE,
		sizeof(double) * SIZE,
		NULL,
		NULL);

	error = clEnqueueWriteBuffer (
		queue_dcpu,
		x_dcpu,
		CL_TRUE,
		0,
		sizeof(double) * SIZE,
		data_x_dcpu,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_x failed: " << error << std::endl;
	}

	error = clEnqueueWriteBuffer (
		queue_dcpu,
		y_dcpu,
		CL_TRUE,
		0,
		sizeof(double) * SIZE,
		data_y_dcpu,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_x failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel_dcpu,
		0,
		sizeof(int),
		&n);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for n failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel_dcpu,
		1,
		sizeof(double),
		&a_d);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for a failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel_dcpu,
		2,
		sizeof(cl_mem),
		&x_dcpu);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for x failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel_dcpu,
		3,
		sizeof(int),
		&incx);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for incx failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel_dcpu,
		4,
		sizeof(cl_mem),
		&y_dcpu);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for y failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel_dcpu,
		5,
		sizeof(int),
		&incy);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for incy failed: " << error << std::endl;
	}

	clGetKernelWorkGroupInfo (
		kernel_dcpu,
		device_dcpu,
		CL_KERNEL_WORK_GROUP_SIZE,
		sizeof (size_t),
		&group,
		NULL );

	cl_event evt_dcpu;

	error = clEnqueueNDRangeKernel (
		queue_dcpu,
		kernel_dcpu,
		1,
		NULL,
		&n,
		NULL,
		0,
		NULL,
		&evt_dcpu );

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue failed: " << error << std::endl;
	}

	clWaitForEvents(1, &evt_dcpu);

	cl_ulong start3 = 0, end3 = 0;
    error = clGetEventProfilingInfo(evt_dcpu, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start3, nullptr);
    if (error != CL_SUCCESS) {
		std::cout << "Error getting start time: " << error <<std::endl;
    }
    error = clGetEventProfilingInfo(evt_dcpu, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end3, nullptr);
    if (error != CL_SUCCESS) {
		std::cout << "Error getting end time: " << error <<std::endl;
    }

	clEnqueueReadBuffer (
		queue_dcpu,
		y_dcpu,
		CL_TRUE,
		0,
		sizeof(double) * n,
		results_dcpu,
		0,
		NULL,
		NULL);

	std::cout << "CPU OpenCL time for DAXPY = " << (cl_double)(end3 - start3)*(cl_double)(1e-06) << "ms" << std::endl;


	_aligned_free(data_x);
	_aligned_free(data_y);
	_aligned_free(data_x_cpu);
	_aligned_free(data_y_cpu);
	_aligned_free(data_x_fomp);
	_aligned_free(data_y_fomp);
	_aligned_free(results);

	_aligned_free(data_x_d);
	_aligned_free(data_y_d);
	_aligned_free(data_x_cpu_d);
	_aligned_free(data_y_cpu_d);
	_aligned_free(data_x_domp);
	_aligned_free(data_y_domp);
	_aligned_free(results_d);

	return 0;
}