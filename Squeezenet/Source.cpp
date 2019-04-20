#include <CL/cl.hpp>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>


using namespace cv;
using namespace std;

#define row 512
#define col 512
int main()
{


	cv::Mat inputImageRaw, b, g, r;
	const char* inputFilename = "cat.jpg";
	inputImageRaw = cv::imread(inputFilename);
	//cout << "M = " << endl << " " << inputImageRaw << endl << endl;
	if (!inputImageRaw.data)
	{
		std::cout << "could not find the file";

		exit(-1);
	}
	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", inputImageRaw);                // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window

	cv::Mat rgbchannel[3];
	// Splits the image in to RGB Channels
	cv::split(inputImageRaw, rgbchannel);
	//Opens the weight file for conv1 layer
	std::fstream myfile("conv1.txt", std::ios_base::in);
	float a;
	std::vector<float> Filter_conv1;
	Filter_conv1.resize(64 * 3 * 3 * 3);
	int i = 0;
	while (myfile >> a)
	{
		Filter_conv1[i] = a;
		//std::cout << Filter_conv1[i] << ' ';
		i++;
	}
	//import image to an array
	//printf("array %f \n", Filter_conv1);
	unsigned int j;
	std::vector<float> image_data;

	image_data.resize(3 * row * col);
	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < row * col; j++)
		{
			image_data[i * row * col + j] = rgbchannel[i].data[j];
		}
	}
	//OpenCL code starts from here
	try {


		// Get available platforms
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		// Select the default platform and create a context using this platform and the GPU
		cl_context_properties cps[3] = {
			CL_CONTEXT_PLATFORM,
			(cl_context_properties)(platforms[0])(),
			0
		};
		cl::Context context(CL_DEVICE_TYPE_GPU, cps);

		// Get a list of devices on this platform
		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

		// Create a command queue and use the first device
		cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);

		// Read source file
		std::ifstream sourceFile("kernel.cl");
		std::string sourceCode(
			std::istreambuf_iterator<char>(sourceFile),
			(std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));

		// Make program of the source code in the context
		cl::Program program = cl::Program(context, source);

		// Build program for these specific devices
		program.build(devices);

		// Make kernel
		cl::Kernel kernel(program, "conv1");
		//float out_size_first_layer = ;
		// Create memory buffers
		cl::Buffer image = cl::Buffer(context, CL_MEM_READ_ONLY, row * col * 3 * sizeof(float));
		cl::Buffer Filter_conv = cl::Buffer(context, CL_MEM_READ_ONLY, 64 * 3 * 3 * 3 * sizeof(float));
		cl::Buffer output_first_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 255 * 255 * 3 * 64 * sizeof(float));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(image, CL_TRUE, 0, row * col * 3 * sizeof(float), image_data.data());
		queue.enqueueWriteBuffer(Filter_conv, CL_TRUE, 0, 64 * 3 * 3 * 3 * sizeof(float), Filter_conv1.data());
		unsigned int input_channel, input_size, stride, output_size;

		input_channel = 3;
		input_size = 512;
		stride = 2;
		output_size = 255;

		kernel.setArg(0, image);
		kernel.setArg(1, Filter_conv);
		kernel.setArg(2, output_first_layer);
		kernel.setArg(3, input_channel);
		kernel.setArg(4, input_size);
		kernel.setArg(5, stride);
		kernel.setArg(6, output_size);

		cl::NDRange global(64);
		cl::NDRange local(1);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
		std::vector<float> output;
		output.resize(255 * 255 * 3 * 64);
		queue.enqueueReadBuffer(output_first_layer, CL_TRUE, 0, ((255 * 255 * 3 * 64) * sizeof(float)), output.data());
		printf("output at host:%f", output[0]);



		////////////////////////////////// Pooling layer //////////////////////////////////////////////////////////////////////////

			// Make kernel
		cl::Kernel kernelpool(program, "pool");
		//float out_size_second_layer = ;
		// Create memory buffers
		cl::Buffer output_second_layer = cl::Buffer(context, CL_MEM_WRITE_ONLY, 127 * 127 * 3 * 64 * sizeof(float));

		input_size = 255;
		output_size = 127;

		kernelpool.setArg(0, output_first_layer);
		kernelpool.setArg(1, output_second_layer);
		kernelpool.setArg(2, input_size);
		kernelpool.setArg(3, output_size);

		cl::NDRange global1(64);
		cl::NDRange local1(1);
		queue.enqueueNDRangeKernel(kernelpool, cl::NullRange, global1, local1);
		std::vector<float> output2;
		output2.resize(127 * 127 * 3 * 64);
		queue.enqueueReadBuffer(output_second_layer, CL_TRUE, 0, ((127 * 127 * 3 * 64) * sizeof(float)), output2.data());
		printf("output at host:%f", output2[0]);


	}
	catch (std::runtime_error error) {
		std::cout << error.what() << std::endl;

	}
	getchar();
	// Get available platforms
	return 0;
}