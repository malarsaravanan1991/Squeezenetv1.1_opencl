#include <CL/cl.hpp>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <time.h>

using namespace cv;
using namespace std;

#define row 512
#define col 512

/*Program to read the kernel code */
long LoadOpenCLKernel(char const* path, char** buf) {
	FILE* fp;
	size_t fsz;
	long off_end;
	int rc;

	/* Open the file */
	fp = fopen(path, "r");
	if (NULL == fp) {
		return -1L;
	}

	/* Seek to the end of the file */
	rc = fseek(fp, 0L, SEEK_END);
	if (0 != rc) {
		return -1L;
	}

	/* Byte offset to the end of the file (size) */
	if (0 > (off_end = ftell(fp))) {
		return -1L;
	}
	fsz = (size_t)off_end;

	/* Allocate a buffer to hold the whole file */
	*buf = (char*)malloc(fsz + 1);
	if (NULL == *buf) {
		return -1L;
	}

	/* Rewind file pointer to start of file */
	rewind(fp);

	/* Slurp file into buffer */
	if (fsz != fread(*buf, 1, fsz, fp)) {
		free(*buf);
		return -1L;
	}

	/* Close the file */
	if (EOF == fclose(fp)) {
		free(*buf);
		return -1L;
	}
	/* Make sure the buffer is NUL-terminated, just in case */
	(*buf)[fsz] = '\0';
	/* Return the file size */
	return (long)fsz;
}


int decode_image(char frame[64*1*16], char filename[]) {
	FILE* pFile;
	pFile = fopen(filename, "r");
	if (pFile == NULL) {
		fprintf(stderr, "Could not open %s\n", filename);
		return -1;
	}
	fseek(pFile, 15, SEEK_SET);
	fread(frame, sizeof(char), 64 * 1 * 16, pFile);
	fclose(pFile);
	return 0;
}


int main()
{

	//std::string binaryFile = "./xclbin/cnn.sw_emu.xilinx_aws-vu9p-f1-04261818_dynamic_5_0.xclbin";

	cv::Mat inputImageRaw, b, g, r;
	const char* inputFilename = "cat.jpg";
	inputImageRaw = cv::imread(inputFilename);
	//cout << "M = " << endl << " " << inputImageRaw << endl << endl;
	if (!inputImageRaw.data)
	{
		std::cout << "could not find the file";

		exit(-1);
	}

	cv::Mat rgbchannel[3];
	// Splits the image in to RGB Channels
	cv::split(inputImageRaw, rgbchannel);
	//Opens the weight file for conv1 layer
	//***********************************Getting  weights of the filter in to the array*******************//
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

	//***********************************Getting  weights of the filter in to the array*******************//
	std::fstream fire2squeeze1x1("fire2squeeze1x1.txt", std::ios_base::in);
	float s;

	std::vector<float> Filter_conv_fire1;
	Filter_conv_fire1.resize(64 * 1 * 16);
	int z = 0;
	while (fire2squeeze1x1 >> s)
	{
		Filter_conv_fire1[z] = s;
		
		z++;
	}
	std::cout << Filter_conv_fire1[0] << ' ';
	
	//***********************************Getting  weights of the filter in to the array*******************//
	std::fstream fire2expand1x1("fire2expand1x1.txt", std::ios_base::in);
	float sf;

	std::vector<float> Filter_convexpand_fire1;
	Filter_convexpand_fire1.resize(64 * 1 * 16);
	int zf = 0;
	while (fire2expand1x1 >> sf)
	{
		Filter_convexpand_fire1[zf] = sf;

		zf++;
	}
	std::cout << Filter_convexpand_fire1[1] << ' ';

	
	//import image to an array
	
	unsigned int j;
	std::vector<float> image_data;

	image_data.resize(3 * row * col);
	int k = 0;
	for (i = 2; i >= 0; i--)
	{
		for (j = 0; j < row * col; j++)
		{
			image_data[k * row * col + j] = rgbchannel[i].data[j];

		}
		k++;
	}
	//namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("Display window", image_data);                // Show our image inside it.
	//waitKey(0); // Wait for a keystroke in the window

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
		//cl_command_queue queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);

		//checkError(status, "Failed to create command queue");
		// Read source file
		std::ifstream sourceFile("kernel.cl");
		std::string sourceCode(
			std::istreambuf_iterator<char>(sourceFile),
			(std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
		/*
		cl_int err;
		char* KernelBinary;
		const char* binaryFile = "./xclbin/cnn.sw_emu.xilinx_aws-vu9p-f1-04261818_dynamic_5_0.xclbin";
		size_t fileBufSize = LoadOpenCLKernel(binaryFile, &KernelBinary);
		cl::Program::Binaries bins{{KernelBinary, fileBufSize} };
		cl::Program program(context, devices, bins, NULL, &err);
		if (fileBufSize < 0L) {
			perror("File read failed");
			return 1;
		}*/
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
		cl::Buffer output_first_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 255 * 255 * 1 * 64 * sizeof(float));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(image, CL_TRUE, 0, row * col * 3 * sizeof(float), image_data.data());
		queue.enqueueWriteBuffer(Filter_conv, CL_TRUE, 0, 64 * 3 * 3 * 3 * sizeof(float), Filter_conv1.data());
		unsigned int input_channel, input_size, stride, output_size, start_channel_fire;

		input_channel = 3;
		input_size = 512;
		stride = 2;
		output_size = 255;
		start_channel_fire = 0;

		kernel.setArg(0, image);
		kernel.setArg(1, Filter_conv);
		kernel.setArg(2, output_first_layer);
		kernel.setArg(3, input_channel);
		kernel.setArg(4, input_size);
		kernel.setArg(5, stride);
		kernel.setArg(6, output_size);
		kernel.setArg(7, start_channel_fire);

		cl::NDRange global(64);
		cl::NDRange local(1);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

		std::vector<float> output;
		output.resize(255 * 255 * 1 * 64);
		queue.enqueueReadBuffer(output_first_layer, CL_TRUE, 0, ((255 * 255 * 1 * 64) * sizeof(float)), output.data());
		printf("output at host:%f \n", output[0]);



		////////////////////////////////// Pooling layer //////////////////////////////////////////////////////////////////////////

			// Make kernel
		cl::Kernel kernelpool(program, "pool");
		//float out_size_second_layer = ;
		// Create memory buffers
		cl::Buffer output_second_layer = cl::Buffer(context, CL_MEM_WRITE_ONLY, 127 * 127 * 1 * 64 * sizeof(float));

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
		output2.resize(127 * 127 * 1 * 64);
		queue.enqueueReadBuffer(output_second_layer, CL_TRUE, 0, ((127 * 127 * 1 * 64) * sizeof(float)), output2.data());
		printf("output at pooling layer:%f \n", output2[0]);

		/***********************************************Fire block 1 - squeeze conv 1x1 ****************************************************/
		cl::Kernel kernelconv1x1fire1(program, "conv2");

		// Create memory buffers

		cl::Buffer Filter_conv_fire1_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 64 * 1 * 16 * sizeof(float));
		cl::Buffer output_fire1_conv1x1_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 127 * 127 * 1 * 16 * sizeof(float));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_conv_fire1_block1, CL_TRUE, 0, 64 * 1 * 16 * sizeof(float), Filter_conv_fire1.data());


		input_channel = 64;
		input_size = 127;

		kernelconv1x1fire1.setArg(0, output_second_layer);
		kernelconv1x1fire1.setArg(1, Filter_conv_fire1_block1);
		kernelconv1x1fire1.setArg(2, output_fire1_conv1x1_layer);
		kernelconv1x1fire1.setArg(3, input_channel);
		kernelconv1x1fire1.setArg(4, input_size);


		cl::NDRange global2(16);
		cl::NDRange local2(1);
		queue.enqueueNDRangeKernel(kernelconv1x1fire1, cl::NullRange, global2, local2);

		std::vector<float> output3;
		output3.resize(127 * 127 * 1 * 16);
		queue.enqueueReadBuffer(output_fire1_conv1x1_layer, CL_TRUE, 0, ((127 * 127 * 1 * 16) * sizeof(float)), output3.data());
		printf("output at third squeezelayer:%f \n", output3[0]);
		
		/***********************************************Fire block 1 - expand conv 1x1 ****************************************************/
		cl::Kernel kernelconv1x1expandfire1(program, "conv3");
		// Create memory buffers
		cl::Buffer Filter_convexpand_fire1_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 64 * 1 * 16 * sizeof(float));
		cl::Buffer output_fire1_conv1x1expand_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 127 * 127 * 1 * 128 * sizeof(float));
		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_convexpand_fire1_block1, CL_TRUE, 0, 64 * 1 * 16 * sizeof(float), Filter_convexpand_fire1.data());
		
		input_channel = 16;
		input_size = 127;

		kernelconv1x1expandfire1.setArg(0, output_fire1_conv1x1_layer);
		kernelconv1x1expandfire1.setArg(1, Filter_convexpand_fire1_block1);
		kernelconv1x1expandfire1.setArg(2, output_fire1_conv1x1expand_layer);
		kernelconv1x1expandfire1.setArg(3, input_channel);
		kernelconv1x1expandfire1.setArg(4, input_size);


		cl::NDRange global3(64);
		cl::NDRange local3(1);
		queue.enqueueNDRangeKernel(kernelconv1x1expandfire1, cl::NullRange, global3, local3);

		std::vector<float> output4;
		output4.resize(127 * 127 * 1 * 128);
		queue.enqueueReadBuffer(output_fire1_conv1x1expand_layer, CL_TRUE, 0, ((127 * 127 * 1 * 128) * sizeof(float)), output4.data());
		printf("output at expand fire:%f \n", output4[255]);
		
		//cl_int err = queue.finish();
	}
	catch (std::runtime_error error) {
		std::cout << error.what() << std::endl;

	}
	// Get available platforms
	return 0;
}