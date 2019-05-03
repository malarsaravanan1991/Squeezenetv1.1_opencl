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

#define row 224
#define col 224

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


int decode_image(char frame[64 * 1 * 16], char filename[]) {
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
	const char* inputFilename = "duck.jpg";
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
	int a;

	std::vector<int> Filter_conv1;
	Filter_conv1.resize(64 * 3 * 3 * 3);
	int i = 0;
	while (myfile >> a)
	{
		Filter_conv1[i] = a;
		//std::cout << Filter_conv1[i] << ' ';
		i++;
	}
	

	//***********************************Getting  weights of the filter in to the array for conv 1x1 squeezelayer *******************//
	std::fstream fire1squeeze1x1("fire2squeeze1x1.txt", std::ios_base::in);
	int s;

	std::vector<int> Filter_conv_fire1;
	Filter_conv_fire1.resize(64 * 1 * 16);
	int z = 0;
	while (fire1squeeze1x1 >> s)
	{
		Filter_conv_fire1[z] = s;

		z++;
	}
	std::cout << Filter_conv_fire1[0] << ' ';

	//***********************************Getting  weights of the filter in to the array for expand 1x1 layer*******************//
	std::fstream fire1expand1x1("fire2expand1x1.txt", std::ios_base::in);
	int sf;

	std::vector<int> Filter_convexpand_fire1;
	Filter_convexpand_fire1.resize(64 * 1 * 16);
	int zf = 0;
	while (fire1expand1x1 >> sf)
	{
		Filter_convexpand_fire1[zf] = sf;

		zf++;
	}
	std::cout << Filter_convexpand_fire1[0] << ' ';
	//***********************************Getting  weights of the filter in to the array*******************//
	std::fstream fire1expand3x3("fire2expand3x3.txt", std::ios_base::in);
	int sff;

	std::vector<int> Filter_convexpand3x3_fire1;
	Filter_convexpand3x3_fire1.resize(64 * 9 * 16);
	int zff = 0;
	while (fire1expand3x3 >> sff)
	{
		Filter_convexpand3x3_fire1[zff] = sff;

		zff++;
	}
	std::cout << Filter_convexpand3x3_fire1[0] << ' ';

	/***********************************Getting  weights of the filter in to the array fire2squeeze1x1*******************/
	std::fstream fire2squeeze1x1("fire3_squeeze1x1.txt", std::ios_base::in);
	int store;

	std::vector<int> Filter_convsqueeze_fire2;
	Filter_convsqueeze_fire2.resize(128 * 1 * 16);
	int dummy = 0;
	while (fire2squeeze1x1 >> store)
	{
		Filter_convsqueeze_fire2[dummy] = store;

		dummy++;
	}
	std::cout << Filter_convsqueeze_fire2[0] << ' ';

	/***********************************Getting  weights of the filter in to the array fire2expand1x1*******************/
	std::fstream fire2expand1x1("fire3_expand1x1.txt", std::ios_base::in);
	int var;

	std::vector<int> Filter_convexpand1x1_fire2;
	Filter_convexpand1x1_fire2.resize(64 * 1 * 16);
	int no = 0;
	while (fire2expand1x1 >> var)
	{
		Filter_convexpand1x1_fire2[no] = var;

		no++;
	}
	std::cout << Filter_convexpand1x1_fire2[0] << ' ';

	/***********************************Getting  weights of the filter in to the array fire2expand3x3 *******************/
	std::fstream fire2expand3x3("fire3_expand3x3.txt", std::ios_base::in);
	var=0;

	std::vector<int> Filter_convexpand3x3_fire2;
	Filter_convexpand3x3_fire2.resize(64 * 9 * 16);
	no = 0;
	while (fire2expand3x3 >> var)
	{
		Filter_convexpand3x3_fire2[no] = var;

		no++;
	}
	std::cout << Filter_convexpand3x3_fire2[0] << ' ';

	/***********************************Getting  weights of the filter in to the array fire3squeeze 1x1 *******************/
	std::fstream fire3squeeze("fire4_squeeze1x1.txt", std::ios_base::in);
	var = 0;

	std::vector<int> Filter_convsqueeze_fire3;
	Filter_convsqueeze_fire3.resize(128 * 1 * 32);
	no = 0;
	while (fire3squeeze >> var)
	{
		Filter_convsqueeze_fire3[no] = var;

		no++;
	}
	std::cout << Filter_convsqueeze_fire3[0] << ' ';
	
	/***********************************Getting  weights of the filter in to the array fire3expand1x1  *******************/
	std::fstream fire3expand1x1("fire4_expand1x1.txt", std::ios_base::in);
	var = 0;

	std::vector<int> Filter_convexpand1x1_fire3;
	Filter_convexpand1x1_fire3.resize(128 * 1 * 32);
	no = 0;
	while (fire3expand1x1 >> var)
	{
		Filter_convexpand1x1_fire3[no] = var;

		no++;
	}
	std::cout << Filter_convexpand1x1_fire3[0] << ' ';


	/***********************************Getting  weights of the filter in to the array fire3expand3x3  *******************/
	std::fstream fire3expand3x3("fire4_expand3x3.txt", std::ios_base::in);
	var = 0;

	std::vector<int> Filter_convexpand3x3_fire3;
	Filter_convexpand3x3_fire3.resize(128 * 9 * 32);
	no = 0;
	while (fire3expand3x3 >> var)
	{
		Filter_convexpand3x3_fire3[no] = var;

		no++;
	}
	std::cout << Filter_convexpand3x3_fire3[0] << ' ';

	/***********************************Getting  weights of the filter in to the array fire4squeeze  *******************/
	std::fstream fire4squeeze("fire5_squeeze1x1.txt", std::ios_base::in);
	var = 0;

	std::vector<int> Filter_convsqueeze_fire4;
	Filter_convsqueeze_fire4.resize(256 * 1 * 32);
	no = 0;
	while (fire4squeeze >> var)
	{
		Filter_convsqueeze_fire4[no] = var;

		no++;
	}
	std::cout << Filter_convsqueeze_fire4[0] << ' ';
	/***********************************Getting  weights of the filter in to the array fire3expand3x3  *******************/
	std::fstream fire4expand1x1("fire5_expand1x1.txt", std::ios_base::in);
	var = 0;

	std::vector<int> Filter_convexpand1x1_fire4;
	Filter_convexpand1x1_fire4.resize(128 * 9 * 32);
	no = 0;
	while (fire4expand1x1 >> var)
	{
		Filter_convexpand1x1_fire4[no] = var;

		no++;
	}
	std::cout << Filter_convexpand1x1_fire4[0] << ' ';
	/***********************************Getting  weights of the filter in to the array fire4expand3x3  *******************/
	std::fstream fire4expand3x3("fire5_expand3x3.txt", std::ios_base::in);
	var = 0;

	std::vector<int> Filter_convexpand3x3_fire4;
	Filter_convexpand3x3_fire4.resize(128 * 9 * 32);
	no = 0;
	while (fire4expand3x3 >> var)
	{
		Filter_convexpand3x3_fire4[no] = var;

		no++;
	}
	std::cout << Filter_convexpand3x3_fire4[0] << ' ';


	/***********************************Getting  weights of the filter in to the array fire5squeeze  *******************/
	std::fstream fire5squeeze("fire6_squeeze1x1.txt", std::ios_base::in);
	var = 0;

	std::vector<int> Filter_convsqueeze_fire5;
	Filter_convsqueeze_fire5.resize(256 * 1 * 48);
	no = 0;
	while (fire5squeeze >> var)
	{
		Filter_convsqueeze_fire5[no] = var;

		no++;
	}
	std::cout << Filter_convsqueeze_fire5[0] << ' ';
	/***********************************Getting  weights of the filter in to the array fire5expand1x1  *******************/
	std::fstream fire5expand1x1("fire6_expand1x1.txt", std::ios_base::in);
	var = 0;

	std::vector<int> Filter_convexpand1x1_fire5;
	Filter_convexpand1x1_fire5.resize(192 * 48);
	no = 0;
	while (fire5expand1x1 >> var)
	{
		Filter_convexpand1x1_fire5[no] = var;

		no++;
	}
	std::cout << Filter_convexpand1x1_fire5[0] << ' ';
	/***********************************Getting  weights of the filter in to the array fire5expand3x3 *******************/
	std::fstream fire5expand3x3("fire6_expand3x3.txt", std::ios_base::in);
	var = 0;

	std::vector<int> Filter_convexpand3x3_fire5;
	Filter_convexpand3x3_fire5.resize(192 * 9 * 48);
	no= 0;
	while (fire5expand3x3 >> var)
	{
		Filter_convexpand3x3_fire5[no] = var;

		no++;
	}
	std::cout << Filter_convexpand3x3_fire5[0] << ' ';
	
	/***********************************Getting  weights of the filter in to the array fire6squeeze  *******************/
	std::fstream fire6squeeze("fire7_squeeze1x1.txt", std::ios_base::in);
	var = 0;

	std::vector<int> Filter_convsqueeze_fire6;
	Filter_convsqueeze_fire6.resize(384 * 1 * 48);
	no = 0;
	while (fire6squeeze >> var)
	{
		Filter_convsqueeze_fire6[no] = var;

		no++;
	}
	std::cout << Filter_convsqueeze_fire6[0] << ' ';
	/***********************************Getting  weights of the filter in to the array fire6expand1x1  *******************/
	std::fstream fire6expand1x1("fire7_expand1x1.txt", std::ios_base::in);
	var = 0;

	std::vector<int> Filter_convexpand1x1_fire6;
	Filter_convexpand1x1_fire6.resize(192 * 48);
	no = 0;
	while (fire6expand1x1 >> var)
	{
		Filter_convexpand1x1_fire6[no] = var;

		no++;
	}
	std::cout << Filter_convexpand1x1_fire6[0] << ' ';

	/***********************************Getting  weights of the filter in to the array fire6expand3x3 *******************/
	std::fstream fire6expand3x3("fire7_expand3x3.txt", std::ios_base::in);
	var = 0;

	std::vector<int> Filter_convexpand3x3_fire6;
	Filter_convexpand3x3_fire6.resize(192 * 9 * 48);
	no = 0;
	while (fire6expand3x3 >> var)
	{
		Filter_convexpand3x3_fire6[no] = var;

		no++;
	}
	std::cout << Filter_convexpand3x3_fire6[0] << ' ';



	/***********************************Getting  weights of the filter in to the array fire7squeeze  *******************/
	std::fstream fire7squeeze("fire7_squeeze1x1.txt", std::ios_base::in);
	var = 0;

	std::vector<int> Filter_convsqueeze_fire7;
	Filter_convsqueeze_fire7.resize(384 * 1 * 64);
	no = 0;
	while (fire7squeeze >> var)
	{
		Filter_convsqueeze_fire7[no] = var;

		no++;
	}
	std::cout << Filter_convsqueeze_fire7[0] << ' ';
	/***********************************Getting  weights of the filter in to the array fire7expand1x1  *******************/
	std::fstream fire7expand1x1("fire7_expand1x1.txt", std::ios_base::in);
	var = 0;

	std::vector<int> Filter_convexpand1x1_fire7;
	Filter_convexpand1x1_fire7.resize(256 * 64);
	no = 0;
	while (fire7expand1x1 >> var)
	{
		Filter_convexpand1x1_fire7[no] = var;

		no++;
	}
	std::cout << Filter_convexpand1x1_fire7[0] << ' ';

	/***********************************Getting  weights of the filter in to the array fire7expand3x3 *******************/
	std::fstream fire7expand3x3("fire7_expand3x3.txt", std::ios_base::in);
	var = 0;

	std::vector<int> Filter_convexpand3x3_fire7;
	Filter_convexpand3x3_fire7.resize(256 * 9 * 64);
	no = 0;
	while (fire7expand3x3 >> var)
	{
		Filter_convexpand3x3_fire7[no] = var;

		no++;
	}
	std::cout << Filter_convexpand3x3_fire7[0] << ' ';



	/***********************************Getting  weights of the filter in to the array fire8squeeze  *******************/
	std::fstream fire8squeeze("fire9_squeeze1x1.txt", std::ios_base::in);
	var = 0;

	std::vector<int> Filter_convsqueeze_fire8;
	Filter_convsqueeze_fire8.resize(512 * 1 * 64);
	no = 0;
	while (fire8squeeze >> var)
	{
		Filter_convsqueeze_fire8[no] = var;

		no++;
	}
	std::cout << Filter_convsqueeze_fire8[0] << ' ';
	
	/***********************************Getting  weights of the filter in to the array fire8expand1x1  *******************/
	std::fstream fire8expand1x1("fire9_expand1x1.txt", std::ios_base::in);
	var = 0;

	std::vector<int> Filter_convexpand1x1_fire8;
	Filter_convexpand1x1_fire8.resize(256 * 64);
	no = 0;
	while (fire8expand1x1 >> var)
	{
		Filter_convexpand1x1_fire8[no] = var;

		no++;
	}
	std::cout << Filter_convexpand1x1_fire8[0] << ' ';

	
	/***********************************Getting  weights of the filter in to the array fire8expand3x3 *******************/
	std::fstream fire8expand3x3("fire9_expand3x3.txt", std::ios_base::in);
	var = 0;

	std::vector<int> Filter_convexpand3x3_fire8;
	Filter_convexpand3x3_fire8.resize(256 * 9 * 64);
	no = 0;
	while (fire8expand3x3 >> var)
	{
		Filter_convexpand3x3_fire8[no] = var;

		no++;
	}
	std::cout << Filter_convexpand3x3_fire8[0] << ' ';

	/***********************************Getting  weights of the filter in to the array conv1000 *******************/
	std::fstream conv1000("conv1000.txt", std::ios_base::in);
	var = 0;

	std::vector<int> Filter_conv1000;
	Filter_conv1000.resize(512 * 1000);
	no = 0;
	while (conv1000 >> var)
	{
		Filter_conv1000[no] = var;

		no++;
	}
	std::cout << Filter_conv1000[0] << ' ';


	//import image to an array

	unsigned int j;
	std::vector<int> image_data;

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
	//for (int in = 224; in < 224*224; in++)
	//printf("image data : %d \n ", image_data[in]);
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
		cl::Buffer image = cl::Buffer(context, CL_MEM_READ_ONLY, row * col * 3 * sizeof(int));
		cl::Buffer Filter_conv = cl::Buffer(context, CL_MEM_READ_ONLY, 64 * 3 * 3 * 3 * sizeof(int));
		cl::Buffer output_first_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 111 * 111 * 1 * 64 * sizeof(int));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(image, CL_TRUE, 0, row * col * 3 * sizeof(int), image_data.data());
		queue.enqueueWriteBuffer(Filter_conv, CL_TRUE, 0, 64 * 3 * 3 * 3 * sizeof(int), Filter_conv1.data());
		unsigned int input_channel, input_size, stride, output_size, start_channel_fire;

		input_channel = 3;
		input_size = 224;
		stride = 2;
		output_size = 111;
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

		std::vector<int> output;
		output.resize(111 * 111 * 1 * 64);
		queue.enqueueReadBuffer(output_first_layer, CL_TRUE, 0, ((111 * 111 * 1 * 64) * sizeof(int)), output.data());
		printf("output at host:%d \n", output[0]);



		////////////////////////////////// Pooling layer //////////////////////////////////////////////////////////////////////////

			// Make kernel
		cl::Kernel kernelpool(program, "pool");
		//float out_size_second_layer = ;
		// Create memory buffers
		cl::Buffer output_second_layer = cl::Buffer(context, CL_MEM_WRITE_ONLY, 55 * 55 * 1 * 64 * sizeof(int));

		input_size = 255;
		output_size = 127;

		kernelpool.setArg(0, output_first_layer);
		kernelpool.setArg(1, output_second_layer);
		kernelpool.setArg(2, input_size);
		kernelpool.setArg(3, output_size);

		cl::NDRange global1(64);
		cl::NDRange local1(1);
		queue.enqueueNDRangeKernel(kernelpool, cl::NullRange, global1, local1);
		std::vector<int> output2;
		output2.resize(55 * 55 * 1 * 64);
		queue.enqueueReadBuffer(output_second_layer, CL_TRUE, 0, ((55 * 55 * 1 * 64) * sizeof(int)), output2.data());
		printf("output at pooling layer:%d \n", output2[0]);

		/***********************************************Fire block 1 - squeeze conv 1x1 ****************************************************/
		cl::Kernel kernelconv1x1fire1(program, "conv2");

		// Create memory buffers

		cl::Buffer Filter_conv_fire1_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 64 * 1 * 16 * sizeof(int));
		cl::Buffer output_fire1_conv1x1_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 55 * 55 * 1 * 16 * sizeof(int));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_conv_fire1_block1, CL_TRUE, 0, 64 * 1 * 16 * sizeof(int), Filter_conv_fire1.data());


		input_channel = 64;
		input_size = 55;

		kernelconv1x1fire1.setArg(0, output_second_layer);
		kernelconv1x1fire1.setArg(1, Filter_conv_fire1_block1);
		kernelconv1x1fire1.setArg(2, output_fire1_conv1x1_layer);
		kernelconv1x1fire1.setArg(3, input_channel);
		kernelconv1x1fire1.setArg(4, input_size);


		cl::NDRange global2(16);
		cl::NDRange local2(1);
		queue.enqueueNDRangeKernel(kernelconv1x1fire1, cl::NullRange, global2, local2);

		std::vector<int> output3;
		output3.resize(55 * 55 * 1 * 16);
		queue.enqueueReadBuffer(output_fire1_conv1x1_layer, CL_TRUE, 0, ((55 * 55 * 1 * 16) * sizeof(int)), output3.data());
		
		printf("output at third squeezelayer:%d \n", output3[100]);

		/***********************************************Fire block 1 - expand conv 1x1 ***************************************************/
		cl::Kernel kernelconv1x1expandfire1(program, "conv2");
		// Create memory buffers
		cl::Buffer Filter_convexpand_fire1_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 64 * 1 * 16 * sizeof(int));
		cl::Buffer output_fire1_conv1x1expand_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 55 * 55 * 1 * 128 * sizeof(int));
		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_convexpand_fire1_block1, CL_TRUE, 0, 64 * 1 * 16 * sizeof(int), Filter_convexpand3x3_fire1.data());

		input_channel = 16;
		input_size = 55;

		kernelconv1x1expandfire1.setArg(0, output_fire1_conv1x1_layer);
		kernelconv1x1expandfire1.setArg(1, Filter_convexpand_fire1_block1);
		kernelconv1x1expandfire1.setArg(2, output_fire1_conv1x1expand_layer);
		kernelconv1x1expandfire1.setArg(3, input_channel);
		kernelconv1x1expandfire1.setArg(4, input_size);

		cl::NDRange global3(64);
		cl::NDRange local3(1);
		queue.enqueueNDRangeKernel(kernelconv1x1expandfire1, cl::NullRange, global3, local3);

		/***********************************************Fire block 1 - expand conv 3x3 ****************************************************/
		cl::Kernel kernelconv3x3expandfire1(program, "conv1");

		// Create memory buffers
		cl::Buffer Filter_convexpand3x3_fire1_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 64 * 9 * 16 * sizeof(int));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_convexpand3x3_fire1_block1, CL_TRUE, 0, 64 * 9 * 16 * sizeof(int), Filter_convexpand_fire1.data());

		input_channel = 16;
		input_size = 55;
		stride = 1;
		start_channel_fire = 64;
		output_size = 55;

		kernelconv3x3expandfire1.setArg(0, output_fire1_conv1x1_layer);
		kernelconv3x3expandfire1.setArg(1, Filter_convexpand3x3_fire1_block1);
		kernelconv3x3expandfire1.setArg(2, output_fire1_conv1x1expand_layer);
		kernelconv3x3expandfire1.setArg(3, input_channel);
		kernelconv3x3expandfire1.setArg(4, input_size);
		kernelconv3x3expandfire1.setArg(5, stride);
		kernelconv3x3expandfire1.setArg(6, output_size);
		kernelconv3x3expandfire1.setArg(7, start_channel_fire);

		cl::NDRange global4(64);
		cl::NDRange local4(1);
		queue.enqueueNDRangeKernel(kernelconv3x3expandfire1, cl::NullRange, global4, local4);
		std::vector<int> output4;
		//std::cout << output4.max_size() << std::endl;
		output4.resize(55 * 55 * 1 * 128);
		queue.enqueueReadBuffer(output_fire1_conv1x1expand_layer, CL_TRUE, 0, ((55 * 55 * 1 * 128) * sizeof(int)), output4.data());
		
		printf("output at expand fire:%d \n", output4[1002]);
 
		/***********************************************Fire block 2 - squeeze conv 1x1 ****************************************************/
		cl::Kernel kernelconv1x1fire2(program, "conv2");

		// Create memory buffers

		cl::Buffer Filter_conv_fire2_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 128 * 1 * 16 * sizeof(int));
		cl::Buffer output_fire2_conv1x1_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 55 * 55 * 1 * 16 * sizeof(int));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_conv_fire2_block1, CL_TRUE, 0, 128 * 1 * 16 * sizeof(int), Filter_convsqueeze_fire2.data());


		input_channel = 128;
		input_size = 55;

		kernelconv1x1fire2.setArg(0, output_fire1_conv1x1expand_layer);
		kernelconv1x1fire2.setArg(1, Filter_conv_fire2_block1);
		kernelconv1x1fire2.setArg(2, output_fire2_conv1x1_layer);
		kernelconv1x1fire2.setArg(3, input_channel);
		kernelconv1x1fire2.setArg(4, input_size);


		cl::NDRange global5(16);
		cl::NDRange local5(1);
		queue.enqueueNDRangeKernel(kernelconv1x1fire2, cl::NullRange, global5, local5);

		std::vector<int> output5;
		output5.resize(55 * 55 * 1 * 16);
		queue.enqueueReadBuffer(output_fire2_conv1x1_layer, CL_TRUE, 0, ((55 * 55 * 1 * 16) * sizeof(int)), output5.data());
		printf("output at fire2 squeezelayer:%d \n", output5[100]);
	
		/***********************************************Fire block 2 - expand conv 1x1 ****************************************************/
		cl::Kernel kernelconv1x1expandfire2(program, "conv2");
		// Create memory buffers
		cl::Buffer Filter_convexpand_fire2_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 64 * 1 * 16 * sizeof(int));
		cl::Buffer output_fire2_conv1x1expand_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 55 * 55 * 1 * 128 * sizeof(int));
		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_convexpand_fire2_block1, CL_TRUE, 0, 64 * 1 * 16 * sizeof(int), Filter_convexpand1x1_fire2.data());

		input_channel = 16;
		input_size = 55;

		kernelconv1x1expandfire2.setArg(0, output_fire2_conv1x1_layer);
		kernelconv1x1expandfire2.setArg(1, Filter_convexpand_fire2_block1);
		kernelconv1x1expandfire2.setArg(2, output_fire2_conv1x1expand_layer);
		kernelconv1x1expandfire2.setArg(3, input_channel);
		kernelconv1x1expandfire2.setArg(4, input_size);


		cl::NDRange global6(64);
		cl::NDRange local6(1);
		queue.enqueueNDRangeKernel(kernelconv1x1expandfire2, cl::NullRange, global6, local6);


		/***********************************************Fire block 2 - expand conv 3x3 ****************************************************/
		cl::Kernel kernelconv3x3expandfire2(program, "conv1");

		// Create memory buffers
		cl::Buffer Filter_convexpand3x3_fire2_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 64 * 9 * 16 * sizeof(int));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_convexpand3x3_fire2_block1, CL_TRUE, 0, 64 * 9 * 16 * sizeof(int), Filter_convexpand3x3_fire2.data());

		input_channel = 16;
		input_size = 55;
		stride = 1;
		start_channel_fire = 64;
		output_size = 55;

		kernelconv3x3expandfire2.setArg(0, output_fire2_conv1x1_layer);
		kernelconv3x3expandfire2.setArg(1, Filter_convexpand3x3_fire2_block1);
		kernelconv3x3expandfire2.setArg(2, output_fire2_conv1x1expand_layer);
		kernelconv3x3expandfire2.setArg(3, input_channel);
		kernelconv3x3expandfire2.setArg(4, input_size);
		kernelconv3x3expandfire2.setArg(5, stride);
		kernelconv3x3expandfire2.setArg(6, output_size);
		kernelconv3x3expandfire2.setArg(7, start_channel_fire);

		cl::NDRange global7(64);
		cl::NDRange local7(1);
		queue.enqueueNDRangeKernel(kernelconv3x3expandfire2, cl::NullRange, global7, local7);
		std::vector<int> output6;
		output6.resize(55 * 55 * 1 * 128);
		queue.enqueueReadBuffer(output_fire2_conv1x1expand_layer, CL_TRUE, 0, ((55 * 55 * 1 * 128) * sizeof(int)), output6.data());
		printf("output at expand fire2:%d \n", output6[100]);
		
		////////////////////////////////// Pooling layer -2 //////////////////////////////////////////////////////////////////////////

		// Make kernel
		cl::Kernel kernelpool1(program, "pool");
		
		// Create memory buffers
		cl::Buffer output_second_fire_pool_layer = cl::Buffer(context, CL_MEM_WRITE_ONLY, 27 * 27 * 1 * 128 * sizeof(int));

		input_size = 55;
		output_size = 27;

		kernelpool1.setArg(0, output_fire2_conv1x1expand_layer);
		kernelpool1.setArg(1, output_second_fire_pool_layer);
		kernelpool1.setArg(2, input_size);
		kernelpool1.setArg(3, output_size);

		cl::NDRange global8(128);
		cl::NDRange local8(1);
		queue.enqueueNDRangeKernel(kernelpool1, cl::NullRange, global8, local8);
		std::vector<int> output7;
		output7.resize(27 * 27 * 1 * 128);
		queue.enqueueReadBuffer(output_second_fire_pool_layer, CL_TRUE, 0, ((27 * 27 * 1 * 128) * sizeof(int)), output7.data());
		printf("output at pooling layer 2:%d \n", output7[100]);
	
		cl_int status = queue.finish();
		printf("finishing the first two fire blocks of the squeezenet %d \n \n \n", status);



	
		/***********************************************Fire block 3 - squeeze conv 1x1 ****************************************************/
		cl::Kernel kernelconv1x1fire3(program, "conv2");

		// Create memory buffers

		cl::Buffer Filter_conv_fire3_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 128 * 1 * 32 * sizeof(int));
		cl::Buffer output_fire3_conv1x1_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 27 * 27 * 1 * 32 * sizeof(int));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_conv_fire3_block1, CL_TRUE, 0, 128 * 1 * 32 * sizeof(int), Filter_convsqueeze_fire3.data());


		input_channel = 128;
		input_size = 27;

		kernelconv1x1fire3.setArg(0, output_second_fire_pool_layer);
		kernelconv1x1fire3.setArg(1, Filter_conv_fire3_block1);
		kernelconv1x1fire3.setArg(2, output_fire3_conv1x1_layer);
		kernelconv1x1fire3.setArg(3, input_channel);
		kernelconv1x1fire3.setArg(4, input_size);


		cl::NDRange global9(32);
		cl::NDRange local9(1);
		queue.enqueueNDRangeKernel(kernelconv1x1fire3, cl::NullRange, global9, local9);

		std::vector<int> output8;
		output8.resize(27 * 27 * 1 * 32);
		queue.enqueueReadBuffer(output_fire3_conv1x1_layer, CL_TRUE, 0, ((27 * 27 * 1 * 32) * sizeof(int)), output8.data());
		printf("output at fire3 squeezelayer:%d \n", output8[100]);
		
		/***********************************************Fire block 3 - expand conv 1x1 ****************************************************/
		cl::Kernel kernelconv1x1expandfire3(program, "conv2");
		// Create memory buffers
		cl::Buffer Filter_convexpand_fire3_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 128 * 1 * 32 * sizeof(int));
		cl::Buffer output_fire3_conv1x1expand_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 27 * 27 * 1 * 256 * sizeof(int));
		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_convexpand_fire3_block1, CL_TRUE, 0, 128 * 1 * 32 * sizeof(int), Filter_convexpand1x1_fire3.data());

		input_channel = 32;
		input_size = 27;

		kernelconv1x1expandfire3.setArg(0, output_fire3_conv1x1_layer);
		kernelconv1x1expandfire3.setArg(1, Filter_convexpand_fire3_block1);
		kernelconv1x1expandfire3.setArg(2, output_fire3_conv1x1expand_layer);
		kernelconv1x1expandfire3.setArg(3, input_channel);
		kernelconv1x1expandfire3.setArg(4, input_size);


		cl::NDRange global10(128);
		cl::NDRange local10(1);
		queue.enqueueNDRangeKernel(kernelconv1x1expandfire3, cl::NullRange, global10, local10);


		/***********************************************Fire block 3 - expand conv 3x3 ****************************************************/
		cl::Kernel kernelconv3x3expandfire3(program, "conv1");

		// Create memory buffers
		cl::Buffer Filter_convexpand3x3_fire3_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 32 * 9 * 128 * sizeof(int));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_convexpand3x3_fire3_block1, CL_TRUE, 0, 32 * 9 * 128 * sizeof(int), Filter_convexpand3x3_fire3.data());

		input_channel = 32;
		input_size = 27;
		stride = 1;
		start_channel_fire = 128;
		output_size = 27;

		kernelconv3x3expandfire3.setArg(0, output_fire3_conv1x1_layer);
		kernelconv3x3expandfire3.setArg(1, Filter_convexpand3x3_fire3_block1);
		kernelconv3x3expandfire3.setArg(2, output_fire3_conv1x1expand_layer);
		kernelconv3x3expandfire3.setArg(3, input_channel);
		kernelconv3x3expandfire3.setArg(4, input_size);
		kernelconv3x3expandfire3.setArg(5, stride);
		kernelconv3x3expandfire3.setArg(6, output_size);
		kernelconv3x3expandfire3.setArg(7, start_channel_fire);

		cl::NDRange global11(128);
		cl::NDRange local11(1);
		queue.enqueueNDRangeKernel(kernelconv3x3expandfire3, cl::NullRange, global11, local11);
		std::vector<int> output9;
		output9.resize(27 * 27 * 1 * 256);
		queue.enqueueReadBuffer(output_fire3_conv1x1expand_layer, CL_TRUE, 0, ((27 * 27 * 1 * 256) * sizeof(int)), output9.data());
		printf("output at expand fire3 layer: %d \n", output9[11]);


		/***********************************************Fire block 4 - squeeze conv 1x1 ****************************************************/
		cl::Kernel kernelconv1x1fire4(program, "conv2");

		// Create memory buffers

		cl::Buffer Filter_conv_fire4_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 256 * 1 * 32 * sizeof(int));
		cl::Buffer output_fire4_conv1x1_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 27 * 27 * 1 * 32 * sizeof(int));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_conv_fire4_block1, CL_TRUE, 0, 256 * 1 * 32 * sizeof(int), Filter_convsqueeze_fire4.data());


		input_channel = 256;
		input_size = 27;

		kernelconv1x1fire4.setArg(0, output_fire3_conv1x1expand_layer);
		kernelconv1x1fire4.setArg(1, Filter_conv_fire4_block1);
		kernelconv1x1fire4.setArg(2, output_fire4_conv1x1_layer);
		kernelconv1x1fire4.setArg(3, input_channel);
		kernelconv1x1fire4.setArg(4, input_size);


		cl::NDRange global12(32);
		cl::NDRange local12(1);
		queue.enqueueNDRangeKernel(kernelconv1x1fire4, cl::NullRange, global12, local12);

		std::vector<int> output10;
		output10.resize(27 * 27 * 1 * 32);
		queue.enqueueReadBuffer(output_fire4_conv1x1_layer, CL_TRUE, 0, ((27 * 27 * 1 * 32) * sizeof(int)), output10.data());
		printf("output at fire4 squeezelayer:%d \n", output10[1000]);

		/***********************************************Fire block 4 - expand conv 1x1 ****************************************************/
		cl::Kernel kernelconv1x1expandfire4(program, "conv2");
		// Create memory buffers
		cl::Buffer Filter_convexpand_fire4_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 128 * 1 * 32 * sizeof(int));
		cl::Buffer output_fire4_conv1x1expand_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 27 * 27 * 1 * 256 * sizeof(int));
		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_convexpand_fire4_block1, CL_TRUE, 0, 128 * 1 * 32 * sizeof(int), Filter_convexpand1x1_fire4.data());

		input_channel = 32;
		input_size = 27;

		kernelconv1x1expandfire4.setArg(0, output_fire4_conv1x1_layer);
		kernelconv1x1expandfire4.setArg(1, Filter_convexpand_fire4_block1);
		kernelconv1x1expandfire4.setArg(2, output_fire4_conv1x1expand_layer);
		kernelconv1x1expandfire4.setArg(3, input_channel);
		kernelconv1x1expandfire4.setArg(4, input_size);


		cl::NDRange global13(128);
		cl::NDRange local13(1);
		queue.enqueueNDRangeKernel(kernelconv1x1expandfire4, cl::NullRange, global13, local13);


		/***********************************************Fire block 4 - expand conv 3x3 ****************************************************/
		cl::Kernel kernelconv3x3expandfire4(program, "conv1");

		// Create memory buffers
		cl::Buffer Filter_convexpand3x3_fire4_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 32 * 9 * 128 * sizeof(int));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_convexpand3x3_fire4_block1, CL_TRUE, 0, 32 * 9 * 128 * sizeof(int), Filter_convexpand3x3_fire4.data());

		input_channel = 32;
		input_size = 27;
		stride = 1;
		start_channel_fire = 128;
		output_size = 27;

		kernelconv3x3expandfire4.setArg(0, output_fire4_conv1x1_layer);
		kernelconv3x3expandfire4.setArg(1, Filter_convexpand3x3_fire4_block1);
		kernelconv3x3expandfire4.setArg(2, output_fire4_conv1x1expand_layer);
		kernelconv3x3expandfire4.setArg(3, input_channel);
		kernelconv3x3expandfire4.setArg(4, input_size);
		kernelconv3x3expandfire4.setArg(5, stride);
		kernelconv3x3expandfire4.setArg(6, output_size);
		kernelconv3x3expandfire4.setArg(7, start_channel_fire);

		cl::NDRange global14(128);
		cl::NDRange local14(1);
		queue.enqueueNDRangeKernel(kernelconv3x3expandfire4, cl::NullRange, global14, local14);
		std::vector<int> output11;
		output11.resize(27 * 27 * 1 * 256);
		queue.enqueueReadBuffer(output_fire4_conv1x1expand_layer, CL_TRUE, 0, ((27 * 27 * 1 * 256) * sizeof(int)), output11.data());
		//for (int in = 0; in < 10; in++)
			printf("output at expand fire4 layer: %d \n", output11[100]);

		////////////////////////////////// Pooling layer -3 //////////////////////////////////////////////////////////////////////////

		// Make kernel
		cl::Kernel kernelpool3(program, "pool");

		// Create memory buffers
		cl::Buffer output_third_fire_pool_layer = cl::Buffer(context, CL_MEM_WRITE_ONLY, 13 * 13 * 1 * 256 * sizeof(int));

		input_size = 27;
		output_size = 13;

		kernelpool3.setArg(0, output_fire4_conv1x1expand_layer);
		kernelpool3.setArg(1, output_third_fire_pool_layer);
		kernelpool3.setArg(2, input_size);
		kernelpool3.setArg(3, output_size);

		cl::NDRange globalsize(128);
		cl::NDRange localsize(1);
		queue.enqueueNDRangeKernel(kernelpool3, cl::NullRange, globalsize, localsize);
		std::vector<int> output12;
		output12.resize(13 * 13 * 1 * 256);
		queue.enqueueReadBuffer(output_second_fire_pool_layer, CL_TRUE, 0, ((13 * 13 * 1 * 256) * sizeof(int)), output12.data());
		printf("output at pooling layer3 : %d \n", output12[100]);

		status = queue.finish();
		printf("finishing the first four fire blocks and max pool of the squeezenet %d \n \n \n", status);

		/***********************************************Fire block 5 - squeeze conv 1x1 ****************************************************/
		cl::Kernel kernelconv1x1fire5(program, "conv2");

		// Create memory buffers

		cl::Buffer Filter_conv_fire5_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 256 * 1 * 48 * sizeof(int));
		cl::Buffer output_fire5_conv1x1_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 13 * 13 * 1 * 48 * sizeof(int));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_conv_fire5_block1, CL_TRUE, 0, 256 * 1 * 48 * sizeof(int), Filter_convsqueeze_fire5.data());


		input_channel = 256;
		input_size = 13;

		kernelconv1x1fire5.setArg(0, output_second_fire_pool_layer);
		kernelconv1x1fire5.setArg(1, Filter_conv_fire5_block1);
		kernelconv1x1fire5.setArg(2, output_fire5_conv1x1_layer);
		kernelconv1x1fire5.setArg(3, input_channel);
		kernelconv1x1fire5.setArg(4, input_size);


		cl::NDRange globalsize1(48);
		cl::NDRange localsize1(1);
		queue.enqueueNDRangeKernel(kernelconv1x1fire5, cl::NullRange, globalsize1, localsize1);

		std::vector<int> out;
		out.resize(13 * 13 * 1 * 48);
		queue.enqueueReadBuffer(output_fire5_conv1x1_layer, CL_TRUE, 0, ((13 * 13 * 1 * 48) * sizeof(int)), out.data());
		printf("output at fire5 squeezelayer:%d \n", out[1000]);
		
		/***********************************************Fire block 5 - expand conv 1x1 ****************************************************/
		cl::Kernel kernelconv1x1expandfire5(program, "conv2");
		// Create memory buffers
		cl::Buffer Filter_convexpand_fire5_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 192 * 1 * 48 * sizeof(int));
		cl::Buffer output_fire5_conv1x1expand_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 13 * 13 * 1 * 384 * sizeof(int));
		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_convexpand_fire5_block1, CL_TRUE, 0, 192 * 1 * 48 * sizeof(int), Filter_convexpand1x1_fire5.data());

		input_channel = 48;
		input_size = 13;

		kernelconv1x1expandfire5.setArg(0, output_fire5_conv1x1_layer);
		kernelconv1x1expandfire5.setArg(1, Filter_convexpand_fire5_block1);
		kernelconv1x1expandfire5.setArg(2, output_fire5_conv1x1expand_layer);
		kernelconv1x1expandfire5.setArg(3, input_channel);
		kernelconv1x1expandfire5.setArg(4, input_size);


		cl::NDRange globalsize3(192);
		cl::NDRange localsize3(1);
		queue.enqueueNDRangeKernel(kernelconv1x1expandfire5, cl::NullRange, globalsize3, localsize3);


		/***********************************************Fire block 5 - expand conv 3x3 ****************************************************/
		cl::Kernel kernelconv3x3expandfire5(program, "conv1");

		// Create memory buffers
		cl::Buffer Filter_convexpand3x3_fire5_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 192 * 9 * 48 * sizeof(int));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_convexpand3x3_fire5_block1, CL_TRUE, 0, 192 * 9 * 48 * sizeof(int), Filter_convexpand3x3_fire5.data());

		input_channel = 48;
		input_size = 13;
		stride = 1;
		start_channel_fire = 192;
		output_size = 13;

		kernelconv3x3expandfire5.setArg(0, output_fire5_conv1x1_layer);
		kernelconv3x3expandfire5.setArg(1, Filter_convexpand3x3_fire5_block1);
		kernelconv3x3expandfire5.setArg(2, output_fire5_conv1x1expand_layer);
		kernelconv3x3expandfire5.setArg(3, input_channel);
		kernelconv3x3expandfire5.setArg(4, input_size);
		kernelconv3x3expandfire5.setArg(5, stride);
		kernelconv3x3expandfire5.setArg(6, output_size);
		kernelconv3x3expandfire5.setArg(7, start_channel_fire);

		cl::NDRange globalid(192);
		cl::NDRange localid(1);
		queue.enqueueNDRangeKernel(kernelconv3x3expandfire5, cl::NullRange, globalid, localid);
		std::vector<int> output20;
		output20.resize(13 * 13 * 1 * 384);
		queue.enqueueReadBuffer(output_fire5_conv1x1expand_layer, CL_TRUE, 0, ((13 * 13 * 1 * 384) * sizeof(int)), output20.data());
		//for (int in = 0; in < 10; in++)
		printf("output at expand fire5 layer: %d \n", output20[100]);




		/***********************************************Fire block 6 - squeeze conv 1x1 ****************************************************/
		cl::Kernel kernelconv1x1fire6(program, "conv2");

		// Create memory buffers

		cl::Buffer Filter_conv_fire6_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 384 * 1 * 48 * sizeof(int));
		cl::Buffer output_fire6_conv1x1_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 13 * 13 * 1 * 48 * sizeof(int));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_conv_fire6_block1, CL_TRUE, 0, 384 * 1 * 48 * sizeof(int), Filter_convsqueeze_fire6.data());


		input_channel = 384;
		input_size = 13;

		kernelconv1x1fire6.setArg(0, output_fire5_conv1x1expand_layer);
		kernelconv1x1fire6.setArg(1, Filter_conv_fire6_block1);
		kernelconv1x1fire6.setArg(2, output_fire6_conv1x1_layer);
		kernelconv1x1fire6.setArg(3, input_channel);
		kernelconv1x1fire6.setArg(4, input_size);


		cl::NDRange globalsize2(48);
		cl::NDRange localsize2(1);
		queue.enqueueNDRangeKernel(kernelconv1x1fire6, cl::NullRange, globalsize2, localsize2);

		std::vector<int> out1;
		out1.resize(13 * 13 * 1 * 48);
		queue.enqueueReadBuffer(output_fire6_conv1x1_layer, CL_TRUE, 0, ((13 * 13 * 1 * 48) * sizeof(int)), out1.data());
		printf("output at fire6 squeezelayer:%d \n", out1[100]);

		/***********************************************Fire block 6 - expand conv 1x1 ****************************************************/
		cl::Kernel kernelconv1x1expandfire6(program, "conv2");
		// Create memory buffers
		cl::Buffer Filter_convexpand_fire6_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 192 * 1 * 48 * sizeof(int));
		cl::Buffer output_fire6_conv1x1expand_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 13 * 13 * 1 * 384 * sizeof(int));
		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_convexpand_fire6_block1, CL_TRUE, 0, 192 * 1 * 48 * sizeof(int), Filter_convexpand1x1_fire6.data());

		input_channel = 48;
		input_size = 13;

		kernelconv1x1expandfire6.setArg(0, output_fire6_conv1x1_layer);
		kernelconv1x1expandfire6.setArg(1, Filter_convexpand_fire6_block1);
		kernelconv1x1expandfire6.setArg(2, output_fire6_conv1x1expand_layer);
		kernelconv1x1expandfire6.setArg(3, input_channel);
		kernelconv1x1expandfire6.setArg(4, input_size);


		cl::NDRange globalsize4(192);
		cl::NDRange localsize4(1);
		queue.enqueueNDRangeKernel(kernelconv1x1expandfire6, cl::NullRange, globalsize4, localsize4);


		/***********************************************Fire block 6 - expand conv 3x3 ****************************************************/
		cl::Kernel kernelconv3x3expandfire6(program, "conv1");

		// Create memory buffers
		cl::Buffer Filter_convexpand3x3_fire6_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 192 * 9 * 48 * sizeof(int));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_convexpand3x3_fire6_block1, CL_TRUE, 0, 192 * 9 * 48 * sizeof(int), Filter_convexpand3x3_fire6.data());
		  
		input_channel = 48;
		input_size = 13;
		stride = 1;
		start_channel_fire = 192;
		output_size = 13;

		kernelconv3x3expandfire6.setArg(0, output_fire6_conv1x1_layer);
		kernelconv3x3expandfire6.setArg(1, Filter_convexpand3x3_fire6_block1);
		kernelconv3x3expandfire6.setArg(2, output_fire6_conv1x1expand_layer);
		kernelconv3x3expandfire6.setArg(3, input_channel);
		kernelconv3x3expandfire6.setArg(4, input_size);
		kernelconv3x3expandfire6.setArg(5, stride);
		kernelconv3x3expandfire6.setArg(6, output_size);
		kernelconv3x3expandfire6.setArg(7, start_channel_fire);

		cl::NDRange globalid1(192);
		cl::NDRange localid1(1);
		queue.enqueueNDRangeKernel(kernelconv3x3expandfire6, cl::NullRange, globalid1, localid1);
		std::vector<int> out20;
		out20.resize(13 * 13 * 1 * 384);
		queue.enqueueReadBuffer(output_fire6_conv1x1expand_layer, CL_TRUE, 0, ((13 * 13 * 1 * 384) * sizeof(int)), out20.data());
		//for (int in = 0; in < 10; in++)
		printf("output at expand fire6 layer: %d \n", out20[100]);



		/***********************************************Fire block 7 - squeeze conv 1x1 ****************************************************/
		cl::Kernel kernelconv1x1fire7(program, "conv2");

		// Create memory buffers

		cl::Buffer Filter_conv_fire7_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 384 * 1 * 64 * sizeof(int));
		cl::Buffer output_fire7_conv1x1_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 13 * 13 * 1 * 64 * sizeof(int));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_conv_fire7_block1, CL_TRUE, 0, 384 * 1 * 48 * sizeof(int), Filter_convsqueeze_fire6.data());


		input_channel = 384;
		input_size = 13;

		kernelconv1x1fire7.setArg(0, output_fire6_conv1x1expand_layer);
		kernelconv1x1fire7.setArg(1, Filter_conv_fire7_block1);
		kernelconv1x1fire7.setArg(2, output_fire7_conv1x1_layer);
		kernelconv1x1fire7.setArg(3, input_channel);
		kernelconv1x1fire7.setArg(4, input_size);


		cl::NDRange globalsize6(64);
		cl::NDRange localsize6(1);
		queue.enqueueNDRangeKernel(kernelconv1x1fire7, cl::NullRange, globalsize6, localsize6);

		std::vector<int> out2;
		out2.resize(13 * 13 * 1 * 64);
		queue.enqueueReadBuffer(output_fire6_conv1x1_layer, CL_TRUE, 0, ((13 * 13 * 1 * 64) * sizeof(int)), out2.data());
		printf("output at fire7 squeezelayer:%d \n", out2[100]);

		/***********************************************Fire block 7 - expand conv 1x1 ****************************************************/
		cl::Kernel kernelconv1x1expandfire7(program, "conv2");
		// Create memory buffers
		cl::Buffer Filter_convexpand_fire7_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 256 * 1 * 64 * sizeof(int));
		cl::Buffer output_fire7_conv1x1expand_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 13 * 13 * 1 * 512 * sizeof(int));
		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_convexpand_fire7_block1, CL_TRUE, 0, 256 * 1 * 64 * sizeof(int), Filter_convexpand1x1_fire6.data());

		input_channel = 64;
		input_size = 13;

		kernelconv1x1expandfire7.setArg(0, output_fire6_conv1x1_layer);
		kernelconv1x1expandfire7.setArg(1, Filter_convexpand_fire7_block1);
		kernelconv1x1expandfire7.setArg(2, output_fire7_conv1x1expand_layer);
		kernelconv1x1expandfire7.setArg(3, input_channel);
		kernelconv1x1expandfire7.setArg(4, input_size);


		cl::NDRange globalsize5(256);
		cl::NDRange localsize5(1);
		queue.enqueueNDRangeKernel(kernelconv1x1expandfire7, cl::NullRange, globalsize5, localsize5);


		/***********************************************Fire block 7 - expand conv 3x3 ****************************************************/
		cl::Kernel kernelconv3x3expandfire7(program, "conv1");

		// Create memory buffers
		cl::Buffer Filter_convexpand3x3_fire7_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 256 * 9 * 64 * sizeof(int));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_convexpand3x3_fire7_block1, CL_TRUE, 0, 256 * 9 * 64 * sizeof(int), Filter_convexpand3x3_fire6.data());

		input_channel = 64;
		input_size = 13;
		stride = 1;
		start_channel_fire = 256;
		output_size = 13;

		kernelconv3x3expandfire6.setArg(0, output_fire6_conv1x1_layer);
		kernelconv3x3expandfire6.setArg(1, Filter_convexpand3x3_fire7_block1);
		kernelconv3x3expandfire6.setArg(2, output_fire7_conv1x1expand_layer);
		kernelconv3x3expandfire6.setArg(3, input_channel);
		kernelconv3x3expandfire6.setArg(4, input_size);
		kernelconv3x3expandfire6.setArg(5, stride);
		kernelconv3x3expandfire6.setArg(6, output_size);
		kernelconv3x3expandfire6.setArg(7, start_channel_fire);

		cl::NDRange globalid2(256);
		cl::NDRange localid2(1);
		queue.enqueueNDRangeKernel(kernelconv3x3expandfire6, cl::NullRange, globalid2, localid2);
		std::vector<int> out10;
		out10.resize(13 * 13 * 1 * 512);
		queue.enqueueReadBuffer(output_fire7_conv1x1expand_layer, CL_TRUE, 0, ((13 * 13 * 1 * 512) * sizeof(int)), out10.data());
		//for (int in = 0; in < 10; in++)
		printf("output at expand fire7 layer: %d \n", out10[100]);
		


		/***********************************************Fire block 8 - squeeze conv 1x1 ****************************************************/
		cl::Kernel kernelconv1x1fire8(program, "conv2");

		// Create memory buffers

		cl::Buffer Filter_conv_fire8_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 512 * 1 * 64 * sizeof(int));
		cl::Buffer output_fire8_conv1x1_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 13 * 13 * 1 * 64 * sizeof(int));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_conv_fire8_block1, CL_TRUE, 0, 512 * 1 * 64 * sizeof(int), Filter_convsqueeze_fire8.data());
		
		input_channel = 512;
		input_size = 13;

		kernelconv1x1fire8.setArg(0, output_fire7_conv1x1expand_layer);
		kernelconv1x1fire8.setArg(1, Filter_conv_fire8_block1);
		kernelconv1x1fire8.setArg(2, output_fire8_conv1x1_layer);
		kernelconv1x1fire8.setArg(3, input_channel);
		kernelconv1x1fire8.setArg(4, input_size);


		cl::NDRange globalid12(64);
		cl::NDRange localid12(1);
		queue.enqueueNDRangeKernel(kernelconv1x1fire8, cl::NullRange, globalid12, localid12);

		std::vector<int> out6;
		out6.resize(13 * 13 * 1 * 64);
		queue.enqueueReadBuffer(output_fire8_conv1x1_layer, CL_TRUE, 0, ((13 * 13 * 1 * 64) * sizeof(int)), out6.data());
		printf("output at fire8 squeezelayer:%d \n", out6[100]);

		/***********************************************Fire block 8 - expand conv 1x1 ****************************************************/
		cl::Kernel kernelconv1x1expandfire8(program, "conv2");
		// Create memory buffers
		cl::Buffer Filter_convexpand_fire8_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 256 * 1 * 64 * sizeof(int));
		cl::Buffer output_fire8_conv1x1expand_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 13 * 13 * 1 * 512 * sizeof(int));
		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_convexpand_fire8_block1, CL_TRUE, 0, 256 * 1 * 64 * sizeof(int), Filter_convexpand1x1_fire8.data());
		 
		input_channel = 64;
		input_size = 13;

		kernelconv1x1expandfire8.setArg(0, output_fire8_conv1x1_layer);
		kernelconv1x1expandfire8.setArg(1, Filter_convexpand_fire8_block1);
		kernelconv1x1expandfire8.setArg(2, output_fire8_conv1x1expand_layer);
		kernelconv1x1expandfire8.setArg(3, input_channel);
		kernelconv1x1expandfire8.setArg(4, input_size);


		cl::NDRange globalsize123(256);
		cl::NDRange localsize123(1);
		queue.enqueueNDRangeKernel(kernelconv1x1expandfire8, cl::NullRange, globalsize123, localsize123);


		/***********************************************Fire block 8 - expand conv 3x3 ****************************************************/
		cl::Kernel kernelconv3x3expandfire8(program, "conv1");

		// Create memory buffers
		cl::Buffer Filter_convexpand3x3_fire8_block1 = cl::Buffer(context, CL_MEM_READ_ONLY, 256 * 9 * 64 * sizeof(int));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_convexpand3x3_fire8_block1, CL_TRUE, 0, 256 * 9 * 64 * sizeof(int), Filter_convexpand3x3_fire8.data());

		input_channel = 64;
		input_size = 13;
		stride = 1;
		start_channel_fire = 256;
		output_size = 13;

		kernelconv3x3expandfire8.setArg(0, output_fire8_conv1x1_layer);
		kernelconv3x3expandfire8.setArg(1, Filter_convexpand3x3_fire8_block1);
		kernelconv3x3expandfire8.setArg(2, output_fire8_conv1x1expand_layer);
		kernelconv3x3expandfire8.setArg(3, input_channel);
		kernelconv3x3expandfire8.setArg(4, input_size);
		kernelconv3x3expandfire8.setArg(5, stride);
		kernelconv3x3expandfire8.setArg(6, output_size);
		kernelconv3x3expandfire8.setArg(7, start_channel_fire);

		cl::NDRange globalid6(256);
		cl::NDRange localid6(1);
		queue.enqueueNDRangeKernel(kernelconv3x3expandfire8, cl::NullRange, globalid6, localid6);
		std::vector<int> out14;
		out14.resize(13 * 13 * 1 * 512);
		queue.enqueueReadBuffer(output_fire8_conv1x1expand_layer, CL_TRUE, 0, ((13 * 13 * 1 * 512) * sizeof(int)), out14.data());
		//for (int in = 0; in < 10; in++)
		printf("output at expand fire8 layer: %d \n", out14[110]);

		status = queue.finish();
		printf("finishing all fire blocks of the squeezenet %d \n \n \n", status);



		/***********************************************Conv 1x1 -1000 kernels****************************************************/
		cl::Kernel kernelconv1000(program, "conv2");

		// Create memory buffers

		cl::Buffer Filter_conv1000_1 = cl::Buffer(context, CL_MEM_READ_ONLY, 512 * 1 * 1000 * sizeof(int));
		cl::Buffer output_conv1000_layer = cl::Buffer(context, CL_MEM_READ_WRITE, 13 * 13 * 1 * 1000 * sizeof(int));

		// Copy lists A and B to the memory buffers
		queue.enqueueWriteBuffer(Filter_conv1000_1, CL_TRUE, 0, 512 * 1 * 1000 * sizeof(int), Filter_conv1000.data());

		input_channel = 512;
		input_size = 13;

		kernelconv1000.setArg(0, output_fire8_conv1x1expand_layer);
		kernelconv1000.setArg(1, Filter_conv1000_1);
		kernelconv1000.setArg(2, output_conv1000_layer);
		kernelconv1000.setArg(3, input_channel);
		kernelconv1000.setArg(4, input_size);


		cl::NDRange globalid13(1000);
		cl::NDRange localid13(1);
		queue.enqueueNDRangeKernel(kernelconv1000, cl::NullRange, globalid12, localid12);

		std::vector<int> out61;
		out61.resize(13 * 13 * 1 * 1000);
		queue.enqueueReadBuffer(output_conv1000_layer, CL_TRUE, 0, ((13 * 13 * 1 * 1000) * sizeof(int)), out61.data());
		printf("output at final conv1x1 squeezelayer:%d \n", out61[3442]);
	
	}
	catch (std::runtime_error error) {
		std::cout << error.what() << std::endl;
		
	}
	// Get available platforms
	return 0;
}