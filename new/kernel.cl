__kernel
void  conv1(global const float* image, global const float* Filter_conv, global float* output_first_layer, int input_channel, int input_size, int stride, int output_size) {
	
	//for(offset = 0; offset <64; offset++)
	//{
	/*int offset = 0;
	int filter_loc = 0,
	int output_first_layer_loc = 0;
	//	for (offset = 0; offset < 64; offset++)
	//	{  
	filter_loc += offset * input_channel * 9;
	output_first_layer_loc += (offset * output_size * output_size);
	printf("chec:%d", filter_loc);
	printf("chec:%d", output_first_layer_loc);*/


	//for (int filter_index=0  ; filter_index < 64; filter_index++) 
	//{
	int filter_index = get_global_id(0);
	Filter_conv += filter_index * input_channel * 9;
	output_first_layer += (filter_index) * output_size * output_size;
	
	int w = 0, h = 0;
	for (int i = 0; i < output_size; i++)
	{
		for (int j = 0; j < output_size; j++)
		{
			float tmp = 0;
			for (int k = 0; k < input_channel; k++)
			{
				
				for (int l = 0; l < 3; l++)
				{
					h = i * stride + l;
					
					for (int m = 0; m < 3; m++)
					{
						int w = j * stride + m;
						//printf("vakue of w %d \n", w);
						if ((h >= 0) && (h < input_size) && (w >= 0) && (w < input_size))
						{

							tmp += image[k * input_size * input_size + (i * stride + l) * input_size + j * stride + m]
								* Filter_conv[9 * k + 3 * l + m];

						}
					}
				}
			}
			//printf("tmp:%f \n", tmp);
			output_first_layer[i * output_size + j] = tmp;
		//	printf("OUTPUT:%f \n", output_first_layer[i * output_size + j]);
		}
	}
}


__kernel void pool (__global float* input_im, __global float* output_im,const int input_size,const int output_size)
{
/*	int channel = get_global_id(0);//get output channel index
	input_im += channel * input_size * input_size;
	output_im += channel * output_size * output_size;
	//loop over output feature map*/
	//for (int i = 0; i < output_size; i++)//row
	//{
	//for (int j = 0; j < output_size; j++)//col
		//{
			float tmp = 0.0;
			int i = 0, j = 0;
			for (int k = 0; k < 3; k++)//row
			{
				for (int l = 0; l < 3; l++)//col
				{
					//printf("input is %f \n",input_im[0]);
					float value = input_im[(i * 2 + k) * input_size + j * 2 + l];
					printf("value is %f \n",value);
					if (value > tmp)
						tmp = value;
					printf("tmp is %f \n", tmp);
				}

			}
			//store the result to output feature map
			output_im[i * output_size + j] = tmp;
			printf("output feature map %f \n", output_im[i * output_size + j]);

		//}

   // }

}