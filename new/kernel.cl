//pipe int p0 __attribute__((xcl_reqd_pipe_depth(128)));
__kernel
void  conv1(global const float* image, global const float* Filter_conv, global float* output_first_layer, 
	int input_channel, int input_size, int stride, int output_size,int start_channel_fire) {


	int filter_loc = 0;
	int output_first_layer_loc = 0;

	int offset = get_global_id(0);
	Filter_conv += offset * input_channel * 9;
	output_first_layer += (offset + start_channel_fire)* output_size * output_size;

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
					__attribute__((xcl_pipeline_loop))
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

			output_first_layer[i * output_size + j] = (tmp > 0.0) ? tmp : 0.0;

		}
	}
	//	write_pipe_block(p0, &output_first_layer);
}



__kernel void pool(__global float* input_im, __global float* output_im, const int input_size, const int output_size)
{
	int channel = get_global_id(0);//get output channel index
	input_im += channel * input_size * input_size;
	output_im += channel * output_size * output_size;
	//loop over output feature map
	for (int i = 0; i < output_size; i++)//row
	{
		for (int j = 0; j < output_size; j++)//col
		{
			float tmp = 0.0;
			for (int k = 0; k < 3; k++)//row
			{
				__attribute__((xcl_pipeline_loop))
					for (int l = 0; l < 3; l++)//col
					{

						float value = input_im[(i * 2 + k) * input_size + j * 2 + l];
						if (value > tmp)
							tmp = value;

					}

			}
			//store the result to output feature map
			output_im[i * output_size + j] = tmp;
			//printf("output feature map %f \n", output_im[i * output_size + j]);

		}

	}

}

__kernel void conv2(global const float* input_image_conv1x1, global const float* filter_conv1x1, global float* output_conv1x1_layer,
	int input_channels, int input_size)
{
	int filter_index = get_global_id(0); // 0 - (output_channels - 1)
	filter_conv1x1 += filter_index * input_channels;
	output_conv1x1_layer += filter_index * input_size * input_size;//start_channel is for 1x1 feature map in fire layer
		//loop over output feature map
	
	for (int i = 0; i < input_size; i++)
	{
		for (int j = 0; j < input_size; j++)
		{
			float tmp = 0;
			for (int k = 0; k < input_channels; k++)
			{
				//	tmp += input_image_conv1x1[0] * filter_conv1x1[0];
				tmp += input_image_conv1x1[k * input_size * input_size + i * input_size + j] * filter_conv1x1[k];

				//add relu after conv
				output_conv1x1_layer[i * input_size + j] = (tmp > 0.0) ? tmp : 0.0;
			}
		}
	}
}

__kernel void conv3(global const float* input_image_conv1x1, global const float* filter_conv1x1, global float* output_conv1x1_layer,
	int input_channels, int input_size)
{
	int filter_index = get_global_id(0); // 0 - (output_channels - 1)
	filter_conv1x1 += filter_index * input_channels;
	output_conv1x1_layer += filter_index * input_size * input_size;//start_channel is for 1x1 feature map in fire layer
		//loop over output feature map

	for (int i = 0; i < input_size; i++)
	{
		for (int j = 0; j < input_size; j++)
		{
			float tmp = 0;
			for (int k = 0; k < input_channels; k++)
			{
				//	tmp += input_image_conv1x1[0] * filter_conv1x1[0];
				tmp += input_image_conv1x1[k * input_size * input_size + i * input_size + j] * filter_conv1x1[k];

				//add relu after conv
				output_conv1x1_layer[i * input_size + j] = (tmp > 0.0) ? tmp : 0.0;
			}
		}
	}
}