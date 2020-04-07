#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "cnn.hpp"
#include "common.hpp"

/*
   Convolution()
 */
int Convolution0
(
  myFixed *filter,     
  int filter_size,    
  myFixed *input_data, 
  int input_width,    
  int input_height,   
  int input_depth,   
  myFixed *conv_out    
)
{
  int x, y;  
  int conv_width  = input_width  - 2 * (filter_size / 2);  
  int conv_height = input_height - 2 * (filter_size / 2);  
  int d;
  fullBit sum[3];
  int offset_i, offset_f; 

  for(y = 0; y < conv_height; ++y){
    for(x = 0; x < conv_width; ++x){

      for(d = 0; d < input_depth; ++d){
        #pragma HLS PIPELINE
        offset_i = (input_width * input_height * d);  
        offset_f = (filter_size * filter_size * d);  
     sum[d]=input_data[offset_i+((y+0)*input_width)+(x+0)]*filter[offset_f+(0*filter_size)+0]
           +input_data[offset_i+((y+0)*input_width)+(x+1)]*filter[offset_f+(0*filter_size)+1]
           +input_data[offset_i+((y+0)*input_width)+(x+2)]*filter[offset_f+(0*filter_size)+2]
           +input_data[offset_i+((y+0)*input_width)+(x+3)]*filter[offset_f+(0*filter_size)+3]
           +input_data[offset_i+((y+0)*input_width)+(x+4)]*filter[offset_f+(0*filter_size)+4]
           +input_data[offset_i+((y+1)*input_width)+(x+0)]*filter[offset_f+(1*filter_size)+0]
           +input_data[offset_i+((y+1)*input_width)+(x+1)]*filter[offset_f+(1*filter_size)+1]
           +input_data[offset_i+((y+1)*input_width)+(x+2)]*filter[offset_f+(1*filter_size)+2]
           +input_data[offset_i+((y+1)*input_width)+(x+3)]*filter[offset_f+(1*filter_size)+3]
           +input_data[offset_i+((y+1)*input_width)+(x+4)]*filter[offset_f+(1*filter_size)+4]
           +input_data[offset_i+((y+2)*input_width)+(x+0)]*filter[offset_f+(2*filter_size)+0]
           +input_data[offset_i+((y+2)*input_width)+(x+1)]*filter[offset_f+(2*filter_size)+1]
           +input_data[offset_i+((y+2)*input_width)+(x+2)]*filter[offset_f+(2*filter_size)+2]
           +input_data[offset_i+((y+2)*input_width)+(x+3)]*filter[offset_f+(2*filter_size)+3]
           +input_data[offset_i+((y+2)*input_width)+(x+4)]*filter[offset_f+(2*filter_size)+4]
           +input_data[offset_i+((y+3)*input_width)+(x+0)]*filter[offset_f+(3*filter_size)+0]
           +input_data[offset_i+((y+3)*input_width)+(x+1)]*filter[offset_f+(3*filter_size)+1]
           +input_data[offset_i+((y+3)*input_width)+(x+2)]*filter[offset_f+(3*filter_size)+2]
           +input_data[offset_i+((y+3)*input_width)+(x+3)]*filter[offset_f+(3*filter_size)+3]
           +input_data[offset_i+((y+3)*input_width)+(x+4)]*filter[offset_f+(3*filter_size)+4]
           +input_data[offset_i+((y+4)*input_width)+(x+0)]*filter[offset_f+(4*filter_size)+0]
           +input_data[offset_i+((y+4)*input_width)+(x+1)]*filter[offset_f+(4*filter_size)+1]
           +input_data[offset_i+((y+4)*input_width)+(x+2)]*filter[offset_f+(4*filter_size)+2]
           +input_data[offset_i+((y+4)*input_width)+(x+3)]*filter[offset_f+(4*filter_size)+3]
           +input_data[offset_i+((y+4)*input_width)+(x+4)]*filter[offset_f+(4*filter_size)+4];
      }
      conv_out[y * conv_width + x] = sum[0]+sum[1]+sum[2];
    }
  }

  return 0;
}

int Convolution1
(
  myFixed *filter,     
  int filter_size,     
  myFixed *input_data, 
  int input_width,     
  int input_height,    
  int input_depth,     
  myFixed *conv_out    
)
{
  int x, y;  
  int conv_width  = input_width  - 2 * (filter_size / 2);  
  int conv_height = input_height - 2 * (filter_size / 2); 
  int d;
  fullBit sum[4];
  int offset_i, offset_f; 

  for(y = 0; y < conv_height; ++y){
    for(x = 0; x < conv_width; ++x){
      

      for(d = 0; d < input_depth; ++d){
        #pragma HLS PIPELINE
        offset_i = (input_width * input_height * d);  
        offset_f = (filter_size * filter_size * d);  
        
     sum[d]=input_data[offset_i+((y+0)*input_width)+(x+0)]*filter[offset_f+(0*filter_size)+0]
           +input_data[offset_i+((y+0)*input_width)+(x+1)]*filter[offset_f+(0*filter_size)+1]
           +input_data[offset_i+((y+0)*input_width)+(x+2)]*filter[offset_f+(0*filter_size)+2]
           +input_data[offset_i+((y+0)*input_width)+(x+3)]*filter[offset_f+(0*filter_size)+3]
           +input_data[offset_i+((y+0)*input_width)+(x+4)]*filter[offset_f+(0*filter_size)+4]
           +input_data[offset_i+((y+1)*input_width)+(x+0)]*filter[offset_f+(1*filter_size)+0]
           +input_data[offset_i+((y+1)*input_width)+(x+1)]*filter[offset_f+(1*filter_size)+1]
           +input_data[offset_i+((y+1)*input_width)+(x+2)]*filter[offset_f+(1*filter_size)+2]
           +input_data[offset_i+((y+1)*input_width)+(x+3)]*filter[offset_f+(1*filter_size)+3]
           +input_data[offset_i+((y+1)*input_width)+(x+4)]*filter[offset_f+(1*filter_size)+4]
           +input_data[offset_i+((y+2)*input_width)+(x+0)]*filter[offset_f+(2*filter_size)+0]
           +input_data[offset_i+((y+2)*input_width)+(x+1)]*filter[offset_f+(2*filter_size)+1]
           +input_data[offset_i+((y+2)*input_width)+(x+2)]*filter[offset_f+(2*filter_size)+2]
           +input_data[offset_i+((y+2)*input_width)+(x+3)]*filter[offset_f+(2*filter_size)+3]
           +input_data[offset_i+((y+2)*input_width)+(x+4)]*filter[offset_f+(2*filter_size)+4]
           +input_data[offset_i+((y+3)*input_width)+(x+0)]*filter[offset_f+(3*filter_size)+0]
           +input_data[offset_i+((y+3)*input_width)+(x+1)]*filter[offset_f+(3*filter_size)+1]
           +input_data[offset_i+((y+3)*input_width)+(x+2)]*filter[offset_f+(3*filter_size)+2]
           +input_data[offset_i+((y+3)*input_width)+(x+3)]*filter[offset_f+(3*filter_size)+3]
           +input_data[offset_i+((y+3)*input_width)+(x+4)]*filter[offset_f+(3*filter_size)+4]
           +input_data[offset_i+((y+4)*input_width)+(x+0)]*filter[offset_f+(4*filter_size)+0]
           +input_data[offset_i+((y+4)*input_width)+(x+1)]*filter[offset_f+(4*filter_size)+1]
           +input_data[offset_i+((y+4)*input_width)+(x+2)]*filter[offset_f+(4*filter_size)+2]
           +input_data[offset_i+((y+4)*input_width)+(x+3)]*filter[offset_f+(4*filter_size)+3]
           +input_data[offset_i+((y+4)*input_width)+(x+4)]*filter[offset_f+(4*filter_size)+4];
      }
      conv_out[y * conv_width + x] = sum[0]+sum[1]+sum[2]+sum[3];
    }
  }

  return 0;
}

/*
   Pooling()関数
 */
void Pooling(
  myFixed *conv_out, 
  int conv_width,   
  int conv_height, 
  myFixed *pool_out, 
  int pool_size,     
  int pool_stride     
)
{
  int x = 0;
  int y = 0;
  int n = 0;
  int pool_width  = conv_width / pool_stride;  
  int pool_height = conv_height / pool_stride; 
  myFixed max_tmp[2];

  for(y = 0; y < pool_height; ++y){
    for(x = 0; x < pool_width; ++x){
      
      for(n = 0; n < pool_size; ++n){ 
          if(conv_out[(y * conv_width * pool_stride) + (n * conv_width ) + (x * pool_stride)] > 
            conv_out[(y * conv_width * pool_stride) + (n * conv_width ) + (x * pool_stride) + 1]) 
              max_tmp[n] = conv_out[(y * conv_width * pool_stride) + (n * conv_width) +(x * pool_stride)];
          else max_tmp[n] = conv_out[(y * conv_width * pool_stride) + (n * conv_width) + (x * pool_stride + 1)];
        }
      
        if(max_tmp[0] > max_tmp[1]) pool_out[(y * pool_width) + x] = max_tmp[0];
        else pool_out[(y * pool_width) + x] = max_tmp[1];

    }
  }
}

/*
   CNNLayer()関数
 */
int CNNLayer0(
  myFixed  *filter,     
  int filter_num,     
  int filter_size,    
  myFixed  *input_data, 
  int input_width,    
  int input_height,   
  int input_depth,    
  myFixed  *conv_out,   
  int conv_width,     
  int conv_height,    
  myFixed  *pool_out,   
  int pool_width,     
  int pool_height,    
  int pool_size,     
  int pool_stride     
)
{
  
  int i;
  int offset;

  for(i = 0; i < filter_num; ++i){
    
    
    Convolution0(
      &filter[filter_size*filter_size*input_depth*i],
      filter_size,
      input_data,
      input_width,
      input_height,
      input_depth,
      conv_out
    );
    
    offset = pool_width * pool_height * i;
    
    Pooling(
      conv_out,
      conv_width,
      conv_height,
      &pool_out[offset],
      pool_size,
      pool_stride
    );
  }
  return 0;
}

 int CNNLayer1(
  myFixed  *filter,     
  int filter_num,     
  int filter_size,    
  myFixed  *input_data, 
  int input_width,    
  int input_height,   
  int input_depth,    
  myFixed  *conv_out,   
  int conv_width,     
  int conv_height,    
  myFixed  *pool_out,   
  int pool_width,     
  int pool_height,    
  int pool_size,     
  int pool_stride     
)
{
  int i;
  int offset;

  for(i = 0; i < filter_num; ++i){
    Convolution1(
      &filter[filter_size*filter_size*input_depth*i],
      filter_size,
      input_data,
      input_width,
      input_height,
      input_depth,
      conv_out
    );
    offset = pool_width * pool_height * i;
    Pooling(
      conv_out,
      conv_width,
      conv_height,
      &pool_out[offset],
      pool_size,
      pool_stride
    );
  }
  return 0;
}

/*
   InitFilter()関数
    Load filter paramter.
*/
void InitFilter_1(
  myFixed  *filter,   
  int filter_size,  
  int filter_channel,  
  int filter_num    
)
{
  FILE *fp;                       
  if((fp=fopen("/mnt/params/weight_conv1.dat", "r+")) != NULL){
    fread(filter, sizeof(myFixed) * filter_size * filter_size * filter_channel * filter_num, 1, fp);
  }
  fclose(fp);
}

void InitFilter_2(
  myFixed *filter,   
  int filter_size,  
  int filter_channel,  
  int filter_num    
)
{
  FILE *fp;                       
  if((fp=fopen("/mnt/params/weight_conv2.dat", "r+")) != NULL){
    fread(filter, sizeof(myFixed) * filter_size * filter_size * filter_channel * filter_num, 1, fp);
  }
  fclose(fp);

}

void InitFilter_3(
  myFixed *filter,   
  int filter_size,  
  int filter_channel,  
  int filter_num    
)
{
  FILE *fp;                       
  if((fp=fopen("/mnt/params/weight_conv3.dat", "r+")) != NULL){
    fread(filter, sizeof(myFixed) * filter_size * filter_size * filter_channel * filter_num, 1, fp);
  }
  fclose(fp);

}

#pragma SDS data access_pattern(filter0:SEQUENTIAL)
#pragma SDS data access_pattern(input_data0:SEQUENTIAL)
#pragma SDS data access_pattern(conv_out0:SEQUENTIAL)
#pragma SDS data access_pattern(pool_out0:SEQUENTIAL)
#pragma SDS data zero_copy(filter0[0:5*5*3*4-1])
#pragma SDS data zero_copy(input_data0[0:60*60*3-1])
#pragma SDS data zero_copy(conv_out0[0:56*56*4-1])
#pragma SDS data zero_copy(pool_out0[0:28*28*4-1])
#pragma SDS data mem_attribute(filter0:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(input_data0:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(conv_out0:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(pool_out0:PHYSICAL_CONTIGUOUS)

#pragma SDS data access_pattern(filter1:SEQUENTIAL)
#pragma SDS data access_pattern(input_data1:SEQUENTIAL)
#pragma SDS data access_pattern(conv_out1:SEQUENTIAL)
#pragma SDS data access_pattern(pool_out1:SEQUENTIAL)
#pragma SDS data zero_copy(filter1[0:5*5*4*4-1])
#pragma SDS data zero_copy(input_data1[0:28*28*4-1])
#pragma SDS data zero_copy(conv_out1[0:24*24*4-1])
#pragma SDS data zero_copy(pool_out1[0:12*12*4-1])
#pragma SDS data mem_attribute(filter1:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(input_data1:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(conv_out1:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(pool_out1:PHYSICAL_CONTIGUOUS)

#pragma SDS data access_pattern(filter2:SEQUENTIAL)
#pragma SDS data access_pattern(input_data2:SEQUENTIAL)
#pragma SDS data access_pattern(conv_out2:SEQUENTIAL)
#pragma SDS data access_pattern(pool_out2:SEQUENTIAL)
#pragma SDS data zero_copy(filter2[0:5*5*4*8-1])
#pragma SDS data zero_copy(input_data2[0:12*12*4-1])
#pragma SDS data zero_copy(conv_out2[0:8*8*8-1])
#pragma SDS data zero_copy(pool_out2[0:4*4*8-1])
#pragma SDS data mem_attribute(filter2:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(input_data2:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(conv_out2:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(pool_out2:PHYSICAL_CONTIGUOUS)

void execCNN(
  myFixed *filter0,
  int filter_num0,     int filter_size0,    myFixed *input_data0,
  int input_width0,    int input_height0,   int input_depth0,
  myFixed *conv_out0,   int conv_width0,     int conv_height0,
  myFixed *pool_out0,   int pool_width0,     int pool_height0,
  int pool_size0, int pool_stride0,

  myFixed *filter1,
  int filter_num1,     int filter_size1,    myFixed *input_data1,
  int input_width1,    int input_height1,   int input_depth1,
  myFixed *conv_out1,   int conv_width1,     int conv_height1,
  myFixed *pool_out1,   int pool_width1,     int pool_height1,
  int pool_size1, int pool_stride1,
  
  myFixed *filter2,
  int filter_num2,     int filter_size2,    myFixed *input_data2,
  int input_width2,    int input_height2,   int input_depth2,
  myFixed *conv_out2,   int conv_width2,     int conv_height2,
  myFixed *pool_out2,   int pool_width2,     int pool_height2,
  int pool_size2, int pool_stride2

)
{
  myFixed out;
  
  myFixed buffer00[5*5*3*4]; myFixed buffer01[5*5*4*4]; myFixed buffer02[5*5*4*8];
  myFixed buffer1[60*60*3];
  myFixed buffer20[56*56*4]; myFixed buffer21[24*24*4]; myFixed buffer22[8*8*8];
  myFixed buffer30[28*28*4]; myFixed buffer31[12*12*4]; myFixed buffer32[4*4*8];
  
  #pragma HLS interface ap_memory port=buffer00
  #pragma HLS interface ap_memory port=buffer01
  #pragma HLS interface ap_memory port=buffer02
  #pragma HLS interface ap_memory port=buffer1
  memcpy(buffer00, filter0, sizeof(myFixed)*filter_size0*filter_size0*input_depth0*filter_num0);
  memcpy(buffer01, filter1, sizeof(myFixed)*filter_size1*filter_size1*input_depth1*filter_num1);
  memcpy(buffer02, filter2, sizeof(myFixed)*filter_size2*filter_size2*input_depth2*filter_num2);
  memcpy(buffer1, input_data0, sizeof(myFixed)*input_width0*input_height0*input_depth0);
  CNNLayer0(
  buffer00, 
  filter_num0,
  filter_size0,
  buffer1, 
  input_width0,
  input_height0,
  input_depth0,
  buffer20, 
  conv_width0,
  conv_height0,
  buffer30, 
  pool_width0,
  pool_height0,
  pool_size0,
  pool_stride0
  );

  CNNLayer1(
  buffer01, 
  filter_num1,
  filter_size1,
  buffer30, 
  input_width1,
  input_height1,
  input_depth1,
  buffer21, 
  conv_width1,
  conv_height1,
  buffer31, 
  pool_width1,
  pool_height1,
  pool_size1,
  pool_stride1
  );
  
  CNNLayer1(
  buffer02, 
  filter_num2,
  filter_size2,
  buffer31, 
  input_width2,
  input_height2,
  input_depth2,
  buffer22, 
  conv_width2,
  conv_height2,
  buffer32, 
  pool_width2,
  pool_height2,
  pool_size2,
  pool_stride2
  );

  #pragma HLS interface ap_memory port=buffer32
  memcpy(pool_out2, buffer32, sizeof(myFixed)*pool_width2*pool_height2*filter_num2);
}

#if 1 
/*
   Forward()関数
   頁E��向�E計箁E */

int Forward(
  myFixed  *input_data,     
  int input_num,          
  fcFixed *weight_hidden,  
  fcFixed *weight_out,     
  int hidden_num,          
  int out_num          
)
{
  int i = 0;
  int j = 0;
  fcFixed hidden_out[80]; 
  fcFullBit sum; 
  fcFullBit out[2];
  int label;

  for(i = 0; i < hidden_num; ++i){
      sum = 0;
    for(j = 0; j < input_num; ++j){
      sum += (fcFixed)input_data[j] * weight_hidden[i * (input_num) + j];
    }
    hidden_out[i] = (sum);
  }
  for(i = 0; i < out_num; ++i){
    out[i] = 0;
    for(j = 0; j < hidden_num; ++j){
    out[i] += hidden_out[j] * weight_out[i * (hidden_num) + j];
    }
  }
#if 0
  printf("Show hidden_out.\n");
  show_fc(hidden_out, 1, hidden_num);
  printf("\nout[0]: %.8lf, out[1]: %.8lf\n", (double)out[0], (double)out[1]);
#else
  printf("out[0]: %.8lf, out[1]: %.8lf,", (double)out[0], (double)out[1]);
#endif
  label = (out[0]>out[1]) ? 0 : 1;
  return label;
}
#endif
