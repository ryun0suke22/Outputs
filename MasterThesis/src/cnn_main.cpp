#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <sds_lib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "cnn.hpp"
#include "bitmap.hpp"
#include "common.hpp"
#include "cnn_main.hpp"

typedef struct{
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  myFixed * data;
} PoolImage;

typedef struct {
  char name[32];
  int teacher;
} InputList;

typedef struct{
  int filter_num;   
  int filter_size;  
  int pool_size;    
  int pool_stride;    

  int input_width, input_height, input_depth; 
  int conv_width, conv_height;                
  int pool_width, pool_height, pool_depth;    

  myFixed * filter;      
  myFixed * input_data;  
  myFixed * conv_out;    
  myFixed * pool_out;    
} CNNLayerImage;

double getusage(){
  struct rusage usage;
  struct timeval ut;

  getrusage(RUSAGE_SELF, &usage );
  ut = usage.ru_utime;

  return ((double)(ut.tv_sec)*1000 + (double)(ut.tv_usec)*0.001);
}

/*
   FreeCNNLayerImage()髢｢謨ｰ
*/
void FreeCNNLayerImage(
  CNNLayerImage * cnn_layer_image
)
{
  int i;

  for(i = 0; i < CNN_LAYER_NUM; ++i){
    sds_free(cnn_layer_image[i].filter);
    sds_free(cnn_layer_image[i].conv_out);
    sds_free(cnn_layer_image[i].pool_out);
  }
  sds_free(cnn_layer_image[0].input_data);
  sds_free(cnn_layer_image);
}

/*
    CNNLayerInit()髢｢謨ｰ
 */
 CNNLayerImage * CNNLayerInit(
)
{
  CNNLayerImage * cnn_layer_image;

  cnn_layer_image = (CNNLayerImage * )sds_alloc(sizeof(CNNLayerImage) * CNN_LAYER_NUM);

  cnn_layer_image[0].filter       = (myFixed *)sds_alloc(sizeof(myFixed) *
                                    CNN_LAYER0_FILTER_SIZE *
                                    CNN_LAYER0_FILTER_SIZE *
                                    INPUT_DATA_DEPTH *
                                    CNN_LAYER0_FILTER_NUM);
  cnn_layer_image[0].filter_num   = CNN_LAYER0_FILTER_NUM;
  cnn_layer_image[0].filter_size  = CNN_LAYER0_FILTER_SIZE;
  cnn_layer_image[0].pool_size    = CNN_LAYER0_POOL_SIZE;
  cnn_layer_image[0].pool_stride  = CNN_LAYER0_POOL_STRIDE;
  cnn_layer_image[0].input_width  = INPUT_DATA_SIZE;
  cnn_layer_image[0].input_height = INPUT_DATA_SIZE;
  cnn_layer_image[0].input_depth  = INPUT_DATA_DEPTH;
  cnn_layer_image[0].input_data   = (myFixed *)sds_alloc(sizeof(myFixed) *
                                    cnn_layer_image[0].input_width *
                                    cnn_layer_image[0].input_height *
                                    cnn_layer_image[0].input_depth );
  cnn_layer_image[0].conv_width   = cnn_layer_image[0].input_width  -
                                    (2 * (cnn_layer_image[0].filter_size / 2));
  cnn_layer_image[0].conv_height  = cnn_layer_image[0].input_height -
                                    (2 * (cnn_layer_image[0].filter_size / 2));
  cnn_layer_image[0].conv_out     = (myFixed *)sds_alloc(sizeof(myFixed) *
                                    cnn_layer_image[0].conv_width *
                                    cnn_layer_image[0].conv_height);
  cnn_layer_image[0].pool_width   = cnn_layer_image[0].conv_width /
                                    cnn_layer_image[0].pool_stride;
  cnn_layer_image[0].pool_height  = cnn_layer_image[0].conv_height /
                                    cnn_layer_image[0].pool_stride;
  cnn_layer_image[0].pool_depth   = cnn_layer_image[0].filter_num;
  cnn_layer_image[0].pool_out     = (myFixed *)sds_alloc(sizeof(myFixed) *
                                    cnn_layer_image[0].pool_width *
                                    cnn_layer_image[0].pool_height *
                                    cnn_layer_image[0].pool_depth);

  cnn_layer_image[1].filter       = (myFixed *)sds_alloc(sizeof(myFixed) *
                                    CNN_LAYER1_FILTER_SIZE *
                                    CNN_LAYER1_FILTER_SIZE *
                                    CNN_LAYER0_FILTER_NUM *
                                    CNN_LAYER1_FILTER_NUM);
  cnn_layer_image[1].filter_num   = CNN_LAYER1_FILTER_NUM;
  cnn_layer_image[1].filter_size  = CNN_LAYER1_FILTER_SIZE;
  cnn_layer_image[1].pool_size    = CNN_LAYER1_POOL_SIZE;
  cnn_layer_image[1].pool_stride  = CNN_LAYER1_POOL_STRIDE;
  cnn_layer_image[1].input_width  = cnn_layer_image[0].pool_width;
  cnn_layer_image[1].input_height = cnn_layer_image[0].pool_height;
  cnn_layer_image[1].input_depth  = cnn_layer_image[0].pool_depth;
  cnn_layer_image[1].input_data   = cnn_layer_image[0].pool_out;
  cnn_layer_image[1].conv_width   = cnn_layer_image[1].input_width  -
                                    (2 * (cnn_layer_image[1].filter_size / 2));
  cnn_layer_image[1].conv_height  = cnn_layer_image[1].input_height -
                                    (2 * (cnn_layer_image[1].filter_size / 2));
  cnn_layer_image[1].conv_out     = (myFixed *)sds_alloc(sizeof(myFixed) *
                                    cnn_layer_image[1].conv_width *
                                    cnn_layer_image[1].conv_height); 
  cnn_layer_image[1].pool_width   = cnn_layer_image[1].conv_width /
                                    cnn_layer_image[1].pool_stride;
  cnn_layer_image[1].pool_height  = cnn_layer_image[1].conv_height /
                                    cnn_layer_image[1].pool_stride;
  cnn_layer_image[1].pool_depth   = cnn_layer_image[1].filter_num;
  cnn_layer_image[1].pool_out     = (myFixed *)sds_alloc(sizeof(myFixed) *
                                    cnn_layer_image[1].pool_width *
                                    cnn_layer_image[1].pool_height *
                                    cnn_layer_image[1].pool_depth);

  cnn_layer_image[2].filter       = (myFixed *)sds_alloc(sizeof(myFixed) *
                                    CNN_LAYER2_FILTER_SIZE *
                                    CNN_LAYER2_FILTER_SIZE *
                                    CNN_LAYER1_FILTER_NUM *
                                    CNN_LAYER2_FILTER_NUM);
  cnn_layer_image[2].filter_num   = CNN_LAYER2_FILTER_NUM;
  cnn_layer_image[2].filter_size  = CNN_LAYER2_FILTER_SIZE;
  cnn_layer_image[2].pool_size    = CNN_LAYER2_POOL_SIZE;
  cnn_layer_image[2].pool_stride  = CNN_LAYER2_POOL_STRIDE;
  cnn_layer_image[2].input_width  = cnn_layer_image[1].pool_width;
  cnn_layer_image[2].input_height = cnn_layer_image[1].pool_height;
  cnn_layer_image[2].input_depth  = cnn_layer_image[1].pool_depth;
  cnn_layer_image[2].input_data   = cnn_layer_image[1].pool_out;
  cnn_layer_image[2].conv_width   = cnn_layer_image[2].input_width  -
                                    (2 * (cnn_layer_image[2].filter_size / 2));
  cnn_layer_image[2].conv_height  = cnn_layer_image[2].input_height -
                                    (2 * (cnn_layer_image[2].filter_size / 2));
  cnn_layer_image[2].conv_out     = (myFixed *)sds_alloc(sizeof(myFixed) *
                                    cnn_layer_image[2].conv_width *
                                    cnn_layer_image[2].conv_height);
  cnn_layer_image[2].pool_width   = cnn_layer_image[2].conv_width /
                                    cnn_layer_image[2].pool_stride;
  cnn_layer_image[2].pool_height  = cnn_layer_image[2].conv_height /
                                    cnn_layer_image[2].pool_stride;
  cnn_layer_image[2].pool_depth   = cnn_layer_image[2].filter_num;
  cnn_layer_image[2].pool_out     = (myFixed *)sds_alloc(sizeof(myFixed) *
                                    cnn_layer_image[2].pool_width *
                                    cnn_layer_image[2].pool_height *
                                    cnn_layer_image[2].pool_depth);

  return cnn_layer_image;
}

/*
   CNN()
 */
int CNN(
  InputList *input_image,
  int num_of_input_data,
  fcFixed *weight_hidden,
  fcFixed *weight_out
)
{
  int out;           
  int i;
  int pool_out_num;

  int input_width, input_height, input_depth; 

  Image *image;
  int d,x,y;
  int src_pt, dst_pt;

  double st, et;
  double middle = 0.0;
  double usage = 0.0;
  int usage_count = 0;

  CNNLayerImage * cnn_layer_image;
  cnn_layer_image = CNNLayerInit(); 

  
  InitFilter_1(cnn_layer_image[0].filter, CNN_LAYER0_FILTER_SIZE, INPUT_DATA_DEPTH, CNN_LAYER0_FILTER_NUM);
  InitFilter_2(cnn_layer_image[1].filter, CNN_LAYER1_FILTER_SIZE, CNN_LAYER0_FILTER_NUM, CNN_LAYER1_FILTER_NUM);
  InitFilter_3(cnn_layer_image[2].filter, CNN_LAYER2_FILTER_SIZE, CNN_LAYER1_FILTER_NUM, CNN_LAYER2_FILTER_NUM);

  for(i = 0; i < num_of_input_data; i++){  
    printf("File: %s(%d)", input_image[i].name, input_image[i].teacher);
    image = ReadBMP(input_image[i].name);
    input_width  = image->width;
    input_height = image->height;
    input_depth  = image->bpp/8;

    for(d = 0; d < input_depth; ++d){
      for(y = 0; y < input_height; ++y){
        for(x = 0; x < input_width; ++x){
          dst_pt = (input_height * input_width * (input_depth-d-1)) + (input_width * (input_width-y-1)) + x;
          src_pt = (input_width * input_depth * y) + (input_depth * x) + d;
          cnn_layer_image[0].input_data[dst_pt] =
           (myFixed)(image->data[src_pt] / 256.0);
        }
      }
    }
    st = getusage(); 

#if 0
  out = execCNN2(
    cnn_layer_image[0].filter,
    cnn_layer_image[0].filter_num,
    cnn_layer_image[0].filter_size,
    cnn_layer_image[0].input_data,
    cnn_layer_image[0].input_width,
    cnn_layer_image[0].input_height,
    cnn_layer_image[0].input_depth,
    cnn_layer_image[0].conv_out, 
    cnn_layer_image[0].conv_width,
    cnn_layer_image[0].conv_height,
    cnn_layer_image[0].pool_out,
    cnn_layer_image[0].pool_width,
    cnn_layer_image[0].pool_height,
    cnn_layer_image[0].pool_size,
    cnn_layer_image[0].pool_stride,

    cnn_layer_image[1].filter,
    cnn_layer_image[1].filter_num,
    cnn_layer_image[1].filter_size,
    cnn_layer_image[1].input_data,
    cnn_layer_image[1].input_width,
    cnn_layer_image[1].input_height,
    cnn_layer_image[1].input_depth,
    cnn_layer_image[1].conv_out, 
    cnn_layer_image[1].conv_width,
    cnn_layer_image[1].conv_height,
    cnn_layer_image[1].pool_out,
    cnn_layer_image[1].pool_width,
    cnn_layer_image[1].pool_height,
    cnn_layer_image[1].pool_size,
    cnn_layer_image[1].pool_stride,

    cnn_layer_image[2].filter,
    cnn_layer_image[2].filter_num,
    cnn_layer_image[2].filter_size,
    cnn_layer_image[2].input_data,
    cnn_layer_image[2].input_width,
    cnn_layer_image[2].input_height,
    cnn_layer_image[2].input_depth,
    cnn_layer_image[2].conv_out, 
    cnn_layer_image[2].conv_width,
    cnn_layer_image[2].conv_height,
    cnn_layer_image[2].pool_out,
    cnn_layer_image[2].pool_width,
    cnn_layer_image[2].pool_height,
    cnn_layer_image[2].pool_size,
    cnn_layer_image[2].pool_stride,

    weight_hidden,
    weight_out
  );
#else

    execCNN(
      cnn_layer_image[0].filter, cnn_layer_image[0].filter_num, cnn_layer_image[0].filter_size,
      cnn_layer_image[0].input_data, 
      cnn_layer_image[0].input_width, cnn_layer_image[0].input_height, cnn_layer_image[0].input_depth,
      cnn_layer_image[0].conv_out, 
      cnn_layer_image[0].conv_width, cnn_layer_image[0].conv_height,
      cnn_layer_image[0].pool_out, 
      cnn_layer_image[0].pool_width, cnn_layer_image[0].pool_height, 
      cnn_layer_image[0].pool_size, cnn_layer_image[0].pool_stride,

      cnn_layer_image[1].filter, cnn_layer_image[1].filter_num, cnn_layer_image[1].filter_size,
      cnn_layer_image[1].input_data,
      cnn_layer_image[1].input_width, cnn_layer_image[1].input_height, cnn_layer_image[1].input_depth,
      cnn_layer_image[1].conv_out,
      cnn_layer_image[1].conv_width, cnn_layer_image[1].conv_height,
      cnn_layer_image[1].pool_out,
      cnn_layer_image[1].pool_width, cnn_layer_image[1].pool_height, 
      cnn_layer_image[1].pool_size, cnn_layer_image[1].pool_stride,

      cnn_layer_image[2].filter, cnn_layer_image[2].filter_num, cnn_layer_image[2].filter_size,
      cnn_layer_image[2].input_data,
      cnn_layer_image[2].input_width, cnn_layer_image[2].input_height, cnn_layer_image[2].input_depth,
      cnn_layer_image[2].conv_out, 
      cnn_layer_image[2].conv_width, cnn_layer_image[2].conv_height,
      cnn_layer_image[2].pool_out,
      cnn_layer_image[2].pool_width, cnn_layer_image[2].pool_height, 
      cnn_layer_image[2].pool_size, cnn_layer_image[2].pool_stride
    );

    et = getusage(); 
      middle += et -st;

    pool_out_num = cnn_layer_image[CNN_LAYER_NUM-1].pool_width *
                   cnn_layer_image[CNN_LAYER_NUM-1].pool_height *
                   cnn_layer_image[CNN_LAYER_NUM-1].pool_depth;

    out = Forward( cnn_layer_image[CNN_LAYER_NUM-1].pool_out, pool_out_num, weight_hidden, weight_out, 
          HIDDEN_NUM, OUT_NUM);

#endif

    et = getusage(); 
      usage += et -st;
      ++usage_count;

    FreeImg(image);

    printf("[Answer] %d\n", out);
  }
    printf("ConvUsageTIme: %6.3lf[ms], TotalUsageTIme: %6.3lf[ms]\n", (middle/usage_count), (usage/usage_count));

  FreeCNNLayerImage(cnn_layer_image);

  return 0;
}

/*
   繝｡繧､繝ｳ髢｢謨ｰ
 */
int main(int argc, char **argv)
{
  fcFixed *weight_hidden;          
  fcFixed *weight_out;             
  int num_of_input_data = 0;      
  char *filename;                 
  FILE *fp;                       
  InputList input_list[MAX_LIST]; 

  printf("CNN - Start\n");

  filename = list_test;
  printf("Mode: Test\n");
  printf("List File: %s\n", filename);

  weight_hidden = (fcFixed *)sds_alloc(sizeof(fcFixed) * HIDDEN_NUM * POOL_OUT_NUM);
  weight_out    = (fcFixed *)sds_alloc(sizeof(fcFixed) * HIDDEN_NUM * OUT_NUM);

  if((fp=fopen("/mnt/params/weight_hidden.dat", "r+")) != NULL){
      fread(weight_hidden, sizeof(fcFixed) * HIDDEN_NUM * (POOL_OUT_NUM), 1, fp);
    }
  fclose(fp);

  if((fp=fopen("/mnt/params/weight_out.dat", "r+")) != NULL){
    fread(weight_out, sizeof(fcFixed) * (HIDDEN_NUM) * (OUT_NUM), 1, fp);
  }
  fclose(fp);

  if((fp=fopen(filename, "r")) != NULL){
    while( fscanf(fp,"%s %d",
                  &input_list[num_of_input_data].name[0],
                  &input_list[num_of_input_data].teacher) != EOF
    ){
      ++num_of_input_data;
    }
  }
  fclose(fp);
  
  printf("Num of Input Data: %d\n", num_of_input_data);

  CNN(input_list, num_of_input_data, weight_hidden, weight_out);

  sds_free(weight_hidden);
  sds_free(weight_out);

  return 0;
}


