#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
//#include <sys/time.h>
//#include <sys/resource.h>
#include "common.hpp"

/*
 * Sigmoid Function.
 */
#if 1
void show_img(
  myFixed *input_data, // 入力データ
  int input_width,    // 入力データの幅
  int input_height,   // 入力データの高さ
  int input_depth    // 入力データの深さ
){
    
  int m = 0, n = 0, d = 0;
  int offset_i; // Channel offset of filter.
  double sum = 0; // 総数の値
  for(d = 0; d < input_depth; ++d){
    offset_i = (input_width * input_height * d);  // 入力データのオフセット
    printf("Depth %d\n",d);
    for(n = 0; n < input_width; ++n){
      for(m = 0; m < input_height; ++m){
        sum = input_data[offset_i + ((n) * input_width) + (m)]; 
        printf("%lf,",(double)sum);
      }
      printf("\n");
    }
  }
};

void show_conv_filter(
  myFixed *filter,     // フィルタ
  int filter_size,    
  int filter_depth,    
  int filter_num     // ふぃるたの数
  ) {
  int m = 0, n = 0,  d = 0, x = 0;
  myFixed sum = 0; // 総数の値
  int offset_f, offset_n; // Channel offset of filter.

  for(x = 0; x < filter_num; ++x){
    offset_n = (filter_size * filter_size * filter_depth * x);  // Offset of filter.
    printf("Num(%d)\n",x);
    for(d = 0; d < filter_depth; ++d){
      offset_f = (filter_size * filter_size * d);  // Offset of filter.
      printf("Depth(%d)\n",d);
      for(n = 0; n < filter_size; ++n){
        for(m = 0; m < filter_size; ++m){
          sum = filter[offset_n + offset_f + (n * filter_size) + m];
          printf("%lf,",(double)sum);
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
};

void show_fc(
  fcFixed *weight,     // フィルタ
  int output_num,    // フィルタのサ 
  int input_num    // フィルタのサ 
  ){
  int n = 0, m = 0;
  fullBit sum = 0; // 総数の値

    for(n = 0; n < output_num; ++n){
      for(m = 0; m < input_num; ++m){
        sum = weight[((n) * input_num) + (m)]; 
        printf("%.8lf,",(double)sum);
      }
      printf("\n");
    }
}; 
#endif

