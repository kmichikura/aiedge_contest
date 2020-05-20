#ifndef __DTYPE_H__
#define __DTYPE_H__

///// dtype.hpp /////
typedef signed  char DType;    //
typedef signed  char PType;    // parameter file data type (int8)
typedef int   CType;    // 
typedef float RType;    //

typedef struct {
    int x;
    int y;
} intXY;

#define FIXED_MAX     ((1<<(FIXED_NBIT-1))-1)             // 固定小数点での最大値
#define FIXED_MIN     (-1*FIXED_MAX)                      // 固定小数点での最小値
#define FIXED_MAX_QAT (1<<(FIXED_NBIT-1))                 // 固定小数点での最大値


#endif  //__DTYPE_H__
