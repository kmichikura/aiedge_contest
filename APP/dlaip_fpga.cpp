#include "dlaip_fpga.hpp"
#include <algorithm>
#include <math.h>                       // for log2(), etc.
#include <string.h>                     // for memcpy()
#include <assert.h>                     // for assert()
#include <sys/stat.h>
#include <omp.h>
#include <fcntl.h>
#include <limits.h>

#include "axis.h"

using std::max;
using std::min;

#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wformat-extra-args"

//################################# For FPGA ###################################
const float i2f_scale0[] = {
    15.875,
    15.875
    
};

////////// functions for AXI //////////
///// N Byte read
/** \param  id   /dev/uioX
    \param  addr DDR領域先頭からのオフセット */
template <typename T>
inline T _read(int id, int addr)
{
  T value;
  axis_read(id, addr, &value);
  return value;
}

///// N Byte write
/** \param  id   /dev/uioX
    \param  addr DDR領域先頭からのオフセット 
    \param  data 書き込む値 */
template <typename T>
inline void _write(int id, int addr, T data)
{
  axis_write(id, addr, data);
}

/// PL Register 32bit read
/** \param  addr Registerアドレス */
inline int reg_read(int addr)
{
    return _read<int>(1, addr);  // PL registerは /dev/uio0 経由
}

/// PL Register 32bit write
/** \param  addr Registerアドレス
    \param  data 書き込む値 */
inline void reg_write(int addr, int data)
{
    _write(1, addr, data);  // PL registerは /dev/uio0 経由
}


static int calc_channel_pad(int ch) {
    int pnum = (AXI_NBIT/8) / sizeof(DType) ;  // AXI 1burst当たりのdata数
    Trace(V_TRACE, "calc_channel_pad %d ",  (ch % pnum == 0) ? 0 : pnum - (ch % pnum));
    return (ch % pnum == 0) ? 0 : pnum - (ch % pnum);
}


static void alignment_output(DType *idata, DType *odata,
		int in_x, int in_y, int in_ch )
{
  int ofs=0;
  int inc_pad = calc_channel_pad(in_ch);
#if MCORE		// 4 core
#pragma omp parallel
#pragma omp sections
{
#pragma omp section
 {
  for ( int y = 0 ; y < in_y/4 ; y++ ){
    for ( int x = 0 ; x < in_x ; x++ ){
      for ( int ch = 0 ; ch < in_ch ; ch++ ){
        odata[x+in_x*y+in_x*in_y*ch] = idata[ch+(x*(in_ch+inc_pad))+(y*in_x*(in_ch+inc_pad))];
      }
    }
  }
 }
#pragma omp section
 {
  for ( int y = in_y/4 ; y < in_y/4*2 ; y++ ){
    for ( int x = 0 ; x < in_x ; x++ ){
      for ( int ch = 0 ; ch < in_ch ; ch++ ){
        odata[x+in_x*y+in_x*in_y*ch] = idata[ch+(x*(in_ch+inc_pad))+(y*in_x*(in_ch+inc_pad))];
      }
    }
  }
 }
#pragma omp section
 {
  for ( int y = in_y/4*2 ; y < in_y/4*3 ; y++ ){
    for ( int x = 0 ; x < in_x ; x++ ){
      for ( int ch = 0 ; ch < in_ch ; ch++ ){
        odata[x+in_x*y+in_x*in_y*ch] = idata[ch+(x*(in_ch+inc_pad))+(y*in_x*(in_ch+inc_pad))];
      }
    }
  }
 }
#pragma omp section
 {
  for ( int y = in_y/4*3 ; y < in_y ; y++ ){
    for ( int x = 0 ; x < in_x ; x++ ){
      for ( int ch = 0 ; ch < in_ch ; ch++ ){
        odata[x+in_x*y+in_x*in_y*ch] = idata[ch+(x*(in_ch+inc_pad))+(y*in_x*(in_ch+inc_pad))];
      }
    }
  }
 }
}
#else		// 1 core
  for ( int y = 0 ; y < in_y ; y++ ){
    for ( int x = 0 ; x < in_x ; x++ ){
      for ( int ch = 0 ; ch < in_ch ; ch++ ){
        odata[x+in_x*y+in_x*in_y*ch] = idata[ch+(x*(in_ch+inc_pad))+(y*in_x*(in_ch+inc_pad))];
      }
    }
  }
#endif
}
void load_output(DType *out, int inadr, int size) {
  typedef unsigned long long Ull;
  typedef long long        Dll;

  int loop;
  Ull src;

  src = (Ull)out;
  loop = size/(sizeof(Dll)/sizeof(DType));
#if MCORE		// 4 core
#pragma omp parallel
#pragma omp sections
{
#pragma omp section
 {
  for(int i=0; i<loop/4; i++){
    volatile Dll r = *(volatile Dll*)(axis_ptr[0] + (inadr + i * sizeof(Dll)));
    *((Dll*)src+i) = r ;
  }
 }
#pragma omp section
 {
  for(int i=loop/4; i<loop/4*2; i++){
    volatile Dll r = *(volatile Dll*)(axis_ptr[0] + (inadr + i * sizeof(Dll)));
    *((Dll*)src+i) = r ;
  }
 }
#pragma omp section
 {
  for(int i=loop/4*2; i<loop/4*3; i++){
    volatile Dll r = *(volatile Dll*)(axis_ptr[0] + (inadr + i * sizeof(Dll)));
    *((Dll*)src+i) = r ;
  }
 }
#pragma omp section
 {
  for(int i=loop/4*3; i<loop; i++){
    volatile Dll r = *(volatile Dll*)(axis_ptr[0] + (inadr + i * sizeof(Dll)));
    *((Dll*)src+i) = r ;
  }
 }
}
#else		// 1 core
  for(int i=0; i<loop; i++){
    volatile Dll r = *(volatile Dll*)(axis_ptr[0] + (inadr + i * sizeof(Dll)));
    *((Dll*)src+i) = r ;
  }
#endif
  return;
}

void load_input(DType *in, int outadr, int size) {
  typedef unsigned long long Ull;
  typedef long long        Dll;

  int loop;
  Ull src;

  src = (Ull)in;
  loop = size/(sizeof(Dll)/sizeof(DType));
#if MCORE		// 4 core
#pragma omp parallel
#pragma omp sections
{
#pragma omp section
 {
  for(int i=0; i<loop/4; i++){
    *(volatile Dll*)(axis_ptr[0] + (outadr + i * sizeof(Dll))) = *((Dll*)src+i);
  }
 }
#pragma omp section
 {
  for(int i=loop/4; i<loop/4*2; i++){
    *(volatile Dll*)(axis_ptr[0] + (outadr + i * sizeof(Dll))) = *((Dll*)src+i);
  }
 }
#pragma omp section
 {
  for(int i=loop/4*2; i<loop/4*3; i++){
    *(volatile Dll*)(axis_ptr[0] + (outadr + i * sizeof(Dll))) = *((Dll*)src+i);
  }
 }
#pragma omp section
 {
  for(int i=loop/4*3; i<loop; i++){
    *(volatile Dll*)(axis_ptr[0] + (outadr + i * sizeof(Dll))) = *((Dll*)src+i);
  }
 }
}
#else		// 1 core
  for(int i=0; i<loop; i++){
    *(volatile Dll*)(axis_ptr[0] + (outadr + i * sizeof(Dll))) = *((Dll*)src+i);
  }
#endif
  return;
}

void FPGA::load_parameter_files(void)
{
    long file_size;
    int fd;
    struct stat stbuf;

    fd = open("./all_prm.bin",O_RDONLY);
    FILE *fp = fdopen(fd, "rb");
    if (fstat(fd , &stbuf) == -1) {
        printf( "Error (Not File Open)\n" );
    }
    file_size = stbuf.st_size;

    PType *dst = (PType *)malloc(file_size);;
    int n = fread(dst, 1, file_size, fp);
    if (n ==! file_size) {
        printf( "Error (File Size Error)\n" );
    }
    int adr = reg_read(OFF_PRM);
    printf( "parm_size=%d parm_addr=%d\n", n , adr);

    load_input(dst, adr, file_size/(sizeof(DType)));

    free(dst);
}


FPGA::FPGA(int w, int h, int ch, int onum, int outch, int dbufoff) :
    baseLayer{"fpga"},
    width(w), 
    height(h), 
    channels(ch),
    out_num(onum),
    out0_ch(outch),
    out1_ch(outch),
    dbufoff(dbufoff)
{
    axis_open();
    // set PL registers
    reg_write(OFF_GLB, 0x40000000);

    // version reg check
    Trace(V_MESSAGE, "VERS0=%lx", reg_read(OFF_VERS0) );
    Trace(V_MESSAGE, "VERS1=%lx", reg_read(OFF_VERS1) );
    Trace(V_MESSAGE, "VERS2=%lx", reg_read(OFF_VERS2) );
    Trace(V_MESSAGE, "VERS3=%lx", reg_read(OFF_VERS3) );
    input0_adr = reg_read(OFF_INP);
    input1_adr = reg_read(OFF_INP) + dbufoff;
    output00_adr = reg_read(OFF_OUT0);
    output01_adr = reg_read(OFF_OUT1);
    output10_adr = reg_read(OFF_OUT0) + dbufoff;
    output11_adr = reg_read(OFF_OUT1) + dbufoff;
    param_adr = reg_read(OFF_PRM);
    temp_adr = reg_read(OFF_TEMP);
    Trace(V_MESSAGE, "input_adr=%d output0_adr=%d output1_adr=%d param_adr=%d temp_adr=%d\n", input_adr , output0_adr , output1_adr , param_adr , temp_adr );
    out0_w = width/32; out0_h = height/32;
    out1_w = width/16; out1_h = height/16;
    out0_ch_w_pad = out0_ch + calc_channel_pad(out0_ch);
    out1_ch_w_pad = out1_ch + calc_channel_pad(out1_ch);
    outsize0 = out0_h * out0_w * out0_ch ;
    outsize1 = out1_h * out1_w * out1_ch ;
    outsize0_w_pad = out0_h * out0_w * out0_ch_w_pad ;
    outsize1_w_pad = out1_h * out1_w * out1_ch_w_pad ;

    inc_pad = calc_channel_pad(ch);
    dsize  = w*h*(ch+inc_pad);
    obuf   = (DType *)malloc(getMaxOutWithPadSize() * sizeof(DType));
    obuf00 = (DType *)malloc(getOutSize(0) * sizeof(DType));
    obuf01 = (DType *)malloc(getOutSize(1) * sizeof(DType));
    obuf10 = (DType *)malloc(getOutSize(0) * sizeof(DType));
    obuf11 = (DType *)malloc(getOutSize(1) * sizeof(DType));
    load_parameter_files();

}

FPGA::~FPGA()
{
    Trace(V_TRACE, "", "");
    free(ibuf);
    free(obuf);
    free(obuf1);
    free(obuf0);
    axis_close();
}

void FPGA::LoadInput(int index) {
    if (index == 0) {
        ibuf      = ibuf0;
        input_adr = input0_adr;
    } else {
        ibuf      = ibuf1;
        input_adr = input1_adr;
    }
    load_input(ibuf , input_adr, dsize);
}

void FPGA::LoadOutput(int index) {
    if (index == 0) {
        obuf0 = obuf00;
        obuf1 = obuf01;
        output0_adr = output00_adr;
        output1_adr = output01_adr;
    } else {
        obuf0 = obuf10;
        obuf1 = obuf11;
        output0_adr = output10_adr;
        output1_adr = output11_adr;
    }
    
    load_output(obuf , output0_adr, getOutWithPadSize(0));
    alignment_output(obuf, obuf0 , out0_w,out0_h,out0_ch);
    load_output(obuf , output1_adr, getOutWithPadSize(1));
    alignment_output(obuf, obuf1 , out1_w,out1_h,out1_ch);

}

/// Forward propagation
int FPGA::ForwardPropagation(int index)
{
    if (index == 0) {
        reg_write(OFF_INP,  input0_adr);
        reg_write(OFF_OUT0, output00_adr) ;
        reg_write(OFF_OUT1, output01_adr) ;
    } else {
        reg_write(OFF_INP,  input1_adr);
        reg_write(OFF_OUT0, output10_adr) ;
        reg_write(OFF_OUT1, output11_adr) ;
    }

    reg_write(OFF_START, VAL_START);
    do {
    } while (reg_read(OFF_BUSY) == 1) ;

    return 0;
}
void FPGA::dumpHostMem(const DType* pf, const int n, const char *filename)
{
    size_t n_written;
    char fname[64];
    sprintf(fname, "dump/%s_%s", name.c_str(), filename);
    FILE *fp = fopen(fname, "wb");
    if (!fp) {
        Trace(V_ERROR, "ERROR: Cannot open file %s", fname);
        exit(1);
    }
    n_written = fwrite(pf, sizeof(DType), n, fp);
    fclose(fp);
    Trace(V_MESSAGE, "wrote %-16s n=%8d n_written=%8d", fname, n, (int)n_written);
}

/// transfer input data to FPGA buffer
void FPGA::setInputData(const DType *pf)
{
    Trace(V_TRACE, "width:%d height:%d channels:%d", width, height, channels);
    memcpy(ibuf0, pf, sizeof(DType) * width * height * channels);
}

void FPGA::setInputPtr(DType **pf, int index)
{
    if (index == 0) ibuf0 = *pf;
    else            ibuf1 = *pf;
}

/// readout result : conv_reg1(A0), conv_reg2(A1), conv_relu2(A2) をコピー
int FPGA::getOutputData(DType *p0, DType *p1)
{
    int outsize = getOutSize(0);
    Trace(V_TRACE, "outsize0=%d", outsize);
    memcpy(p0, obuf0, sizeof(DType) * outsize);

    outsize = getOutSize(1);
    Trace(V_TRACE, "outsize1=%d", outsize);
    memcpy(p1, obuf1, sizeof(DType) * outsize);

    return 0;
}

void FPGA::getOutputPtr(DType **p0, DType **p1, int index)
{
    if (index == 0) {
        *p0 = obuf00;
        *p1 = obuf01;
    } else {
        *p0 = obuf10;
        *p1 = obuf11;
    }
}


/// get output data size
int FPGA::getOutSize(int i)
{
    int outsize = (i == 0) ? outsize0 : outsize1;
    Trace(V_TRACE, "outsize=%d", outsize);
    return outsize;
}

/// get output data size with pad
int FPGA::getOutWithPadSize(int i)
{
    int outsize = (i == 0) ? outsize0_w_pad : outsize1_w_pad ;
    Trace(V_TRACE, "outsize=%d", outsize);
    return outsize;
}

int FPGA::getMaxOutWithPadSize(void) {
    
    int ret = 0;
    
    for(int i = 0; i < out_num; i++) {
        if (ret < getOutWithPadSize(i)) {
            ret = getOutWithPadSize(i);
        }
    }

    return ret;

}

/// get i2f_scale0[index]
float FPGA::get_i2f_scale0(int index)
{
    Trace(V_TRACE, "index=%d scale=%f", index, i2f_scale0[index]);
    return i2f_scale0[index];
}


int FPGA::getIncPad(void)
{
    Trace(V_TRACE, "inc_pad=%d", inc_pad);
    return inc_pad;
}

