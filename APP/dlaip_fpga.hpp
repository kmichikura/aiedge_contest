#ifndef DLAIP_FPGA_HPP
#define DLAIP_FPGA_HPP

#include <string>
#include <iostream>
#include <sys/time.h>                   // for gettimeofday()
#include <stdio.h>                      // for printf()
#include <stdarg.h>  // for variab
#include <vector>

#include "ui.hpp"
#include "dtype.hpp"
using std::string;
using std::vector;

#define MCORE 1

#define OFF_VERS0        (0x000)
#define OFF_VERS1        (0x004)
#define OFF_VERS2        (0x008)
#define OFF_VERS3        (0x00C)

#define OFF_START        (0x010)
#define OFF_BUSY         (0x014)
#define OFF_OPCODE       (0x01C)
#define OFF_RESUME       (0x020)
#define OFF_GLB          (0x024)
#define OFF_TEMP         (0x028)
#define OFF_OUT0         (0x02C)
#define OFF_OUT1         (0x030)
#define OFF_INP          (0x034)
#define OFF_PRM          (0x038)

#define VAL_START        (1)
#define VAL_BUSY         (1)



#define restrict


// ################################# For FPGA ###############################
/// common data for all layers
class baseLayer
{
public:
    std::string name;
    baseLayer(const char *name_, int ic_ = 1, int bs_ = 1) : 
        name(name_),
        ic(ic_),
        bs(bs_)
    { in_size.x = 1; in_size.y = 1; }
    baseLayer(const char *name_, intXY in_size_, int ic_ = 1, int bs_ = 1) : 
        name(name_),
        in_size(in_size_),
        ic(ic_),
        bs(bs_)
    {}
    int   getInWidth(void)    { return in_size.x; }
    int   getInHeight(void)   { return in_size.y; }
    int   getInChannels(void) { return ic; }
    int   getBatchSize(void)  { return bs; }
    int   getInSize(void)     { return getInWidth() * getInHeight() * getInChannels(); }
    intXY getInWH(void)       { return in_size; }

private:
    intXY in_size;                      // input image width, height
    int ic;                             // input image channels
    int bs;                             // batch size
};

class FPGA : public baseLayer        // baseLayer : for dumpHostMem()
{
public:
    FPGA(int w, int h, int ch, int onum, int outch, int dbufoff);
    ~FPGA();
    void dumpHostMem(const DType* pf, const int n, const char *filename);
    void setInputData(const DType *pf);               /// set input data
    void setInputPtr(DType **pf, int index);              /// set input data
    void LoadInput(int index);
    int ForwardPropagation(int index);   /// Forward propagation
    int getOutSize(int i);
    int getOutWithPadSize(int i);
    int getMaxOutWithPadSize(void);
    int getOutputData(DType *p0, DType *p1);  /// readout result
    void getOutputPtr(DType **p0, DType **p1, int index);  /// readout result
    void LoadOutput(int index);
    float get_i2f_scale0(int index);  /// get i2f_scale0[index]
    int getIncPad(void);
    void load_parameter_files(void);
    vector<float>  in_align_time, in_set_time;
    vector<float>  hw_exec_time;
    vector<float>  out0_align_time, out0_set_time;
    vector<float>  out1_align_time, out1_set_time;

protected:
    // buffer for forward propagation

    DType *ibuf;
    DType *ibuf0, *ibuf1;
    DType *obuf;
    DType *obuf0, *obuf1;
    DType *obuf00, *obuf01;
    DType *obuf10, *obuf11;

private:
    int width;                                  /// input data Horizontal size
    int height;                                 /// input data Vertical size
    int channels;                               /// input data number of channels
    int out_num;
    int input_adr;
    int input0_adr;
    int input1_adr;
    int output0_adr;
    int output1_adr;
    int output00_adr;
    int output01_adr;
    int output10_adr;
    int output11_adr;
    int param_adr;
    int temp_adr;
    int dsize;
    int inc_pad;
    int out0_w, out0_h, out0_ch;
    int out1_w, out1_h, out1_ch;
    int out0_ch_w_pad;
    int out1_ch_w_pad;
    int outsize0;
    int outsize1;
    int outsize0_w_pad;
    int outsize1_w_pad;
    int dbufoff;

};


#endif  // DLAIP_FPGA_HPP
