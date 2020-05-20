#include "dlaip_fpga.hpp"
#include <string>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>
//#include <sys/stat.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <sys/time.h>                   // for gettimeofday()
#include <numeric>
#include <thread>

#include "opencv2/opencv.hpp"
#include "cmdline.h"                    // command line parser (ref. https://github.com/tanakh/cmdline)
#include "detector.hpp"
#include "ui.hpp"

/// command line parser
cmdline::parser cmdparser;
int verbose;
StopWatch stopwatch;
#define FILENAME_INPUT ""
std::string imageDir(FILENAME_INPUT);
#define DEFAULT_WSIZE 1936
#define DEFAULT_HSIZE 1216
static float thresh;
static float nms;
static int   imw = 1936;
static int   imh = 1094;
static int   classes = 6;
static int   num = -1;
#define OUTPUT_DIR "./dump_detect/"
static int netw = 512;
static int neth = 288;
static int outc = 33;
static int i2fsel = 0;
static int fixedmax = FIXED_MAX_QAT;
bool valid = false;
static int batch = 20;

void scanarg(int argc, char *argv[])
{
    cmdparser.add<int>        ("verbose",  'v', "verbose level",     false, V_MESSAGE);
    cmdparser.add<std::string>("image",    'i', "image directory name",   false, FILENAME_INPUT);
    cmdparser.add<float>      ("thresh",   'T', "thresh detection",  false, 0.5);
    cmdparser.add<float>      ("nms",      'n', "nms value",         false, 0.45);
    cmdparser.add<int>        ("exenum",   'e', "number of execute", false, -1);
    cmdparser.add<int>        ("batchnum", 'b', "number of batch", false, 20);
    cmdparser.add             ("valid",    'd', "validation");
    if (!cmdparser.parse(argc, argv)) {
        std::cout << cmdparser.error_full() << cmdparser.usage();
        exit(1);
    }
    verbose   = cmdparser.get<int>("verbose");
    imageDir  = cmdparser.get<std::string>("image");
    thresh    = cmdparser.get<float>("thresh");
    nms       = cmdparser.get<float>("nms");
    num       = cmdparser.get<int>("exenum");
    batch     = cmdparser.get<int>("batchnum");

    if (cmdparser.exist("valid")) valid = true;

    Trace(V_MESSAGE, "verbose : %d", verbose);
    Trace(V_MESSAGE, "image   : %s", imageDir.c_str());
    Trace(V_MESSAGE, "thresh  : %f", thresh);
    Trace(V_MESSAGE, "nms     : %f", nms);
    Trace(V_MESSAGE, "exenum  : %d", num);
    Trace(V_MESSAGE, "bacchnum: %d", batch);
}


typedef struct {
    float label;
    float conf;
    float x_min;
    float y_min;
    float x_max;
    float y_max;
} result;

bool result_compare( const result& left, const result& right ) {
    return left.label == right.label ? left.conf > right.conf : left.label < right.label;
}
void create_results(vector<detection> &dets, int classes, int total, float thresh, int imw, int imh, vector<result> &results) {

    result res;
    int count[classes];
    for (int i = 0; i < classes; i++) count[i] = 0;

    for(int i = 0; i < total; ++i){
        char labelstr[4096] = {0};
        int classn = -1;
        for(int j = 0; j < classes; ++j){
            if (dets[i].prob[j] > thresh){
                box b = dets[i].bbox;
                int left  = ((b.x-b.w/2.) < 0 ? 0 : (b.x-b.w/2.))*imw;
                int right = ((b.x+b.w/2.) > DEFAULT_WSIZE/imw ? DEFAULT_WSIZE/imw : (b.x+b.w/2.))*imw;
                int top   = ((b.y-b.h/2.) < 0 ? 0 : (b.y-b.h/2.))*imh;
                int bot   = ((b.y+b.h/2.) > DEFAULT_HSIZE/imh ? DEFAULT_HSIZE/imh : (b.y+b.h/2.))*imh;
                res.label = j;
                res.conf = dets[i].prob[j];
                res.x_min = left;
                res.y_min = top;
                res.x_max = right;
                res.y_max = bot;
                results.push_back(res);
                count[j]++;
            }
        }
    }
    sort(results.begin(), results.end(), result_compare);
    int sp = 0;
    int max_detect = 99;
    for (int i = 0; i < classes; i++) {
        if (count[i] <= max_detect) {
            sp += count[i];
        }
        else {
            results.erase(results.begin()+sp+max_detect, results.begin()+sp+count[i]);
            sp += max_detect;
        }
    }
}

void print_results(vector<result> &results) {

    std::string fnames = "aiedge.names";

    std::ifstream ifs(fnames);

    char str[256];
    char *names[11];
    for (int i = 0; i < (classes+1); i++) {
        names[i] = (char*)calloc(256, sizeof(char));
    }
    int i = 0;
    while (ifs.getline(str, 256 - 1))
    {
        strcpy(names[i], str);
        i++;
    }

    printf("    Object Conf.   left  top right bottom\n");
    for (int i = 0; i < results.size(); i++) {
        printf("%s %3.f%% :[ %4d %4d  %4d %4d ]\n", 
            names[(int)results[i].label], results[i].conf*100 ,
            (int)results[i].x_min, (int)results[i].y_min, (int)results[i].x_max, (int)results[i].y_max);
    }
    printf("========================================\n");
}

void output_results(vector<result> &results, std::string fname) {

    std::fstream file;
    file.open(fname, std::ios::binary | std::ios::out);

    for (int i = 0; i < results.size(); i++) {
        file.write((char*)&results[i], sizeof(result));
    }
    file.close();
}

float* i2f(DType *data, int size, float scale) {
	
    float *dst = (float *)calloc(size, sizeof(float));

    for (int i = 0; i < size; ++i) {
        dst[i] = (float)data[i] * scale / FIXED_MAX;
    }

    return dst;
}


bool getFileNames(std::string folderPath, vector<std::string> &file_names)
{
    namespace fs = std::experimental::filesystem ;
    fs::directory_iterator iter(folderPath), end;

    for (; iter != end; iter++) {
        const fs::directory_entry entry = *iter;

        file_names.push_back( entry.path().string() );
    }

    return true;
}



std::string getFileName(std::string fullpath) {
    int path_i = fullpath.find_last_of("/")+1;
    int ext_i  = fullpath.find_last_of(".");

    std::string pathname = fullpath.substr(0,path_i+1);
    std::string filename = fullpath.substr(path_i,ext_i-path_i);
    
    return filename;
}

std::string getFileExt(std::string fullpath) {
    int path_i = fullpath.find_last_of("/")+1;
    int ext_i  = fullpath.find_last_of(".");

    std::string extname  = fullpath.substr(ext_i,fullpath.size()-ext_i); 
    
    return extname;
}

vector<detection> get_detection(DType *top0_data, int size_top0, int out0_w, int out0_h, float scale0, 
                         DType *top1_data, int size_top1, int out1_w, int out1_h, float scale1, 
                         int *total)
{
    int nboxes0 = 0;
    float *top0_data_f;
    long long stime, etime;
    top0_data_f = i2f(top0_data, size_top0, scale0);
    layer *l0 =  make_layer(top0_data_f, 0, size_top0, out0_w, out0_h, 0, classes);
    detection *dets0 = get_layer_boxes(netw, neth, *l0, thresh, &nboxes0); 

    int nboxes1 = 0;
    float *top1_data_f;
    top1_data_f = i2f(top1_data, size_top1, scale1);
    layer *l1 =  make_layer(top1_data_f, 0, size_top1, out1_w, out1_h, 1, classes);
    detection *dets1 = get_layer_boxes(netw, neth, *l1, thresh, &nboxes1);
    
    *total = nboxes0 + nboxes1;
    vector<detection> dets_v;
    for (int i = 0; i < nboxes0; ++i) 
        dets_v.push_back(dets0[i]);
    for (int i = 0; i < nboxes1; ++i)
        dets_v.push_back(dets1[i]);

    free(top0_data_f);
    free(top1_data_f);
    free(dets0);
    free(dets1);

    return dets_v;
}

void mat2dt_w_norm_and_pad(cv::Mat m, DType* dst, int fixedmax, int netw, int neth, int incpad) {
    int h = neth;
    int w = netw;
    int c;
    if (m.type() == CV_8UC3) {
        c = 3;
    } else {
        printf("Not Support type\n");
    }
    int i;
    #pragma omp parallel for
    for (i = 0; i < h; i++) {
        cv::Vec3b *src = m.ptr<cv::Vec3b>(i);   
        for (int j = 0; j < w; j++) {
            cv::Vec3b rgb = src[j];
            for(int k = 0; k < c; k++) {
                int ofs = (c+incpad) * w * i + (c+incpad) * j + k;
                DType norm;
                norm = (DType)((float)rgb(2-k)-fixedmax);
                dst[ofs] = norm;
            }
        }
    }

}

void preProcess(const bool flg, cv::Mat &img, DType* data, const int resize_w, const int resize_h, const int pad, FPGA* pfpga, const int id, float *time) {
    if (flg) {
        double stime = stopwatch.getTime();
        cv::Mat resize_img;
        cv::resize(img, resize_img, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_NEAREST);
        mat2dt_w_norm_and_pad(resize_img, data, fixedmax, netw, neth, pad);
        pfpga->LoadInput(id);
        double etime = stopwatch.getTime();
        *time        = (float)(etime-stime);
    }
}

void ForwardPropagation(const bool flg, FPGA* pfpga, const int id, float* time) {
    if (flg) {
        double stime = stopwatch.getTime();
        pfpga->ForwardPropagation(id);
        double etime = stopwatch.getTime();
        *time        = (float)(etime-stime);
    }
}


void postProcess(const bool flg, FPGA* pfpga, const int id,
                 DType* top0_data, const int size_top0, const int out0_w, const int out0_h, const float i2f_scale_out0,
                 DType* top1_data, const int size_top1, const int out1_w, const int out1_h, const float i2f_scale_out1,
                 const std::string fname, float* time)
{
    if (flg) {
        pfpga->LoadOutput(id);
        double stime = stopwatch.getTime();
        int total;
        vector<detection> dets_v;
        vector<result> results;
        dets_v = get_detection(top0_data, size_top0, out0_w, out0_h, i2f_scale_out0,
                               top1_data, size_top1, out1_w, out1_h, i2f_scale_out1,
                               &total);
        
        if (nms) do_nms_sort_v(dets_v, total, classes, nms);

        create_results(dets_v, classes, total, thresh, imw, imh, results);

        if(valid) output_results(results, OUTPUT_DIR + getFileName(fname) + ".bin");
        else print_results(results);
        double etime = stopwatch.getTime();
        *time        = (float)(etime-stime);
    }
}

int main(int argc, char** argv)
{
    scanarg(argc, argv);
    int INPUT_CHANNELS = 3 ;
    int OUTPUT_NUM = 2 ;
    int RESIZE_WIDTH = 512 ;
    int RESIZE_HEIGHT = 320 ;

    const int out0_w = netw/32, out0_h = neth/32, out0_ch = outc;
    const int out1_w = netw/16, out1_h = neth/16, out1_ch = outc;
    const int size_top0 = out0_h * out0_w * out0_ch ;
    const int size_top1 = out1_h * out1_w * out1_ch ;

    DType* top0_data;
    DType* top1_data;
    DType* top00_data;
    DType* top01_data;
    DType* top10_data;
    DType* top11_data;

    long long stime,etime;

    FPGA* pfpga = nullptr;
    pfpga = new FPGA(netw, neth, INPUT_CHANNELS, OUTPUT_NUM, outc, 0x1200000);

    vector<std::string> file_names;
    getFileNames(imageDir, file_names );
    int insize = RESIZE_WIDTH * RESIZE_WIDTH * pfpga->getIncPad();
    DType *data;
    DType *data0 = (DType *)malloc(insize * sizeof(DType));
    DType *data1 = (DType *)malloc(insize * sizeof(DType));
    if (!std::experimental::filesystem::exists(OUTPUT_DIR)) std::experimental::filesystem::create_directory(OUTPUT_DIR);

    num = (num == -1 || file_names.size() < num) ? file_names.size() : num;
    float *itime ;
    float *ptime ;
    float *rtime ;
    float *otime ;
    itime = new float[num];
    ptime = new float[num];
    rtime = new float[num];
    otime = new float[num];
    vector<float> atime;

    pfpga->setInputPtr(&data0, 0);
    pfpga->setInputPtr(&data1, 1);
    pfpga->getOutputPtr(&top00_data, &top01_data, 0);
    pfpga->getOutputPtr(&top10_data, &top11_data, 1);

    vector<cv::Mat> img_v;
    int multinum;
    for (int i = 0; i < num; i+=multinum) {
        multinum = (num-i) > batch ? batch : (num-i);

        for (int n = 0; n < multinum; n++) {
            stime = stopwatch.getTime();
            img_v.push_back(cv::imread(file_names[i+n]));
            etime = stopwatch.getTime();
            itime[i+n] = (float)(etime-stime);
            printf("%04d / %04d %s \n", i+n, file_names.size(), file_names[i+n].c_str());
            fflush(stdout);
        }

        int pre_count  = 0;
        int run_count  = 0;
        int post_count = 0;

        while (pre_count < multinum || run_count < multinum || post_count < multinum) {

            stime = stopwatch.getTime();
            if ((pre_count%2) == 0) data = data0;
            else                    data = data1;
            bool pflg = (pre_count < multinum);
            std::thread th_pre (preProcess, pflg, std::ref(img_v[pre_count]), data, RESIZE_WIDTH, RESIZE_HEIGHT, pfpga->getIncPad(), pfpga, (pre_count%2), &ptime[i+pre_count]);

            bool rflg = (pre_count != 0 && run_count < multinum);
            std::thread th_run (ForwardPropagation, rflg, pfpga, (run_count%2), &rtime[i+run_count]);

            if ((post_count%2) == 0) {
                top0_data = top00_data;
                top1_data = top01_data;
            }
            else {
                top0_data = top10_data;
                top1_data = top11_data;
            }
            bool oflg = (run_count != 0 && post_count < multinum);
            std::thread th_post(postProcess, oflg, pfpga, (post_count%2),
                                             top0_data, size_top0, out0_w, out0_h, pfpga->get_i2f_scale0(i2fsel),
                                             top1_data, size_top1, out1_w, out1_h, pfpga->get_i2f_scale0(i2fsel + 1),
                                             file_names[i+post_count], &otime[i+post_count]);

            th_pre.join();
            th_run.join();
            th_post.join();
             
            if (pflg) pre_count++;
            if (rflg) run_count++;
            if (oflg) post_count++;
            etime = stopwatch.getTime();
            atime.push_back((float)(etime-stime));
        }

        img_v.clear();
    }

    float sum_itime=0 ;
    float sum_ptime=0 ;
    float sum_rtime=0 ;
    float sum_otime=0 ;
    float sum_atime=0 ;
    for (int i=0 ; i < num ; i++){
      Trace(V_DEBUG,"Image Read    TIME[%d]=%0.3fms\n",i,itime[i]/1000);
      sum_itime += itime[i];
      Trace(V_DEBUG,"Pre Process TIME[%d]=%0.3fms\n",i,ptime[i]/1000);
      sum_ptime += ptime[i];
      Trace(V_DEBUG,"RUN          TIME[%d]=%0.3fms\n",i,rtime[i]/1000);
      sum_rtime += rtime[i];
      Trace(V_DEBUG,"Post Process TIME[%d]=%0.3fms\n",i,otime[i]/1000);
      sum_otime += otime[i];
    }

    for (int i=0 ; i < atime.size(); i++) {
        sum_atime += atime[i];
    }

    printf("Average Image Read    TIME=%0.3fms\n",sum_itime/num/1000);
    printf("Average Pre Procoess  TIME=%0.3fms\n",sum_ptime/num/1000);
    printf("Average RUN           TIME=%0.3fms\n",sum_rtime/num/1000);
    printf("Average Post Process  TIME=%0.3fms\n",sum_otime/num/1000);
    printf("Average ALL  Process  TIME=%0.3fms\n",sum_atime/num/1000);
    free(pfpga);
    free(data);
    delete [] top0_data;
    delete [] top1_data;

    return 0;
  
}
