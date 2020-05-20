#ifndef __DETECTOR_H__
#define __DETECTOR_H__
#include <vector>

typedef struct{
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

typedef struct layer{
     int batch;
     int outputs;
     int w;
     int h;
     int n;
     int classes;
     int *mask;
     float *biases;
     float *output;
} layer;


layer *make_layer(float *output, int batch, int outputs, int w, int h, int layernum, int classes);
//detection *get_layer_boxes(int netw, int neth, layer l, int w, int h, float thresh, int *num);
detection *get_layer_boxes(int netw, int neth, layer l, float thresh, int *num);
void do_nms_sort(detection *dets, int total, int classes, float thresh);
void do_nms_sort_v(std::vector<detection> &dets, int total, int classes, float thresh);
void free_detections(detection *dets, int n);

#endif // __DETECTOR_H__
