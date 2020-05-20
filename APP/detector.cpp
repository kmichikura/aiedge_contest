#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>                   // for gettimeofday()
#include <algorithm>
#include "detector.hpp"
#include "yolo_param.h"

static inline float  sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

layer *make_layer(float *output, int batch, int outputs, int w, int h, int layernum, int classes) {

    layer *l = (layer*)calloc(1, sizeof(layer));

    l->batch   = batch;
    l->outputs = outputs;
    l->w       = w;
    l->h       = h;
    l->n       = w;
    l->n       = sizeof(mask[0])/sizeof(int);
    l->classes = classes;
    l->mask    = (int*)calloc(l->n,   sizeof(int));
    int total  = sizeof(anchors)/sizeof(float);
    for (int i = 0; i < l->n; ++i) {
            l->mask[i] = mask[layernum][i];
    }
    l->biases  = (float*)calloc(total, sizeof(float));
    for (int i = 0; i < total; ++i) {
        l->biases[i]   = anchors[i];
    }
    l->output  = output;

    return l;
};

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

struct nms_comparator_vv {
    bool operator()(const detection& left, const detection& right ) const { return left.prob[right.sort_class] > right.prob[right.sort_class]; }
};

void do_nms_sort_v(std::vector<detection> &dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total-1;
    for(i = 0; i <= k; ++i){
        if(dets[i].objectness == 0){
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k+1;

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            dets[i].sort_class = k;
        }
        sort(dets.begin(), dets.end(), nms_comparator_vv());
        for(i = 0; i < total; ++i){
            if(dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for(int j = i+1; j < total; ++j){
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

static int entry_index(layer l, int batch, int location, int entry)
{        
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    #pragma omp parallel for private(n) reduction(+:count)
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(sigmoid(l.output[obj_index]) > thresh){
                ++count;
            }
        }
    }
    return count;
}


detection *make_layer_boxes(layer l, float thresh, int *num)
{
    int i;
    int nboxes = yolo_num_detections(l, thresh);
    if(num) *num = nboxes;
    detection *dets = (detection*)calloc(nboxes, sizeof(detection));
    for(i = 0; i < nboxes; ++i){
        dets[i].prob = (float*)calloc(l.classes, sizeof(float));
    }
    return dets;
}


box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + sigmoid(x[index + 0*stride])) / lw;
    b.y = (j + sigmoid(x[index + 1*stride])) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}


int get_yolo_detections(layer l, int netw, int neth, float thresh, detection *dets)
{        
    int i,j,n;
    float *predictions = l.output;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = sigmoid(predictions[obj_index]);
            if(objectness <= thresh) continue;
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes; 
            box b = dets[count].bbox;
            for(j = 0; j < l.classes; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*sigmoid(predictions[class_index]);
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    return count;
}

detection *get_layer_boxes(int netw, int neth, layer l, float thresh, int *num) {

    detection *dets = make_layer_boxes(l, thresh, num);
    *num = get_yolo_detections(l, netw, neth, thresh, dets);

    return dets;

}

void free_detections(detection *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(dets[i].prob);
        if(dets[i].mask) free(dets[i].mask);
    }    
    free(dets);
}
