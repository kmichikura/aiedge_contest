#ifndef __YOLO_PARAM_H__
#define __YOLO_PARAM_H__
// Parameter of Yolov3-tiny
int   anchors[] = {10,14,  23,27,  37,58,  81,82,  135,169,  344,319};
int   num           = 6;
float jitter        = .3;
float ignore_thresh = .7;
float truth_thresh  = 1;
int   yolo_randam   = 1;
// --------
// outbuf0 : mask = 3, 4, 5
// outbuf1 : mask = 0, 1, 2
// --------
int  mask[2][3] = { {3 , 4, 5}, {0, 1, 2} };

#endif // __YOLO_PARAM_H__
