// -*- coding: utf-8; -*-
////////////////////////////////////////////////////////////////////////////
// This confidential and proprietary software may be used
// only as authorized by a licensing agreement from KONICA MINOLTA, Inc.
// In the event of publication, the following notice is applicable:
//
// (C) COPYRIGHT 2016 KONICA MINOLTA, Inc.
//  ALL RIGHTS RESERVED
//
// File     : ui.hpp
// Abstract : definition for user I/F process
////////////////////////////////////////////////////////////////////////////
#ifndef __UI_H__
#define __UI_H__

#include <sys/time.h>                   // for gettimeofday()
class StopWatch {
public:
    StopWatch()  { gettimeofday(&t_start, NULL); }
    ~StopWatch() {}
    long getTime(void) {
        struct timeval now;
        gettimeofday(&now, NULL);
        return (long)(now.tv_sec - t_start.tv_sec) * 1000000 + (now.tv_usec - t_start.tv_usec);
    }
private:
    struct timeval t_start;
};
extern StopWatch stopwatch;

/// message
#define V_ERROR     10
#define V_MESSAGE0  50
#define V_MESSAGE   100
#define V_MESSAGE1  150
#define V_MESSAGE2  200
#define V_MESSAGE3  300
#define V_MESSAGE4  400
#define V_MESSAGE5  500
#define V_MESSAGE6  600
#define V_MESSAGE7  700
#define V_MESSAGE8  800
#define V_MESSAGE9  900
#define V_TIME      20
#define V_TRACE     1000
#define V_DEBUG     2000
extern int verbose;
#if 1
#define Trace(v, fmt, ...) if (verbose >= (v)) printf("# %-28s %5d %11ld " fmt "\n", __FUNCTION__, __LINE__, stopwatch.getTime(), __VA_ARGS__); else
#else
#define Trace(v, fmt, ...)
#endif

#endif  //__UI_H__
