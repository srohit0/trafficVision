/* 
Copyright (c) 2018. All rights reserved.
 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#pragma once

#ifndef _REGION_H
#define _REGION_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <python2.7/Python.h>

struct ibox
{
    float x; // x co-ordinate
    float y; // y co-ordinate
    float w; // box width
    float h; // box-height
};

struct indexsort
{
    int    iclass;
    int    index;
    int    channel;
    float* prob;
};

class Region
{
    private:
        int                      size;
        int                      totalLength;
        int                      totalObjects;
        std::vector<float>       output;
        std::vector<ibox>        boxes;
        std::vector<indexsort>   s;

        static const int         numBoxesPerGrid ;
        static const float       biases[];
        static const std::string objectnames[] ;

    protected:
        Region(); // prohibited.

        static int   indexsort_comparator(const void *pa, const void *pb);
        float        logistic_activate   (float x);
        void         transpose           (float *src, float* tar, int k, int n);
        void         softmax             (float *input, int n, float temp, float *output);
        float        overlap             (float x1, float w1, float x2, float w2);
        float        box_intersection    (ibox a, ibox b);
        float        box_union           (ibox a, ibox b);
        float        box_iou             (ibox a, ibox b);
        int          max_index           (float *a, int n);

    public:
        Region                           (int c, int h, int w, int classes);

        PyObject* GetDetections          (float* data, int c, int h, int w,
                                          int classes, int imgw, int imgh,
                                          float thresh, float nms,
                                          int blockwd);
};

#endif // _REGION_H
