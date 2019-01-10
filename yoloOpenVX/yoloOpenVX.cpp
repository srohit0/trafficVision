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

#include "annpython.h"
#include "region.h"
#include <python2.7/Python.h>

#include <sstream>

#define RETURN_ON_ERROR(status, gstate) { if(status != VX_SUCCESS) { printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); PyGILState_Release(gstate); return 0x0; } }

using namespace std;

struct yoloDetails {
    protected:
        vector<string> tokenize(string info_str, char delim)
        {
            string token; 
            vector <string> tokens; 
            stringstream info_stream(info_str); 

            // Tokenizing by comma char ',' 
            while(getline(info_stream, token, delim)) 
                tokens.push_back(token); 

            return tokens;
        }
    public:
        int ni, ci, hi, wi; // network input parameters.
        int numBoxesParams, maxBoxWidth, maxBoxHeight; // network output parameters.
        int targetBlockwd, classes ;
        float threshold, nms ;

        yoloDetails()
        {
            const char* info_cstr             = annQueryInference();
            vector<string> yolo_terminal_info = tokenize(info_cstr, ';');
            vector<string> yolo_inputs        = tokenize(yolo_terminal_info[0], ',');
            vector<string> yolo_outputs       = tokenize(yolo_terminal_info[1], ',');

            istringstream(yolo_inputs[2]) >> ni ; // numInputs     = 1
            istringstream(yolo_inputs[3]) >> ci ; // channelInputs = 3
            istringstream(yolo_inputs[4]) >> hi ; // heightImage   = 416
            istringstream(yolo_inputs[5]) >> wi ; // widthImage    = 416

            istringstream(yolo_outputs[3]) >> numBoxesParams ; // 125
            istringstream(yolo_outputs[4]) >> maxBoxWidth ; // 12
            istringstream(yolo_outputs[5]) >> maxBoxHeight ; // 12

            targetBlockwd = 13 ;        // yolo grid size
            classes = 20 ;              // yolo number of classes
            threshold = 0.18 ;          // yolo confidence threshold 
            nms = 0.4 ;                 // yolo non-maximum suppression threshold
        }
};

static yoloDetails yoloInfo ;

extern "C" PyObject* detectBoxes(pyif_ann_handle handle, float* image) {

    // 1. runInference
    // 1a. Check for graph
    if ( !handle )
        return 0x0 ;

    PyGILState_STATE gstate = PyGILState_Ensure();

    size_t inp_size = yoloInfo.hi  * yoloInfo.wi * yoloInfo.ci * sizeof(float);
    bool is_nhwc=0;
    // 1b. Prepare input for graph
    vx_status status = annCopyToInferenceInput(handle, image, inp_size, is_nhwc);
    RETURN_ON_ERROR(status, gstate)

    // 1c. Process bipertite graph consisting of a set of Nodes and a set of data objects
    status = annRunInference(handle, 1);
    RETURN_ON_ERROR(status, gstate)

    // 1d. Extract results from the graph outputs.
    size_t out_size = yoloInfo.numBoxesParams * yoloInfo.maxBoxWidth * yoloInfo.maxBoxHeight * sizeof(float) ;
    static float * out_ptr = (float*) malloc(out_size);
    status = annCopyFromInferenceOutput(handle, out_ptr, out_size);
    RETURN_ON_ERROR(status, gstate)


    // 2. DetectBoxes
    Region box(yoloInfo.numBoxesParams, yoloInfo.maxBoxHeight, yoloInfo.maxBoxWidth, yoloInfo.classes);
    PyObject* objects = box.GetDetections(out_ptr, yoloInfo.numBoxesParams,
                                yoloInfo.maxBoxHeight, yoloInfo.maxBoxWidth,
                                yoloInfo.classes, yoloInfo.wi, yoloInfo.hi,
                                yoloInfo.threshold, yoloInfo.nms, yoloInfo.targetBlockwd);

    PyGILState_Release(gstate);
    return objects;
}

