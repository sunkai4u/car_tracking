#include "opencv2/video/background_segm.hpp"
#include "opencv2/legacy/blobtrack.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <opencv/cv.h>
#include <iostream>
using namespace std;

/* Select appropriate case insensitive string comparison function: */
#if defined WIN32 || defined _MSC_VER
#define MY_STRNICMP strnicmp
#define MY_STRICMP stricmp
#else
#define MY_STRNICMP strncasecmp
#define MY_STRICMP strcasecmp
#endif
using namespace cv;
class BackgroundSubtractor_FGD : public CvFGDetector
{
private:
	BackgroundSubtractorMOG2 bg_model;
	IplImage *image, *mask;
public:
	BackgroundSubtractor_FGD() : bg_model(300, 16)
	{
		image = NULL;
		mask = NULL;
	}
	virtual IplImage* GetMask()
	{
		return mask;
	}
	/* Process current image: */
	virtual void Process(IplImage* pImg)
	{
		if(image==NULL) image=cvCloneImage(pImg);
		if(mask==NULL) mask=cvCreateImage(cvGetSize(pImg), IPL_DEPTH_8U, 1);
		cvSmooth(pImg, image, CV_GAUSSIAN, 5);
		bg_model(Mat(image), Mat(mask));
		threshold(Mat(mask), Mat(mask), 128, 255, THRESH_BINARY);
		morphologyEx(Mat(mask), Mat(mask), MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	}
	/* Release foreground detector: */
	virtual void Release()
	{
		if(image) cvReleaseImage(&image);
		image = NULL;
		if(mask) cvReleaseImage(&mask);
		mask = NULL;
	}
	~BackgroundSubtractor_FGD()
	{
		Release();
	}
};
/* List of foreground (FG) DETECTION modules: */
inline CvFGDetector* cvCreateFGDetector0      () { return cvCreateFGDetectorBase(CV_BG_MODEL_FGD,        NULL); }
inline CvFGDetector* cvCreateFGDetector0Simple() { return cvCreateFGDetectorBase(CV_BG_MODEL_FGD_SIMPLE, NULL); }
inline CvFGDetector* cvCreateFGDetector1      () { return cvCreateFGDetectorBase(CV_BG_MODEL_MOG,        NULL); }
inline CvFGDetector* cvCreateFGDetector2      () { return new BackgroundSubtractor_FGD; }
struct DefModule_FGDetector
{
    CvFGDetector* (*create)();
    const char* nickname;
    const char* description;
};

DefModule_FGDetector FGDetector_Modules[] =
{
	{cvCreateFGDetector0,"FG_0","Foreground Object Detection from Videos Containing Complex Background. ACM MM2003."},
	{cvCreateFGDetector1,"FG_1","Adaptive background mixture models for real-time tracking. CVPR1999"},
	{cvCreateFGDetector0Simple,"FG_0S","Simplified version of FG_0"},
	{cvCreateFGDetector2,"BS_MOG2","BackgroundSubtractorMOG2"},
	//{cvCreateFGDetector3,"BS_MOG","BackgroundSubtractorMOG"},
	//{cvCreateFGDetector4,"BS_GMG","BackgroundSubtractorGMG"},
    {NULL,NULL,NULL}
};

/* List of BLOB DETECTION modules: */
struct DefModule_BlobDetector
{
    CvBlobDetector* (*create)();
    const char* nickname;
    const char* description;
};

DefModule_BlobDetector BlobDetector_Modules[] =
{
	{cvCreateBlobDetectorCC,"BD_CC","Detect new blob by tracking CC of FG mask"},
	{cvCreateBlobDetectorSimple,"BD_Simple","Detect new blob by uniform moving of connected components of FG mask"},
    {NULL,NULL,NULL}
};

/* List of BLOB TRACKING modules: */
struct DefModule_BlobTracker
{
    CvBlobTracker* (*create)();
    const char* nickname;
    const char* description;
};

DefModule_BlobTracker BlobTracker_Modules[] =
{
	{cvCreateBlobTrackerCCMSPF,"CCMSPF","connected component tracking and MSPF resolver for collision"},
	{cvCreateBlobTrackerMSPF,"MSPF","Particle filtering based on MS weight"},
	{cvCreateBlobTrackerMSFG,"MSFG","Mean shift algorithm with FG mask using"},
	{cvCreateBlobTrackerMS,"MS","Mean shift algorithm "},
	{cvCreateBlobTrackerCC,"CC","Simple connected component tracking"},
    {NULL,NULL,NULL}
};

/* List of BLOB TRAJECTORY GENERATION modules: */
struct DefModule_BlobTrackGen
{
    CvBlobTrackGen* (*create)();
    const char* nickname;
    const char* description;
};

DefModule_BlobTrackGen BlobTrackGen_Modules[] =
{
    {cvCreateModuleBlobTrackGenYML,"YML","Generate track record in YML format as synthetic video data"},
    {cvCreateModuleBlobTrackGen1,"RawTracks","Generate raw track record (x,y,sx,sy),()... in each line"},
    {NULL,NULL,NULL}
};

/* List of BLOB TRAJECTORY POST PROCESSING modules: */
struct DefModule_BlobTrackPostProc
{
    CvBlobTrackPostProc* (*create)();
    const char* nickname;
    const char* description;
};

DefModule_BlobTrackPostProc BlobTrackPostProc_Modules[] =
{
	{cvCreateModuleBlobTrackPostProcKalman,"Kalman","Kalman filtering of blob position and size"},
	{cvCreateModuleBlobTrackPostProcTimeAverExp,"TimeAverExp","Average by time using exponential window"},
	{cvCreateModuleBlobTrackPostProcTimeAverRect,"TimeAverRect","Average by time using rectangle window"},
    {NULL,"None","No post processing filter"},
    {NULL,NULL,NULL}
};

/* List of BLOB TRAJECTORY ANALYSIS modules: */
//CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisDetector();

struct DefModule_BlobTrackAnalysis
{
    CvBlobTrackAnalysis* (*create)();
    const char* nickname;
    const char* description;
};

DefModule_BlobTrackAnalysis BlobTrackAnalysis_Modules[] =
{
	{cvCreateModuleBlobTrackAnalysisHistPVS,"HistPVS","Histogram of 5D feature vector analysis (x,y,vx,vy,state)"},
	{cvCreateModuleBlobTrackAnalysisHistSS,"HistSS","Histogram of 4D feature vector analysis (startpos,endpos)"},
	{cvCreateModuleBlobTrackAnalysisIOR,"IOR","Integrator (by OR operation) of several analysers "},
	{cvCreateModuleBlobTrackAnalysisTrackDist,"TrackDist","Compare tracks directly"},
	{cvCreateModuleBlobTrackAnalysisHistPV,"HistPV","Histogram of 4D feature vector analysis (x,y,vx,vy)"},
	{cvCreateModuleBlobTrackAnalysisHistP,"HistP","Histogram of 2D feature vector analysis (x,y)"},
    {NULL,"None","No trajectory analiser"},
    {NULL,NULL,NULL}
};

/* List of Blob Trajectory ANALYSIS modules: */
/*================= END MODULES DECRIPTION ===================================*/

/* Run pipeline on all frames: */
int RunBlobTrackingAuto( CvCapture* pCap, CvBlobTrackerAuto* pTracker,char* fgavi_name = NULL, char* btavi_name = NULL )
{
    CvVideoWriter*          pFGAvi = NULL;
    CvVideoWriter*          pBTAvi = NULL;
    IplImage*               pImg = NULL;
    const int               max_size = 512;
    
	cvNamedWindow("FG");
	cvNamedWindow("Tracking");
    /* Main loop: */
    for(int FrameNum=0; pCap && (cvWaitKey(30)!=27); FrameNum++)
    {   /* Main loop: */
        IplImage* pFrame = cvQueryFrame(pCap);
        if(pFrame == NULL) break;
        if(pImg == NULL)
        {
            CvSize nSize;
            double scale = static_cast<double>(max_size)/max(pFrame->width, pFrame->height);
            nSize.width = pFrame->width*scale;
            nSize.height = pFrame->height*scale;
            pImg = cvCreateImage(nSize, IPL_DEPTH_8U, 3);
        }
        cvResize(pFrame, pImg);
        /* Process: */
        pTracker->Process(pImg, NULL);
        
        if(pTracker->GetFGMask())
        {   /* Debug FG: */
			IplImage*    pI = cvCreateImage(cvGetSize(pTracker->GetFGMask()), IPL_DEPTH_8U, 3);
			CvSize       S = cvSize(pI->width, pI->height);
            
            cvCvtColor(pTracker->GetFGMask(), pI, CV_GRAY2BGR);
            /* Draw detected blobs: */
			for(int i=pTracker->GetBlobNum();i>0;i--)
			{
				CvBlob* pB = pTracker->GetBlob(i-1);
				const CvPoint p = cvPointFrom32f(CV_BLOB_CENTER(pB));
				const CvSize  s = cvSize(MAX(1,cvRound(CV_BLOB_RX(pB))), MAX(1,cvRound(CV_BLOB_RY(pB))));
				const int c = cvRound(255*pTracker->GetState(CV_BLOB_ID(pB)));
				//cvEllipse(pI, p, s, 0, 0, 360, CV_RGB(c,255-c,0), cvRound(1+(3*c)/255));
				cvEllipse(pI, p, s, 0, 0, 360, CV_RGB(0,255,0), cvRound(1+(3*c)/255));
			}   /* Next blob: */;
#ifndef _WIN32
			if(fgavi_name)
			{   /* Save fg to avi file: */
				if(pFGAvi==NULL)
					pFGAvi=cvCreateVideoWriter(fgavi_name,CV_FOURCC('d','i','v','x'), 25, S);
				cvWriteFrame( pFGAvi, pI );
			}
#endif
            cvShowImage("FG",pI);
			cvReleaseImage(&pI);
        }   /* Debug FG. */
        
        
        /* Draw debug info: */
		/* Draw all information about test sequence: */
		char        str[1024];
		CvFont      font;
		cvInitFont( &font, CV_FONT_HERSHEY_PLAIN, 0.7, 0.7, 0, 1, CV_AA );
        
		IplImage*   pI = cvCloneImage(pImg);
		for(int i=pTracker->GetBlobNum(); i>0; i--)
		{
			CvSize  TextSize;
			CvBlob* pB = pTracker->GetBlob(i-1);
			CvPoint p = cvPoint(cvRound(pB->x*256),cvRound(pB->y*256));
			CvSize  s = cvSize(MAX(1,cvRound(CV_BLOB_RX(pB)*256)), MAX(1,cvRound(CV_BLOB_RY(pB)*256)));
			//int c = cvRound(255*pTracker->GetState(CV_BLOB_ID(pB)));
            
			//cvEllipse(pI, p, s, 0, 0, 360, CV_RGB(c,255-c,0), cvRound(1+(3*0)/255), CV_AA, 8);
			cvEllipse(pI, p, s, 0, 0, 360, CV_RGB(0, 255, 0), cvRound(1+(3*0)/255), CV_AA, 8);
            
			p.x >>= 8;
			p.y >>= 8;
			s.width >>= 8;
			s.height >>= 8;
			sprintf(str,"%03d",CV_BLOB_ID(pB));
			cvGetTextSize( str, &font, &TextSize, NULL );
			p.y -= s.height;
			cvPutText( pI, str, p, &font, CV_RGB(0,255,255));
            //{
			//    const char* pS = pTracker->GetStateDesc(CV_BLOB_ID(pB));
			//    if(pS)
			//    {
			//        char* pStr = strdup(pS);
			//        char* pStrFree = pStr;
            
			//        while (pStr && strlen(pStr) > 0)
			//        {
			//            char* str_next = strchr(pStr,'\n');
            
			//            if(str_next)
			//            {
			//                str_next[0] = 0;
			//                str_next++;
			//            }
            
			//            p.y += TextSize.height+1;
			//            cvPutText( pI, pStr, p, &font, CV_RGB(0,255,255));
			//            pStr = str_next;
			//        }
			//        free(pStrFree);
			//    }
			//}
		}   /* Next blob. */;
        
		cvShowImage( "Tracking",pI );
        
		if(btavi_name && pI)
		{   /* Save to avi file: */
			CvSize      S = cvSize(pI->width,pI->height);
			if(pBTAvi==NULL)
				pBTAvi=cvCreateVideoWriter(btavi_name, CV_FOURCC('d','i','v','x'), 25, S);
			cvWriteFrame( pBTAvi, pI );
		}
        
		cvReleaseImage(&pI);
        /* Draw all information about test sequence. */
    }   /*  Main loop. */
    
    if(pFGAvi)cvReleaseVideoWriter( &pFGAvi );
	if(pBTAvi)cvReleaseVideoWriter( &pBTAvi );
	cvDestroyWindow("FG");
	cvDestroyWindow("Tracking");
    if(pImg) cvReleaseImage(&pImg);
    return 0;
}   /* RunBlobTrackingAuto */

/* Read parameters from command line
 * and transfer to specified module:
 */
static void set_params(int argc, char* argv[], CvVSModule* pM, const char* prefix, const char* module)
{
    int prefix_len = (int)strlen(prefix);
    int i;
    for(i=0; i<argc; ++i)
    {
        int j;
        char* ptr_eq = NULL;
        int   cmd_param_len=0;
        char* cmd = argv[i];
        if(MY_STRNICMP(prefix,cmd,prefix_len)!=0) continue;
        cmd += prefix_len;
        if(cmd[0]!=':')continue;
        cmd++;
        
        ptr_eq = strchr(cmd,'=');
        if(ptr_eq)
            cmd_param_len = (int)(ptr_eq-cmd);
        
        for(j=0; ; ++j)
        {
            int     param_len;
            const char*   param = pM->GetParamName(j);
            if(param==NULL) break;
            param_len = (int)strlen(param);
            if(cmd_param_len!=param_len) continue;
            if(MY_STRNICMP(param,cmd,param_len)!=0) continue;
            cmd+=param_len;
            if(cmd[0]!='=')continue;
            cmd++;
            pM->SetParamStr(param,cmd);
            printf("%s:%s param set to %g\n",module,param,pM->GetParam(param));
        }
    }
    
    pM->ParamUpdate();
    
}   /* set_params */

/* Print all parameter values for given module: */
static void print_params(CvVSModule* pM, const char* module, const char* log_name)
{
    FILE* log = log_name?fopen(log_name,"at"):NULL;
    int i;
    if(pM->GetParamName(0) == NULL ) return;
    
    
    printf("%s(%s) module parameters:\n",module,pM->GetNickName());
    if(log)
        fprintf(log,"%s(%s) module parameters:\n",module,pM->GetNickName());
    
    for (i=0; ; ++i)
    {
        const char*   param = pM->GetParamName(i);
        const char*   str = param?pM->GetParamStr(param):NULL;
        if(param == NULL)break;
        if(str)
        {
            printf("  %s: %s\n",param,str);
            if(log)
                fprintf(log,"  %s: %s\n",param,str);
        }
        else
        {
            printf("  %s: %g\n",param,pM->GetParam(param));
            if(log)
                fprintf(log,"  %s: %g\n",param,pM->GetParam(param));
        }
    }
    
    if(log) fclose(log);
    
}   /* print_params */

template<class T>
class MyList
{
private:
    T* data;
    const unsigned long size;
    const unsigned int bits;
    unsigned long head;
    unsigned long tail;
public:
    MyList(const unsigned int _bits) : size(1L<<_bits), bits(_bits)
    {
        data = new T[size];
        head = 0;
        tail = 0;
    }
    
    MyList()
    {
        delete [] data;
    }
    
    T& get(const unsigned long pos)
    {
        return data[pos-((pos>>bits)<<bits)];
    }
    void add(const T& v)
    {
        get(tail++) = v;
        if(tail > size)
            ++head;
    }
    unsigned long getHead() const { return head; }
    unsigned long getTail() const { return tail; }
    unsigned long getSize() const { return tail-head; }
    unsigned long capacity() const { return size; }
};
void testMyList()
{
    MyList<int> list(3);
    for(unsigned int i=0; i<24; ++i)
    {
        cout<<i<<": ";
        for(unsigned long k=list.getHead(); k<list.getTail(); ++k)
        {
            cout<<list.get(k)<<'\t';
        }
        cout<<endl;
        list.add(i);
    }
}

enum VALUE_STATUS { nochanged=0, up, down};
enum FRAME_STATUS { no, yes, maybe };
struct FrameInfo
{
    float v;
    VALUE_STATUS vstatus; //
    FRAME_STATUS fstatus;
};

class Detector
{
private:
    MyList<FrameInfo> list;
    double fps;
    
public:
    Detector(double _fps, unsigned int list_bits) : fps(_fps), list(list_bits)
    {
        
    }
    unsigned long capacity() const { return list.capacity(); }
    FRAME_STATUS update(FrameInfo _v)
    {
        list.add(_v);
        
        const unsigned long cFrame = list.getTail()-1;
        const unsigned long maxHistory = min((unsigned long)(fps*3), list.getSize()-1);
        FrameInfo& v = list.get(cFrame);
        const float candidateThre = 0.02;
        if(list.getSize() > maxHistory)
        {
            FrameInfo& pv = list.get(cFrame-1);
            v.v = (v.v+pv.v)*0.5;
            if(v.v > candidateThre)
            {
                if(v.v >= pv.v)
                {
                    v.vstatus = up;
                    v.fstatus = maybe;
                }
                else
                {
                    unsigned int i=0;
                    unsigned int nCount = 0;
                    const unsigned int maxNoCount = 3;
                    for(i=0; i<maxHistory; ++i)
                    {
                        FrameInfo& tv = list.get(cFrame-1-i);
                        if(tv.fstatus == yes)
                        {
                            nCount=0;
                            v.fstatus = maybe;
                            break;
                        }
                        else if(tv.fstatus==maybe)
                        {
                            nCount = 0;
                            v.fstatus = maybe;
                        }
                        else if(tv.fstatus == no)
                        {
                            nCount++;
                            v.fstatus = maybe;
                        }
                    }
                    if(i==maxHistory || nCount>maxNoCount)
                    {
                        v.fstatus = yes;
                    }
                }
            }
            else
            {
                v.fstatus = no;
                v.vstatus = nochanged;
            }
            
        }
        else
        {
            v.fstatus = no;
            v.vstatus = nochanged;
        }
        return v.fstatus;
    }
    
    void show(Mat& img)
    {
        img = Mat::zeros(img.size(), CV_8U);
        Point p1;
        p1.x = 0;
        p1.y = 0;
        for(unsigned long k=list.getHead(); k<list.getTail(); ++k)
        {
            Point p2;
            p2.x = (int)(k-list.getHead());
            FrameInfo& v = list.get(k);
            p2.y = img.rows-v.v*img.rows;
            line(img, p1, p2, Scalar(128));
            if(v.fstatus==yes)
            {
                circle(img, p2, 1, Scalar(255), 2);
            }
            p1 = p2;
        }
    }
};

int main(int argc, char* argv[])
{
    VideoCapture cap;
    if(argc>1)
    {
        cap.open(argv[1]);
    }
    else
    {
        cap.open(0);
    }
    if(!cap.isOpened())
    {
        if(argc>1) cout<<"Can't open "<<argv[1]<<endl;
        else cout<<"Can't open Camera"<<endl;
        return -1;
    }
    CvFGDetector* bgfg = new BackgroundSubtractor_FGD;//cvCreateFGDetectorBase(CV_BG_MODEL_FGD, NULL);
    namedWindow("frame");
    namedWindow("roi");
    namedWindow("mask");
    namedWindow("info");
    const int maxWidth = 512;
    const int windowSize = 7;
    Mat frame, roiImg, mask, weight;
    weight.create(windowSize, maxWidth, CV_32F);
    for(int j=0; j<weight.rows; ++j)
    {
        float* fdata = weight.ptr<float>(j);
        float lineWeight = j-windowSize/2;
        lineWeight*=lineWeight;
        lineWeight = exp(-lineWeight/(windowSize*windowSize/4));
        for(unsigned int i=0; i<weight.cols; ++i)
        {
            fdata[i] = lineWeight;
        }
    }
    const int listBits = 9;
    Detector detector(cap.get(CV_CAP_PROP_FPS), listBits);
    Mat info(512, (int)detector.capacity(), CV_8U);
    cap>>frame;
    unsigned long nFrame = 0;
    while (!frame.empty() && waitKey(30)!=27)
    {
        Size nsize;
        nsize.width = maxWidth/(double)frame.cols*frame.cols;
        nsize.height = maxWidth/(double)frame.cols*frame.rows;
        resize(frame, frame, nsize);
        Rect roiRect(0, (nsize.height-windowSize)*0.5, nsize.width, windowSize);
        roiImg = Mat(frame, roiRect);
        
        IplImage image = IplImage(roiImg);
        bgfg->Process(&image);
        mask= Mat(bgfg->GetMask()).clone();
        multiply(mask, weight, mask, 1, CV_32F);
        FrameInfo v;
        v.v = sum(mask)[0]/(255*mask.rows*mask.cols);
        FRAME_STATUS status = detector.update(v);
        
        detector.show(info);
        Scalar color;
        if(status==yes)
        {
            Mat diff;
            subtract(255, frame, diff);
            frame = diff;
            color = Scalar(0, 0, 255);
        }
        else if(status==maybe)
            color = Scalar(255, 0, 0);
        else
            color = Scalar(0, 255, 0);
        rectangle(frame, roiRect, color);
        imshow("frame", frame);
        imshow("info", info);
        //imshow("roi", roiImg);
        //imshow("mask", mask);
        cap>>frame;
        ++nFrame;
    }
    cvReleaseFGDetector(&bgfg);
    destroyAllWindows();
    return 0;
}
