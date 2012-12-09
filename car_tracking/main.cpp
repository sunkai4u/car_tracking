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
			int c = cvRound(255*pTracker->GetState(CV_BLOB_ID(pB)));

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

//int main(int argc, char* argv[])
//{   /* Main function: */
//    CvCapture*                  pCap = NULL;
//    CvBlobTrackerAutoParam1     param = {0,0,0,0,0,0,0,0};
//    CvBlobTrackerAuto*          pTracker = NULL;
//
//    //float       scale = 1;
//    const char* scale_name = NULL;
//    char*       yml_name = NULL;
//    char**      yml_video_names = NULL;
//    int         yml_video_num = 0;
//    char*       avi_name = NULL;
//    const char* fg_name = NULL;
//    char*       fgavi_name = NULL;
//    char*       btavi_name = NULL;
//    const char* bd_name = NULL;
//    const char* bt_name = NULL;
//    const char* btgen_name = NULL;
//    const char* btpp_name = NULL;
//    const char* bta_name = NULL;
//    char*       bta_data_name = NULL;
//    char*       track_name = NULL;
//    //char*       comment_name = NULL;
//    char*       FGTrainFrames = NULL;
//    char*       log_name = NULL;
//    char*       savestate_name = NULL;
//    char*       loadstate_name = NULL;
//    const char* bt_corr = NULL;
//    DefModule_FGDetector*           pFGModule = NULL;
//    DefModule_BlobDetector*         pBDModule = NULL;
//    DefModule_BlobTracker*          pBTModule = NULL;
//    DefModule_BlobTrackPostProc*    pBTPostProcModule = NULL;
//    DefModule_BlobTrackGen*         pBTGenModule = NULL;
//    DefModule_BlobTrackAnalysis*    pBTAnalysisModule = NULL;
//
//    cvInitSystem(argc, argv);
//
//    if(argc < 2)
//    {   /* Print help: */
//        int i;
//        printf("blobtrack [fg=<fg_name>] [bd=<bd_name>]\n"
//            "          [bt=<bt_name>] [btpp=<btpp_name>]\n"
//            "          [bta=<bta_name>\n"
//            "          [bta_data=<bta_data_name>\n"
//            "          [bt_corr=<bt_corr_way>]\n"
//            "          [btgen=<btgen_name>]\n"
//            "          [track=<track_file_name>]\n"
//            "          [scale=<scale val>] [noise=<noise_name>] [IVar=<IVar_name>]\n"
//            "          [FGTrainFrames=<FGTrainFrames>]\n"
//            "          [btavi=<avi output>] [fgavi=<avi output on FG>]\n"
//            "          <avi_file>\n");
//
//        printf("  <bt_corr_way> is the method of blob position correction for the \"Blob Tracking\" module\n"
//            "     <bt_corr_way>=none,PostProcRes\n"
//            "  <FGTrainFrames> is number of frames for FG training\n"
//            "  <track_file_name> is file name for save tracked trajectories\n"
//            "  <bta_data> is file name for data base of trajectory analysis module\n"
//            "  <avi_file> is file name of avi to process by BlobTrackerAuto\n");
//
//        puts("\nModules:");
//#define PR(_name,_m,_mt)\
//        printf("<%s> is \"%s\" module name and can be:\n",_name,_mt);\
//        for(i=0; _m[i].nickname; ++i)\
//        {\
//            printf("  %d. %s",i+1,_m[i].nickname);\
//            if(_m[i].description)printf(" - %s",_m[i].description);\
//            printf("\n");\
//        }
//
//        PR("fg_name",FGDetector_Modules,"FG/BG Detection");
//        PR("bd_name",BlobDetector_Modules,"Blob Entrance Detection");
//        PR("bt_name",BlobTracker_Modules,"Blob Tracking");
//        PR("btpp_name",BlobTrackPostProc_Modules, "Blob Trajectory Post Processing");
//        PR("btgen_name",BlobTrackGen_Modules, "Blob Trajectory Generation");
//        PR("bta_name",BlobTrackAnalysis_Modules, "Blob Trajectory Analysis");
//#undef PR
//        return 0;
//    }   /* Print help. */
//
//    {   /* Parse arguments: */
//        int i;
//        for(i=1; i<argc; ++i)
//        {
//            int bParsed = 0;
//            size_t len = strlen(argv[i]);
//#define RO(_n1,_n2) if(strncmp(argv[i],_n1,strlen(_n1))==0) {_n2 = argv[i]+strlen(_n1);bParsed=1;};
//            RO("fg=",fg_name);
//            RO("fgavi=",fgavi_name);
//            RO("btavi=",btavi_name);
//            RO("bd=",bd_name);
//            RO("bt=",bt_name);
//            RO("bt_corr=",bt_corr);
//            RO("btpp=",btpp_name);
//            RO("bta=",bta_name);
//            RO("bta_data=",bta_data_name);
//            RO("btgen=",btgen_name);
//            RO("track=",track_name);
//            //RO("comment=",comment_name);
//            RO("FGTrainFrames=",FGTrainFrames);
//            RO("log=",log_name);
//            RO("savestate=",savestate_name);
//			RO("loadstate=",loadstate_name);
//			RO("fn=",avi_name);
//#undef RO
//        }
//    }   /* Parse arguments. */
//
//    if(track_name)
//    {   /* Set Trajectory Generator module: */
//        int i;
//        if(!btgen_name)btgen_name=BlobTrackGen_Modules[0].nickname;
//
//        for(i=0; BlobTrackGen_Modules[i].nickname; ++i)
//        {
//            if(MY_STRICMP(BlobTrackGen_Modules[i].nickname,btgen_name)==0)
//                pBTGenModule = BlobTrackGen_Modules + i;
//        }
//    }   /* Set Trajectory Generato module. */
//
//    /* Initialize postprocessing module if tracker
//     * correction by postprocessing is required.
//     */
//    if(bt_corr && MY_STRICMP(bt_corr,"PostProcRes")!=0 && !btpp_name)
//    {
//        btpp_name = bt_corr;
//        if(MY_STRICMP(btpp_name,"none")!=0)bt_corr = "PostProcRes";
//    }
//
//    {   /* Set default parameters for one processing: */
//        if(!bt_corr) bt_corr = "none";
//        if(!fg_name) fg_name = FGDetector_Modules[0].nickname;
//        if(!bd_name) bd_name = BlobDetector_Modules[0].nickname;
//        if(!bt_name) bt_name = BlobTracker_Modules[0].nickname;
//        if(!btpp_name) btpp_name = BlobTrackPostProc_Modules[0].nickname;
//        if(!bta_name) bta_name = BlobTrackAnalysis_Modules[0].nickname;
//        if(!scale_name) scale_name = "1";
//    }
//
////    if(scale_name)
//  //      scale = (float)atof(scale_name);
//
//    for(pFGModule=FGDetector_Modules; pFGModule->nickname; ++pFGModule)
//        if( fg_name && MY_STRICMP(fg_name,pFGModule->nickname)==0 ) break;
//
//    for(pBDModule=BlobDetector_Modules; pBDModule->nickname; ++pBDModule)
//        if( bd_name && MY_STRICMP(bd_name,pBDModule->nickname)==0 ) break;
//
//    for(pBTModule=BlobTracker_Modules; pBTModule->nickname; ++pBTModule)
//        if( bt_name && MY_STRICMP(bt_name,pBTModule->nickname)==0 ) break;
//
//    for(pBTPostProcModule=BlobTrackPostProc_Modules; pBTPostProcModule->nickname; ++pBTPostProcModule)
//        if( btpp_name && MY_STRICMP(btpp_name,pBTPostProcModule->nickname)==0 ) break;
//
//    for(pBTAnalysisModule=BlobTrackAnalysis_Modules; pBTAnalysisModule->nickname; ++pBTAnalysisModule)
//        if( bta_name && MY_STRICMP(bta_name,pBTAnalysisModule->nickname)==0 ) break;
//
//    /* Create source video: */
//    if(avi_name)
//        pCap = cvCaptureFromFile(avi_name);
//
//    if(pCap==NULL)
//    {
//        printf("Can't open %s file\n",avi_name);
//        return -1;
//    }
//
//
//    {   /* Display parameters: */
//        int i;
//        FILE* log = log_name?fopen(log_name,"at"):NULL;
//        if(log)
//        {   /* Print to log file: */
//            fprintf(log,"\n=== Blob Tracking pipline in processing mode===\n");
//            if(avi_name)
//            {
//                fprintf(log,"AVIFile: %s\n",avi_name);
//            }
//            fprintf(log,"FGDetector:   %s\n", pFGModule->nickname);
//            fprintf(log,"BlobDetector: %s\n", pBDModule->nickname);
//            fprintf(log,"BlobTracker:  %s\n", pBTModule->nickname);
//            fprintf(log,"BlobTrackPostProc:  %s\n", pBTPostProcModule->nickname);
//            fprintf(log,"BlobCorrection:  %s\n", bt_corr);
//
//            fprintf(log,"Blob Trajectory Generator:  %s (%s)\n",
//                pBTGenModule?pBTGenModule->nickname:"None",
//                track_name?track_name:"none");
//
//            fprintf(log,"BlobTrackAnalysis:  %s\n", pBTAnalysisModule->nickname);
//            fclose(log);
//        }
//
//        printf("\n=== Blob Tracking pipline in %s mode===\n","processing");
//        if(yml_name)
//        {
//            printf("ConfigFile: %s\n",yml_name);
//            printf("BG: %s\n",yml_video_names[0]);
//            printf("FG: ");
//            for(i=1;i<(yml_video_num);++i){printf("%s",yml_video_names[i]);if((i+1)<yml_video_num)printf("|");};
//            printf("\n");
//        }
//        if(avi_name)
//        {
//            printf("AVIFile: %s\n",avi_name);
//        }
//        printf("FGDetector:   %s\n", pFGModule->nickname);
//        printf("BlobDetector: %s\n", pBDModule->nickname);
//        printf("BlobTracker:  %s\n", pBTModule->nickname);
//        printf("BlobTrackPostProc:  %s\n", pBTPostProcModule->nickname);
//        printf("BlobCorrection:  %s\n", bt_corr);
//
//        printf("Blob Trajectory Generator:  %s (%s)\n",
//            pBTGenModule?pBTGenModule->nickname:"None",
//            track_name?track_name:"none");
//
//        printf("BlobTrackAnalysis:  %s\n", pBTAnalysisModule->nickname);
//
//    }   /* Display parameters. */
//
//    {   /* Create autotracker module and its components: */
//        param.FGTrainFrames = FGTrainFrames?atoi(FGTrainFrames):0;
//
//        /* Create FG Detection module: */
//        param.pFG = pFGModule->create();
//        if(!param.pFG)
//            puts("Can not create FGDetector module");
//        param.pFG->SetNickName(pFGModule->nickname);
//        set_params(argc, argv, param.pFG, "fg", pFGModule->nickname);
//
//        /* Create Blob Entrance Detection module: */
//        param.pBD = pBDModule->create();
//        if(!param.pBD)
//            puts("Can not create BlobDetector module");
//        param.pBD->SetNickName(pBDModule->nickname);
//        set_params(argc, argv, param.pBD, "bd", pBDModule->nickname);
//
//        /* Create blob tracker module: */
//        param.pBT = pBTModule->create();
//        if(!param.pBT)
//            puts("Can not create BlobTracker module");
//        param.pBT->SetNickName(pBTModule->nickname);
//        set_params(argc, argv, param.pBT, "bt", pBTModule->nickname);
//
//        /* Create blob trajectory generation module: */
//        param.pBTGen = NULL;
//        if(pBTGenModule && track_name && pBTGenModule->create)
//        {
//            param.pBTGen = pBTGenModule->create();
//            param.pBTGen->SetFileName(track_name);
//        }
//        if(param.pBTGen)
//        {
//            param.pBTGen->SetNickName(pBTGenModule->nickname);
//            set_params(argc, argv, param.pBTGen, "btgen", pBTGenModule->nickname);
//        }
//
//        /* Create blob trajectory post processing module: */
//        param.pBTPP = NULL;
//        if(pBTPostProcModule && pBTPostProcModule->create)
//        {
//            param.pBTPP = pBTPostProcModule->create();
//        }
//        if(param.pBTPP)
//        {
//            param.pBTPP->SetNickName(pBTPostProcModule->nickname);
//            set_params(argc, argv, param.pBTPP, "btpp", pBTPostProcModule->nickname);
//        }
//
//        param.UsePPData = (bt_corr && MY_STRICMP(bt_corr,"PostProcRes")==0);
//
//        /* Create blob trajectory analysis module: */
//        param.pBTA = NULL;
//        if(pBTAnalysisModule && pBTAnalysisModule->create)
//        {
//            param.pBTA = pBTAnalysisModule->create();
//            param.pBTA->SetFileName(bta_data_name);
//        }
//        if(param.pBTA)
//        {
//            param.pBTA->SetNickName(pBTAnalysisModule->nickname);
//            set_params(argc, argv, param.pBTA, "bta", pBTAnalysisModule->nickname);
//        }
//
//        /* Create whole pipline: */
//        pTracker = cvCreateBlobTrackerAuto1(&param);
//        if(!pTracker)
//            puts("Can not create BlobTrackerAuto");
//    }
//
//    {   /* Load states of each module from state file: */
//        CvFileStorage* fs = NULL;
//        if(loadstate_name)
//            fs=cvOpenFileStorage(loadstate_name,NULL,CV_STORAGE_READ);
//        if(fs)
//        {
//            printf("Load states for modules...\n");
//            if(param.pBT)
//            {
//                CvFileNode* fn = cvGetFileNodeByName(fs,NULL,"BlobTracker");
//                param.pBT->LoadState(fs,fn);
//            }
//
//            if(param.pBTA)
//            {
//                CvFileNode* fn = cvGetFileNodeByName(fs,NULL,"BlobTrackAnalyser");
//                param.pBTA->LoadState(fs,fn);
//            }
//
//            if(pTracker)
//            {
//                CvFileNode* fn = cvGetFileNodeByName(fs,NULL,"BlobTrackerAuto");
//                pTracker->LoadState(fs,fn);
//            }
//
//            cvReleaseFileStorage(&fs);
//            printf("... Modules states loaded\n");
//        }
//    }   /* Load states of each module. */
//
//    {   /* Print module parameters: */
//        struct DefMMM
//        {
//            CvVSModule* pM;
//            const char* name;
//        } Modules[] = {
//            {(CvVSModule*)param.pFG,"FGdetector"},
//            {(CvVSModule*)param.pBD,"BlobDetector"},
//            {(CvVSModule*)param.pBT,"BlobTracker"},
//            {(CvVSModule*)param.pBTGen,"TrackGen"},
//            {(CvVSModule*)param.pBTPP,"PostProcessing"},
//            {(CvVSModule*)param.pBTA,"TrackAnalysis"},
//            {NULL,NULL}
//        };
//        int     i;
//        for(i=0; Modules[i].name; ++i)
//        {
//            if(Modules[i].pM)
//                print_params(Modules[i].pM,Modules[i].name,log_name);
//        }
//    }   /* Print module parameters. */
//
//    /* Run pipeline: */
//    RunBlobTrackingAuto( pCap, pTracker, fgavi_name, btavi_name );
//
//    {   /* Save state and release modules: */
//        CvFileStorage* fs = NULL;
//        if(savestate_name)
//        {
//            fs=cvOpenFileStorage(savestate_name,NULL,CV_STORAGE_WRITE);
//        }
//        if(fs)
//        {
//            cvStartWriteStruct(fs,"BlobTracker",CV_NODE_MAP);
//            if(param.pBT)param.pBT->SaveState(fs);
//            cvEndWriteStruct(fs);
//            cvStartWriteStruct(fs,"BlobTrackerAuto",CV_NODE_MAP);
//            if(pTracker)pTracker->SaveState(fs);
//            cvEndWriteStruct(fs);
//            cvStartWriteStruct(fs,"BlobTrackAnalyser",CV_NODE_MAP);
//            if(param.pBTA)param.pBTA->SaveState(fs);
//            cvEndWriteStruct(fs);
//            cvReleaseFileStorage(&fs);
//        }
//        if(param.pBT)cvReleaseBlobTracker(&param.pBT);
//        if(param.pBD)cvReleaseBlobDetector(&param.pBD);
//        if(param.pBTGen)cvReleaseBlobTrackGen(&param.pBTGen);
//        if(param.pBTA)cvReleaseBlobTrackAnalysis(&param.pBTA);
//        if(param.pFG)cvReleaseFGDetector(&param.pFG);
//        if(pTracker)cvReleaseBlobTrackerAuto(&pTracker);
//
//    }   /* Save state and release modules. */
//
//    if(pCap)
//        cvReleaseCapture(&pCap);
//    return 0;
//
//}   /* main() */
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

enum FRAME_STATUS { nochanged=0, up, down};
enum PEAK_STATUS { no, yes, maybe };
struct FrameInfo
{
    unsigned long nFrame;
    FRAME_STATUS status; //
    float v;
    PEAK_STATUS bpeak;
};
void show(MyList<FrameInfo>& list, Mat img)
{
    img = Mat::zeros(img.size(), CV_8U);
    Point p1;
    p1.x = 0;
    p1.y = 0;
    for(unsigned long k=list.getHead(); k<list.getTail(); ++k)
    {
        Point p2;
        p2.x = k-list.getHead();
        FrameInfo& v = list.get(k);
        p2.y = img.rows-v.v*img.rows;
        line(img, p1, p2, Scalar(128));
        if(v.bpeak==yes)
        {
            circle(img, p2, 1, Scalar(255), 2);
        }
        p1 = p2;
    }
}

PEAK_STATUS update(MyList<FrameInfo>& list)
{
    const unsigned long cFrame = list.getTail()-1;
    const unsigned long maxHistory = 45;
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
                v.status = up;
                v.bpeak = maybe;
            }
            else
            {
                unsigned int i=0;
                for(i=0; i<maxHistory; ++i)
                {
                    FrameInfo& tv = list.get(cFrame-1-i);
                    if(tv.bpeak == yes)
                    {
                        v.bpeak = maybe;
                        break;
                    }
                    else if(tv.bpeak == no)
                    {
                        v.bpeak = yes;
                        break;
                    }
                }
                if(i==maxHistory)
                {
                    v.bpeak = yes;
                }
            }
        }
        else
        {
            v.bpeak = no;
            v.status = nochanged;
        }

    }
    else
    { 
        v.bpeak = no;
        v.status = nochanged;
    }
    return v.bpeak;
}
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
    const int listSize = 1<<listBits;
    MyList<FrameInfo> info_list(listBits);
    Mat info(512, listSize, CV_8U);
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
        mask= Mat(bgfg->GetMask());
        multiply(mask, weight, mask, 1, CV_32F);
        FrameInfo v;
        v.v = sum(mask)[0]/(255*mask.rows*mask.cols);
        v.nFrame = nFrame; 
        info_list.add(v);
        PEAK_STATUS bpeak = update(info_list);
        
        show(info_list, info);
        Scalar color;
        if(bpeak==yes)
        {
            color = Scalar(0, 0, 255);
            cout<<nFrame<<'\t';
        }
        else if(bpeak==maybe)
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
