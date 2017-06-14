#ifndef CAUSTICWAVEDENSEFILTER_H
#define CAUSTICWAVEDENSEFILTER_H

#include <opencv2/opencv.hpp>

#include <opencv2/reg/mapaffine.hpp>
#include <opencv2/reg/mapshift.hpp>
#include <opencv2/reg/mapprojec.hpp>
#include <opencv2/reg/mappergradshift.hpp>
#include <opencv2/reg/mappergradeuclid.hpp>
#include <opencv2/reg/mappergradsimilar.hpp>
#include <opencv2/reg/mappergradaffine.hpp>
#include <opencv2/reg/mappergradproj.hpp>
#include <opencv2/reg/mapperpyramid.hpp>

#include <boost/thread.hpp>

#include "tictoc.h"

#define FILTERXSIZE 2
#define FILTERYSIZE 2

class CausticWaveDenseFilter
{

private:

    std::vector<cv::Mat> imageVector;
    std::vector<cv::Mat> homographyVector;
    std::vector<cv::Mat> opticalFlowVector;

    std::vector<std::vector<uchar> > imageChains;
    std::vector<std::vector<uchar> > newImageChains;

    cv::Mat preProcessedImage;
    cv::Mat filteredImage;
    cv::Mat denoisedImage;

    int cols, rows;

    cv::reg::MapperGradProj *mapper;
    cv::reg::MapperPyramid *mappPyr;
    cv::Ptr<cv::reg::Map> mapPtr_global;

    cv::reg::MapProjec* mapProj;

    int param_frameWindowSize;

    void updateImageVector(const cv::Mat &frame, bool newLink);
    cv::Mat preProcessing(const cv::Mat &rawImage);
    cv::Mat makeImagePrediction(const cv::Mat &rawImage);
    cv::Mat removeNoise(const cv::Mat &noisyImage, const cv::Mat &_notSoNoisyImage);

    cv::Mat calculateHomography();
    cv::Mat calculateOpticalFlow();
    void updateHomographyVector(const cv::Mat &homography, bool newLink);
    void updateOpticalFlowVector(const cv::Mat &opticalFlow, bool newLink);
    bool needNewLink(const cv::Mat &homography);
    bool needNewLink2(const cv::Mat &opticalFlow);
    bool nnl;
    void filter();
    void filterThreaded();
    void filterROI(cv::Rect roi);
    void filterROI2(cv::Rect roi);
    void filterROI3(cv::Rect roi);
    void filterWorkerRunLoop(int id);

    tic_toc elapsedTimeCalculator;

    bool calculateElapsedTime;
    bool initialized;

    //optimizea worker runloop;
    bool running;
    boost::thread filterWorkerThread[FILTERXSIZE*FILTERYSIZE];
    bool filterWorkerIsDone[FILTERXSIZE*FILTERYSIZE];
    boost::mutex filterWorkerExMutex;
    boost::condition_variable filter_worker_todo_signal;
    boost::condition_variable filter_worker_done_signal;

public:

    CausticWaveDenseFilter();
    void setParam(int frameWindowSize);
    void updateFrame(const cv::Mat &frame);
    void init(const cv::Mat &frame);
    cv::Mat getFilteredImage();
    cv::Mat getLastUnfilteredImage();
};

#endif

