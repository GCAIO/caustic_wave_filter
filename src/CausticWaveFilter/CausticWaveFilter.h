#ifndef CAUSTICWAVEFILTER_H
#define CAUSTICWAVEFILTER_H

#include <opencv2/opencv.hpp>
#include "tictoc.h"

#define CWNUMTHREADS 4

class OpticalFlow
{
public:
    cv::Mat flow;
    int from;
    int to;
};

class CausticWaveFilter
{

private:

    std::vector<cv::Mat> imageVector;
    std::vector<OpticalFlow> opticalFlowVector;

    cv::Mat preProcessedImage;
    cv::Mat filteredImage;

    int param_frameWindowSize;

    void integrate_image(cv::Mat &src, cv::Mat &dst, cv::Mat &ini);

    inline bool isFlowCorrect(cv::Point2f u);
    cv::Vec3b computeColor(float fx, float fy);
    void drawOpticalFlow(const cv::Mat_<cv::Point2f>& flow, cv::Mat& dst, float maxmotion = -1);
    void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step, const cv::Scalar& color, float escala);
    void updateImageVector(cv::Mat &_frame);
    cv::Mat preProcessing(cv::Mat &rawFrame);
    cv::Mat removeNoise(cv::Mat noisyImage, cv::Mat notSoNoisyImage);
    cv::Mat removeNoise2(cv::Mat noisyImage, cv::Mat notSoNoisyImage);

    int param_adaptiveThresh;
    cv::Mat removeNoise3(cv::Mat noisyImage, cv::Mat notSoNoisyImage);

    void updateOpticalFlowVector();
    cv::Mat filter();
    void feedbackFilteredImage(cv::Mat &filteredImage);

    tic_toc elapsedTimeCalculator;
    bool calculateElapsedTime;

public:

    CausticWaveFilter();
    void setParam(int frameWindowSize);
    void updateFrame(cv::Mat &frame);
    cv::Mat getFilteredImage();
    cv::Mat getLastUnfilteredImage();
};

#endif

