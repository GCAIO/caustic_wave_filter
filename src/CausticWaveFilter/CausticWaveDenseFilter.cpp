#include "CausticWaveDenseFilter.h"

inline float truncateFloatIntoBounds(float number, float lowerBound, float higherBound)
{
    if(number > higherBound)
    {
        number = higherBound;
    }
    else
    {
        if(number < lowerBound)
        {
            number = lowerBound;
        }
    }
    return number;
}

CausticWaveDenseFilter::CausticWaveDenseFilter()
{
    param_frameWindowSize = 3;

    calculateElapsedTime = false;
    initialized = false;

    mapper =  new cv::reg::MapperGradProj();
    mappPyr = new cv::reg::MapperPyramid(*mapper);

    //mappPyr->numIterPerScale_ = 1;
    //mappPyr->numLev_ = 3;

    mapProj = 0;

    running = true;

    for(int id = 0; id < FILTERXSIZE*FILTERYSIZE; id++)
    {
        filterWorkerThread[id] = boost::thread(&CausticWaveDenseFilter::filterWorkerRunLoop, this, id);
        filterWorkerIsDone[id] = true;
    }

    nnl = true;
}

void CausticWaveDenseFilter::setParam(int frameWindowSize)
{
    param_frameWindowSize = frameWindowSize;
}

void CausticWaveDenseFilter::init(const cv::Mat &frame)
{
    preProcessedImage = preProcessing(frame);
    imageVector.push_back(preProcessedImage);
    for(int i =0; i < param_frameWindowSize-1; i++)
    {
        imageVector.push_back(preProcessedImage);
        cv::Mat homography = calculateHomography();
        homographyVector.push_back(homography);

        cv::Mat of = calculateOpticalFlow();
        opticalFlowVector.push_back(of);
    }
    filteredImage = preProcessedImage.clone();
    denoisedImage = preProcessedImage.clone();

    cols = frame.cols;
    rows = frame.rows;

    for(int y = 0; y < rows; y++)
    {
        for(int x = 0; x < cols; x++)
        {
            std::vector<uchar> initChain;
            for(int z = 0; z < param_frameWindowSize; z++)
            {
                initChain.push_back(preProcessedImage.at<uchar>(y,x));
            }
            imageChains.push_back(initChain);
        }
    }
    newImageChains = imageChains;
}

void CausticWaveDenseFilter::updateFrame(const cv::Mat &frame)
{
    if(!initialized)
    {
        init(frame);
        initialized = true;
    }

    preProcessedImage = preProcessing(frame);

    if(calculateElapsedTime) elapsedTimeCalculator.tic();
    cv::Mat predicted;
    predicted = filteredImage;
    //predicted = makeImagePrediction(filteredImage);
    //mapProj->inverseWarp(filteredImage, predicted);
    denoisedImage = removeNoise(preProcessedImage, predicted);
    //denoisedImage = preProcessedImage;
    if(calculateElapsedTime) printf("removeNoise %f\n", elapsedTimeCalculator.toc());

    if(calculateElapsedTime) elapsedTimeCalculator.tic();
    updateImageVector(denoisedImage, nnl);
    if(calculateElapsedTime) printf("updateImageVector %f\n", elapsedTimeCalculator.toc());

    if(calculateElapsedTime) elapsedTimeCalculator.tic();
    cv::Mat homography = calculateHomography();
    updateHomographyVector(homography, nnl);
    if(calculateElapsedTime) printf("updateHomographyVector %f\n", elapsedTimeCalculator.toc());

//    if(calculateElapsedTime) elapsedTimeCalculator.tic();
//    cv::Mat opticalFlow = calculateOpticalFlow();
//    updateOpticalFlowVector(opticalFlow, nnl);
//    if(calculateElapsedTime) printf("updateOpticalFlowVector %f\n", elapsedTimeCalculator.toc());

    if(needNewLink(homography)){
    //if(needNewLink2(opticalFlow)){
        nnl = true;
    }
    else{
        nnl = false;
    }


    if(calculateElapsedTime) elapsedTimeCalculator.tic();
    filter();
    //filterThreaded();
    if(calculateElapsedTime) printf("filter %f\n", elapsedTimeCalculator.toc());


    //updateImageVector(filteredImage, false);

    //denoisedImage.copyTo(filteredImage);

    //        cv::imshow("preProcessed", preProcessedImage);
    //        cv::imshow("denoised", denoisedImage);
    //        cv::imshow("filtered", filtered);
    //        cv::waitKey(1);
}

cv::Mat CausticWaveDenseFilter::makeImagePrediction(const cv::Mat &rawImage)
{
    cv::Mat predicted(rows, cols, CV_8UC1, cv::Scalar(0));
    for(int x = 0; x < cols; x++)
    {
        for(int y = 0; y < rows; y++)
        {
            cv::Point2f trackedPoint; trackedPoint.x = x; trackedPoint.y = y;
            cv::Mat flow = opticalFlowVector.at(param_frameWindowSize-2);
            cv::Point2f nextTrackedPoint = trackedPoint - flow.at<cv::Point2f>(trackedPoint.y, trackedPoint.x);
            nextTrackedPoint.x = truncateFloatIntoBounds(nextTrackedPoint.x, 0, cols - 1);
            nextTrackedPoint.y = truncateFloatIntoBounds(nextTrackedPoint.y, 0, rows - 1);
            predicted.at<uchar>(y,x) = rawImage.at<uchar>(nextTrackedPoint.y,nextTrackedPoint.x);
        }
    }
    return predicted;
}

cv::Mat CausticWaveDenseFilter::preProcessing(const cv::Mat &rawImage)
{
    cv::Mat frame_gray;
    cv::cvtColor(rawImage,frame_gray,cv::COLOR_BGR2GRAY);
    return frame_gray;
}

cv::Mat CausticWaveDenseFilter::removeNoise(const cv::Mat &noisyImage, const cv::Mat &_notSoNoisyImage)
{
    cv::Mat notSoNoisyImage;
    notSoNoisyImage = _notSoNoisyImage;

    int kernelSize = 5;//noisyImage.cols/32;
    cv::Mat noise;
    cv::subtract(noisyImage, notSoNoisyImage, noise);

    cv::Mat zeroMask(rows, cols, CV_8UC1);
    zeroMask = noise.clone();
    //zeroMask.setTo(0);
    //int border = 10;
    //noise(cv::Rect(border, border, cols-border*2, rows-border*2)).copyTo(zeroMask(cv::Rect(border, border, cols-border*2, rows-border*2)));

    cv::Mat noiseThresh;
    //cv::adaptiveThreshold(noise, noiseThresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_TOZERO, 11, 2);
    cv::threshold(zeroMask, noiseThresh, 0, 255, cv::THRESH_OTSU + cv::THRESH_TOZERO);
    //noiseThresh = noise;

    cv::Mat filteredNoise;
    //cv::GaussianBlur(noiseThresh, filteredNoise, cv::Size(kernelSize,kernelSize), 0.0);
    cv::blur(noiseThresh, filteredNoise, cv::Size(kernelSize,kernelSize));
    //cv::bilateralFilter( noiseThresh, filteredNoise, kernelSize, kernelSize*2, kernelSize/2);
    //filteredNoise = noise;

    cv::Mat filteredImage = noisyImage.clone();
    cv::subtract(noisyImage, filteredNoise, filteredImage, (zeroMask !=0 ));
    //cv::subtract(noisyImage, noiseThresh, filteredImage);

//    cv::imshow("noise", noise);
//    cv::imshow("noise thresh", noiseThresh);
//    cv::imshow("filtered noise", filteredNoise);
//    cv::imshow("filtered image", filteredImage);
//    cv::waitKey(1);

    return filteredImage;
}

void CausticWaveDenseFilter::updateImageVector(const cv::Mat &frame, bool newLink)
{
    // imageVector size = param_frameWindowSize
    if(newLink)
    {
        std::vector<cv::Mat> newImageVector;
        for(int x = 0; x < param_frameWindowSize-1; x++)
        {
            cv::Mat newImage;
            newImage = imageVector.at(x+1);
            newImageVector.push_back(newImage);
        }
        newImageVector.push_back(frame);

        imageVector = newImageVector;
    }
    else
    {
        imageVector.at(param_frameWindowSize-1) = frame;
    }
}

cv::Mat CausticWaveDenseFilter::calculateHomography()
{
    cv::Mat currentFrame, lastFrame;
    imageVector.at(imageVector.size()-1).convertTo(currentFrame, CV_64F);
    imageVector.at(imageVector.size()-2).convertTo(lastFrame, CV_64F);

    cv::Ptr<cv::reg::Map> mapPtr_local;
    mapPtr_global.release();
    mappPyr->calculate(currentFrame, lastFrame, mapPtr_global);

    // Print result
    mapProj = dynamic_cast<cv::reg::MapProjec*>(mapPtr_global.get());
    //mapProj->normalize();

    cv::Mat homography = cv::Mat(mapProj->getProjTr());

    return homography;
}

cv::Mat CausticWaveDenseFilter::calculateOpticalFlow()
{
    cv::Mat opticalFlow;
    //cv::calcOpticalFlowFarneback(imageVector.at(imageVector.size()-1), imageVector.at(imageVector.size()-2), opticalFlow, 0.5, 3, 15, 3, 5, 1.2, 0);
    cv::calcOpticalFlowFarneback(imageVector.at(imageVector.size()-1), imageVector.at(imageVector.size()-2), opticalFlow, 0.5, 3, 30, 3, 5, 1.2, /*cv::OPTFLOW_USE_INITIAL_FLOW*/0);
    return opticalFlow;
}

bool CausticWaveDenseFilter::needNewLink(const cv::Mat &homography)
{
    cv::Point3f translation;
    translation.x = homography.at<double>(0,2);
    translation.y = homography.at<double>(1,2);
    translation.z = homography.at<double>(2,2);

    //printf("translation norm %f\n", translation.norm());

    //std::cout << homography << std::endl;

    if(cv::norm(translation) > 5.0)
        return true;
    else
        return false;
}

bool CausticWaveDenseFilter::needNewLink2(const cv::Mat &opticalFlow)
{
    cv::Point2f translation;
    translation.x = opticalFlow.at<cv::Point2f>(rows/2, cols/2).x;
    translation.y = opticalFlow.at<cv::Point2f>(rows/2, cols/2).y;

    //printf("translation norm %f\n", translation.norm());

    //std::cout << homography << std::endl;

    if(cv::norm(translation) > 5.0)
        return true;
    else
        return false;
}

void CausticWaveDenseFilter::updateHomographyVector(const cv::Mat &homography, bool newLink)
{
    // calcular el of de to a from, en vez de from a to. Esto da un of invertido, pero va a permitir obtener la traza de los pixeles empezando desde
    // to, y yendo hasta la ultima imagen

    // homography vector size = param_frameWindowSize - 1

    if(newLink)
    {
        std::vector<cv::Mat> newHomographyVector;
        for(int x = 0; x < param_frameWindowSize-2; x++)
        {
            newHomographyVector.push_back(homographyVector.at(x+1));
        }
        newHomographyVector.push_back(homography);

        homographyVector = newHomographyVector;
    }
    else
    {
        homographyVector.at(param_frameWindowSize - 2) = homography;
    }

}

void CausticWaveDenseFilter::updateOpticalFlowVector(const cv::Mat &opticalFlow, bool newLink)
{
    // calcular el of de to a from, en vez de from a to. Esto da un of invertido, pero va a permitir obtener la traza de los pixeles empezando desde
    // to, y yendo hasta la ultima imagen

    // homography vector size = param_frameWindowSize - 1

    if(newLink)
    {
        std::vector<cv::Mat> newOpticalFlowVector;
        for(int x = 0; x < param_frameWindowSize-2; x++)
        {
            newOpticalFlowVector.push_back(opticalFlowVector.at(x+1));
        }
        newOpticalFlowVector.push_back(opticalFlow);

        opticalFlowVector = newOpticalFlowVector;
    }
    else
    {
        opticalFlowVector.at(param_frameWindowSize - 2) = opticalFlow;
    }

}

void CausticWaveDenseFilter::filter()
{
    for(int id = 0; id < FILTERXSIZE*FILTERYSIZE; id++)
    {
        int roiWidth = cols/FILTERXSIZE;
        int roiHeight = rows/FILTERYSIZE;

        int y = id/FILTERXSIZE;
        int x = id - y*FILTERXSIZE;

        cv::Rect roi(x*roiWidth, y*roiHeight, roiWidth, roiHeight);
        filterROI(roi);
        //filterROI3(roi);
    }
    imageChains = newImageChains;
}

void CausticWaveDenseFilter::filterThreaded()
{
    boost::unique_lock<boost::mutex> lock(filterWorkerExMutex);

    for(int id = 0; id < FILTERXSIZE*FILTERYSIZE; id++)
        filterWorkerIsDone[id] = false;

    filter_worker_todo_signal.notify_all();

    // wait for all worker threads to signal they are done.
    while(true)
    {
        // wait for at least one to finish
        filter_worker_done_signal.wait(lock);
        //printf("thread finished!\n");

        // check if actually all are finished.
        bool allDone = true;
        for(int i=0; i<FILTERXSIZE*FILTERYSIZE; i++)
            allDone = allDone && filterWorkerIsDone[i];

        // all are finished! exit.
        if(allDone)
            break;
    }

    imageChains = newImageChains;
}

void CausticWaveDenseFilter::filterWorkerRunLoop(int id)
{
    boost::unique_lock<boost::mutex> lock(filterWorkerExMutex);

    while(running)
    {
        filter_worker_todo_signal.wait(lock);
        // if got something: do it (unlock in the meantime)
        lock.unlock();

        int roiWidth = cols/FILTERXSIZE;
        int roiHeight = rows/FILTERYSIZE;

        int y = id/FILTERXSIZE;
        int x = id - y*FILTERXSIZE;

        cv::Rect roi(x*roiWidth, y*roiHeight, roiWidth, roiHeight);
        //filterROI(roi);
        filterROI3(roi);

        lock.lock();

        filterWorkerIsDone[id] = true;
        //printf("worker %d waiting..\n", idx);
        filter_worker_done_signal.notify_all();
    }
}

void CausticWaveDenseFilter::filterROI(cv::Rect roi)
{
    cv::Mat filtered(roi.height, roi.width, CV_8UC1, cv::Scalar(0));
    for(int x = 0; x < roi.width; x++)
    {
        for(int y = 0; y < roi.height; y++)
        {
            int filteredPixelValue = 0;
            cv::Point2f trackedPoint;
            trackedPoint.x = x + roi.x; trackedPoint.y = y + roi.y;

            filteredPixelValue = filteredPixelValue + imageVector.at(param_frameWindowSize-1).at<uchar>(trackedPoint.y, trackedPoint.x);

            for(int z = param_frameWindowSize - 2; z >= 0; z--)
            {
                cv::Mat homography = homographyVector.at(z);
                cv::Point2f nextTrackedPoint;
                float depth        = (homography.at<double>(2,0)*trackedPoint.x + homography.at<double>(2,1)*trackedPoint.y + homography.at<double>(2,2));
                nextTrackedPoint.x = (homography.at<double>(0,0)*trackedPoint.x + homography.at<double>(0,1)*trackedPoint.y + homography.at<double>(0,2))/depth;
                nextTrackedPoint.y = (homography.at<double>(1,0)*trackedPoint.x + homography.at<double>(1,1)*trackedPoint.y + homography.at<double>(1,2))/depth;
                nextTrackedPoint.x = truncateFloatIntoBounds(nextTrackedPoint.x, 0, cols - 1);
                nextTrackedPoint.y = truncateFloatIntoBounds(nextTrackedPoint.y, 0, rows - 1);
                trackedPoint = nextTrackedPoint;

                filteredPixelValue = filteredPixelValue + imageVector.at(z).at<uchar>(trackedPoint.y, trackedPoint.x);
            }
            filteredPixelValue = filteredPixelValue/float(param_frameWindowSize);
            filtered.at<uchar>(y, x) = filteredPixelValue;
        }
    }
    filtered.copyTo(filteredImage(roi));
}

void CausticWaveDenseFilter::filterROI3(cv::Rect roi)
{
    cv::Mat filtered(roi.height, roi.width, CV_8UC1, cv::Scalar(0));
    for(int x = 0; x < roi.width; x++)
    {
        for(int y = 0; y < roi.height; y++)
        {
            int filteredPixelValue = 0;
            cv::Point2f trackedPoint;
            trackedPoint.x = x + roi.x; trackedPoint.y = y + roi.y;

            filteredPixelValue = filteredPixelValue + imageVector.at(param_frameWindowSize-1).at<uchar>(trackedPoint.y, trackedPoint.x);

            for(int z = imageVector.size() - 2; z >= 0; z--)
            {
                cv::Mat flow = opticalFlowVector.at(z);
                cv::Point2f nextTrackedPoint = trackedPoint + flow.at<cv::Point2f>(trackedPoint.y, trackedPoint.x);
                nextTrackedPoint.x = truncateFloatIntoBounds(nextTrackedPoint.x, 0, cols - 1);
                nextTrackedPoint.y = truncateFloatIntoBounds(nextTrackedPoint.y, 0, rows - 1);
                trackedPoint = nextTrackedPoint;

                filteredPixelValue = filteredPixelValue + imageVector.at(z).at<uchar>(trackedPoint.y, trackedPoint.x);
            }
            filteredPixelValue = filteredPixelValue/float(param_frameWindowSize);
            filtered.at<uchar>(y,x) = filteredPixelValue;
        }
    }
    filtered.copyTo(filteredImage(roi));
}

void CausticWaveDenseFilter::filterROI2(cv::Rect roi)
{
    cv::Mat filtered(roi.height, roi.width, CV_8UC1, cv::Scalar(0));
    for(int x = 0; x < roi.width; x++)
    {
        for(int y = 0; y < roi.height; y++)
        {
            int filteredPixelValue = 0;
            cv::Point2f trackedPoint;
            trackedPoint.x = x + roi.x; trackedPoint.y = y + roi.y;

            //actualiza la cadena

            cv::Mat homography = homographyVector.at(param_frameWindowSize-2);
            cv::Point2f nextTrackedPoint;
            float depth        = (homography.at<double>(2,0)*trackedPoint.x + homography.at<double>(2,1)*trackedPoint.y + homography.at<double>(2,2));
            nextTrackedPoint.x = (homography.at<double>(0,0)*trackedPoint.x + homography.at<double>(0,1)*trackedPoint.y + homography.at<double>(0,2))/depth;
            nextTrackedPoint.y = (homography.at<double>(1,0)*trackedPoint.x + homography.at<double>(1,1)*trackedPoint.y + homography.at<double>(1,2))/depth;
            nextTrackedPoint.x = truncateFloatIntoBounds(nextTrackedPoint.x, 0, cols - 1);
            nextTrackedPoint.y = truncateFloatIntoBounds(nextTrackedPoint.y, 0, rows - 1);

            std::vector<uchar> oldChain = imageChains.at(nextTrackedPoint.y*rows + nextTrackedPoint.x);
            std::vector<uchar> newChain(oldChain.begin() + 1, oldChain.end());
            newChain.push_back(imageVector.at(param_frameWindowSize-1).at<uchar>(trackedPoint.y, trackedPoint.x));

            newImageChains.at(trackedPoint.y*rows + trackedPoint.x) = newChain;

            for(int z = 0; z < param_frameWindowSize; z++)
            {
                filteredPixelValue = filteredPixelValue + newChain.at(z);
            }
            filteredPixelValue = filteredPixelValue/float(param_frameWindowSize);
            filtered.at<uchar>(y, x) = filteredPixelValue;
        }
    }
    filtered.copyTo(filteredImage(roi));
}

cv::Mat CausticWaveDenseFilter::getFilteredImage()
{
    return filteredImage;
}

cv::Mat CausticWaveDenseFilter::getLastUnfilteredImage()
{
    return preProcessedImage;
}
