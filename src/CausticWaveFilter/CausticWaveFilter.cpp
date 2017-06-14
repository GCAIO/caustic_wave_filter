#include "CausticWaveFilter.h"

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

void CausticWaveFilter::integrate_image(cv::Mat &src, cv::Mat &dst, cv::Mat &ini) // src 8U o 32F dst 8U
{
    if(src.depth()!=CV_32F)
        printf("error no es flotante\n");

    cv::Mat integral_x(src.rows,src.cols,CV_32F,cv::Scalar(0));
    cv::Mat integral_y(src.rows,src.cols,CV_32F,cv::Scalar(0));

    cv::Mat col_aux;
    ini.copyTo(col_aux);

    cv::Mat row_aux;
    integral_y.row(0).copyTo(row_aux);

    cv::Mat col_offset(src.rows,1,CV_32F,cv::Scalar(127));
    cv::Mat row_offset(1,src.cols,CV_32F,cv::Scalar(127));

    cv::Mat aux_c(src.rows,1,CV_32F);
    cv::Mat aux_r(1,src.cols,CV_32F);

    for(int x=0;x<src.cols;x++)
    {
        //      subtract(src.col(x),col_offset,aux_c,noArray(),CV_32F);
        //      add(col_aux,aux_c,col_aux,noArray(),CV_32F);
        //      col_aux.copyTo(integral_x.col(x));

        cv::add(col_aux,src.col(x),col_aux,cv::noArray(),CV_32F);
        col_aux.copyTo(integral_x.col(x));
    }

    for(int y=0;y<src.rows;y++)
    {
        cv::subtract(src.row(y),row_offset,aux_r,cv::noArray(),CV_32F);
        cv::add(row_aux,aux_r,row_aux,cv::noArray(),CV_32F);
        row_aux.copyTo(integral_y.row(y));
    }

    cv::Mat integral;
    addWeighted( integral_x, 0.5, integral_y, 0.5, 0, integral);

    integral_x.convertTo(dst,CV_8UC1,0.10,0.0);
    //integral_x.copyTo(dst);
    return;
}

inline bool CausticWaveFilter::isFlowCorrect(cv::Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

cv::Vec3b CausticWaveFilter::computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static cv::Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = cv::Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = cv::Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float)CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    cv::Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.f;
        const float col1 = colorWheel[k1][b] / 255.f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.f * col);
    }

    return pix;
}

void CausticWaveFilter::drawOpticalFlow(const cv::Mat_<cv::Point2f>& flow, cv::Mat& dst, float maxmotion)
{
    dst.create(flow.size(), CV_8UC3);
    dst.setTo(cv::Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flow.rows; ++y)
        {
            for (int x = 0; x < flow.cols; ++x)
            {
                cv::Point2f u = flow(y, x);

                if (!isFlowCorrect(u))
                    continue;

                maxrad = cv::max(maxrad, cv::sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flow.rows; ++y)
    {
        for (int x = 0; x < flow.cols; ++x)
        {
            cv::Point2f u = flow(y, x);

            if (isFlowCorrect(u))
                dst.at<cv::Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

void CausticWaveFilter::drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step, const cv::Scalar& color,float escala)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
            cv::line(cflowmap, cv::Point(x,y), cv::Point(cvRound(x+fxy.x*escala), cvRound(y+fxy.y*escala)),
                     color);
            cv::circle(cflowmap, cv::Point(x,y), 2, color, -1);
        }
}

void CausticWaveFilter::updateImageVector(cv::Mat &_frame)
{
    cv::Mat frame;
    frame = _frame;

    if(imageVector.size() < param_frameWindowSize)
    {
        imageVector.push_back(frame);
    }
    else
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
}

cv::Mat CausticWaveFilter::preProcessing(cv::Mat &rawFrame)
{
    cv::Mat frame_gray;
    cv::cvtColor(rawFrame,frame_gray,cv::COLOR_BGR2GRAY);
    return frame_gray;
}

cv::Mat CausticWaveFilter::removeNoise(cv::Mat noisyImage, cv::Mat notSoNoisyImage)
{
    int kernelSize = noisyImage.cols/32;
    cv::Mat noise;
    cv::subtract(noisyImage, notSoNoisyImage, noise);

//        cv::imshow("noise", noise);

    cv::Mat filteredNoise;
    //cv::GaussianBlur(noise, filteredNoise, cv::Size(kernelSize,kernelSize), 0.0);
    //cv::blur(noise, filteredNoise, cv::Size(kernelSize,kernelSize));
    cv::bilateralFilter( noise, filteredNoise, kernelSize, kernelSize*2, kernelSize/2);
    //filteredNoise = noise;

    cv::imshow("filtered noise", filteredNoise);
    cv::waitKey(1);

    cv::Mat filteredImage;
    cv::subtract(noisyImage, filteredNoise, filteredImage);

    return filteredImage;
}

cv::Mat CausticWaveFilter::removeNoise2(cv::Mat noisyImage, cv::Mat notSoNoisyImage)
{
    int kernelSize = noisyImage.cols/32;
    cv::Mat noise;
    cv::subtract(noisyImage, notSoNoisyImage, noise);

    cv::Mat noiseThresh;
    //cv::adaptiveThreshold(noise, noiseThresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_TOZERO, 11, 2);
    cv::threshold(noise, noiseThresh, 0, 255, cv::THRESH_OTSU + cv::THRESH_TOZERO);

    cv::Mat filteredNoise;
    //cv::GaussianBlur(noise, filteredNoise, cv::Size(kernelSize,kernelSize), 0.0);
    cv::blur(noise, filteredNoise, cv::Size(kernelSize,kernelSize));
    //cv::bilateralFilter( noiseThresh, filteredNoise, kernelSize, kernelSize*2, kernelSize/2);
    //filteredNoise = noise;

    cv::Mat filteredImage = noisyImage.clone();
    cv::subtract(noisyImage, filteredNoise, filteredImage, (noise !=0 ));
    //cv::subtract(noisyImage, noiseThresh, filteredImage);

//    cv::imshow("noise", noise);
//    cv::imshow("noise thresh", noiseThresh);
//    cv::imshow("filtered noise", filteredNoise);
//    cv::waitKey(1);

    return filteredImage;
}

cv::Mat CausticWaveFilter::removeNoise3(cv::Mat noisyImage, cv::Mat notSoNoisyImage)
{
    int kernelSize = noisyImage.cols/32;
    cv::Mat noise;
    cv::subtract(noisyImage, notSoNoisyImage, noise);

    cv::Mat noiseThresh;
    cv::threshold(noise, noiseThresh, param_adaptiveThresh, 255, cv::THRESH_BINARY);



    cv::Mat filteredNoise;
    //cv::GaussianBlur(noise, filteredNoise, cv::Size(kernelSize,kernelSize), 0.0);
    //cv::blur(noise, filteredNoise, cv::Size(kernelSize,kernelSize));
    //cv::bilateralFilter( noise, filteredNoise, kernelSize, kernelSize*2, kernelSize/2);
    filteredNoise = noise;


    cv::Mat filteredImage;
    cv::subtract(noisyImage, filteredNoise, filteredImage);

    cv::Mat filteredImage2;
    noisyImage.copyTo(filteredImage2);
    filteredImage.copyTo(filteredImage2, noiseThresh);





    //adapt thresh
    int dilation_type = cv::MORPH_RECT;//MORPH_CROSS MORPH_ELLIPSE
    int dilation_size = 3;
    cv::Mat element = cv::getStructuringElement( dilation_type, cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                         cv::Point( dilation_size, dilation_size ) );

    cv::Mat noiseThreshDilated;
    cv::dilate(noiseThresh, noiseThreshDilated, element);

    cv::Mat noiseThreshSurround = noiseThreshDilated - noiseThresh;

    cv::Scalar filteredMean = cv::mean(filteredImage2, noiseThresh);
    cv::Scalar filteredSurroundMean = cv::mean(filteredImage2, noiseThreshSurround);


    std::cout << param_adaptiveThresh << std::endl;
    std::cout << filteredMean[0] << std::endl;
    std::cout << filteredSurroundMean[0] << std::endl;
    //

    if(filteredMean[0] < filteredSurroundMean[0])
        param_adaptiveThresh--;
    if(filteredMean[0] > filteredSurroundMean[0])
        param_adaptiveThresh++;
    if(fabs(filteredMean[0] - filteredSurroundMean[0]) < 10)
        param_adaptiveThresh++;



    cv::imshow("noise", noise);
    cv::imshow("noise thresh", noiseThresh);
    cv::imshow("noise thresh surround", noiseThreshSurround);
    cv::imshow("filtered noise", filteredNoise);
    cv::imshow("filtered image", filteredImage);
    cv::imshow("filtered image 2", filteredImage2);

    cv::waitKey(1);




    return filteredImage2;
}

void CausticWaveFilter::updateOpticalFlowVector()
{
    if(imageVector.size() < 2)
    {
        printf("updateOpticalFlowVector not enough frames\n");
        return;
    }
    else
    {
        OpticalFlow opticalFlow;
        // calcular el of de to a from, en vez de from a to. Esto da un of invertido, pero va a permitir obtener la traza de los pixeles empezando desde
        // to, y yendo hasta la ultima imagen

        cv::calcOpticalFlowFarneback(imageVector.at(imageVector.size()-1), imageVector.at(imageVector.size()-2), opticalFlow.flow, 0.5, 3, 200, 3, 7, 1.5, /*cv::OPTFLOW_USE_INITIAL_FLOW*/0);
        //cv::calcOpticalFlowFarneback(imageVector.at(imageVector.size()-1), imageVector.at(imageVector.size()-2), opticalFlow.flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        if(opticalFlowVector.size() < param_frameWindowSize)
        {
            opticalFlowVector.push_back(opticalFlow);
        }
        else
        {
            std::vector<OpticalFlow> newOpticalFlowVector;
            for(int x = 0; x < param_frameWindowSize-1; x++)
            {
                newOpticalFlowVector.push_back(opticalFlowVector.at(x+1));
            }
            newOpticalFlowVector.push_back(opticalFlow);

            opticalFlowVector = newOpticalFlowVector;
        }
    }
}

cv::Mat CausticWaveFilter::filter()
{
    cv::Mat filtered(imageVector.at(imageVector.size()-1).rows, imageVector.at(imageVector.size()-1).cols, CV_8UC1, cv::Scalar(0));
    for(int x = 0; x < filtered.cols; x++)
    {
        for(int y = 0; y < filtered.rows; y++)
        {
            int filteredPixelValue = 0;
            cv::Point2f trackedPoint;
            trackedPoint.x = x; trackedPoint.y = y;

            for(int z = imageVector.size() - 1; z >= 0; z--)
            {
                cv::Mat frame = imageVector.at(z);
                filteredPixelValue = filteredPixelValue + frame.at<uchar>(trackedPoint.y, trackedPoint.x);

                cv::Mat flow = opticalFlowVector.at(z).flow;
                cv::Point2f nextTrackedPoint = trackedPoint + flow.at<cv::Point2f>(trackedPoint.y, trackedPoint.x);
                nextTrackedPoint.x = truncateFloatIntoBounds(nextTrackedPoint.x, 0, filtered.cols - 1);
                nextTrackedPoint.y = truncateFloatIntoBounds(nextTrackedPoint.y, 0, filtered.rows - 1);
                trackedPoint = nextTrackedPoint;
            }
            filteredPixelValue = filteredPixelValue/float(imageVector.size());
            filtered.at<uchar>(y,x) = filteredPixelValue;
        }
    }
    return filtered;
}

void CausticWaveFilter::feedbackFilteredImage(cv::Mat &filteredImage)
{
    imageVector.at(imageVector.size()-1) = filteredImage;
}

CausticWaveFilter::CausticWaveFilter()
{
    param_frameWindowSize = 5;
    param_adaptiveThresh = 32;

    calculateElapsedTime = true;
}

void CausticWaveFilter::setParam(int frameWindowSize)
{
    param_frameWindowSize = frameWindowSize;
}

void CausticWaveFilter::updateFrame(cv::Mat &frame)
{
//    try
    {
        preProcessedImage = preProcessing(frame);

        cv::Mat denoisedImage;
        if(imageVector.size() > 0)
        {
            if(calculateElapsedTime) elapsedTimeCalculator.tic();
            denoisedImage = removeNoise2(preProcessedImage, imageVector.at(imageVector.size()-1));
            if(calculateElapsedTime) printf("removeNoise %f\n", elapsedTimeCalculator.toc());
            //denoisedImage = removeNoise2(preProcessedImage, filtered);
            //denoisedImage = removeNoise(preProcessedImage, filtered);
        }
        else
        {
            denoisedImage = preProcessedImage;
        }

        if(calculateElapsedTime) elapsedTimeCalculator.tic();
        updateImageVector(denoisedImage);
        if(calculateElapsedTime) printf("updateImageVector %f\n", elapsedTimeCalculator.toc());

        if(calculateElapsedTime) elapsedTimeCalculator.tic();
        updateOpticalFlowVector();
        if(calculateElapsedTime) printf("updateOpticalFlowVector %f\n", elapsedTimeCalculator.toc());

        if(calculateElapsedTime) elapsedTimeCalculator.tic();
        cv::Mat filtered = filter();
        if(calculateElapsedTime) printf("filter %f\n", elapsedTimeCalculator.toc());
        //feedbackFilteredImage(filtered);

        //for(int x = opticalFlowVector.size()-1; x >= 0; x--)
//            int x = opticalFlowVector.size()-1;
//            {
//                cv::Mat flow = opticalFlowVector.at(x).flow;
//                cv::Mat flow_vector, flow_color;
//                cvtColor(imageVector.at(0), flow_vector, cv::COLOR_GRAY2BGR);
//                drawOptFlowMap(flow, flow_vector, 16, cv::Scalar(0, 255, 0),10.0);
//                drawOpticalFlow(flow, flow_color);

//                std::string flow_vector_window_name = "flow vector " + std::to_string(x);
//                std::string flow_color_window_name = "flow color " + std::to_string(x);
//                cv::imshow(flow_vector_window_name, flow_vector);
//                cv::imshow(flow_color_window_name, flow_color);
//                cv::waitKey(1);
//            }

        filtered.copyTo(filteredImage);
        //denoisedImage.copyTo(filteredImage);

//        cv::imshow("preProcessed", preProcessedImage);
//        cv::imshow("denoised", denoisedImage);
//        cv::imshow("filtered", filtered);
//        cv::waitKey(1);
    }
//    catch(exception &e)
//    {
//        std::cout << e.what() << std::endl;
//    }
}

cv::Mat CausticWaveFilter::getFilteredImage()
{
    return filteredImage;
}

cv::Mat CausticWaveFilter::getLastUnfilteredImage()
{
    return preProcessedImage;
}
