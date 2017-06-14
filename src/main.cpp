#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sstream>
#include <boost/assign/list_of.hpp>

#include "CausticWaveFilter/CausticWaveDenseFilter.h"
#include "CausticWaveFilter/CausticWaveFilter.h"

cv::Mat frame;
bool frameUpdated = false;

void newImageCb(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;

    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_ptr->image.copyTo(frame);
    frameUpdated = true;
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "CausticWaveFilter");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    image_transport::Publisher filteredImagePublisher = it.advertise("/CausticWaveFilter/image_filtered", 1);
    image_transport::Subscriber imageSubscriber = it.subscribe("image", 1, &newImageCb);

    CausticWaveFilter causticWaveFilter;
    CausticWaveDenseFilter causticWaveDenseFilter;

    ros::Rate r(30);
    while (nh.ok()) {
        if(frameUpdated)
        {
            causticWaveDenseFilter.updateFrame(frame);
            //cv::Mat filteredImage = causticWaveFilter->getFilteredImage();
            cv::Mat filteredImage = causticWaveDenseFilter.getFilteredImage();

            sensor_msgs::ImagePtr filteredImageMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", filteredImage).toImageMsg();
            filteredImagePublisher.publish(filteredImageMsg);
        }
        ros::spinOnce();
        r.sleep();
    }
}
