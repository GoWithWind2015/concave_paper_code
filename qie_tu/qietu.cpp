#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>    
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
 
cv::Mat inverseColor4(cv::Mat srcImage)
{
	cv::Mat tempImage = srcImage.clone();
	// 初始化源图像迭代器
	cv::MatConstIterator_<cv::Vec3b> srcIterStart  = srcImage.begin<cv::Vec3b>();
	cv::MatConstIterator_<cv::Vec3b> srcIterEnd = srcImage.end<cv::Vec3b>();
    // 初始化输出图像迭代器
	cv::MatIterator_<cv::Vec3b> resIterStart = tempImage.begin<cv::Vec3b>();
	cv::MatIterator_<cv::Vec3b> resIterEnd = tempImage.end<cv::Vec3b>();
	// 遍历图像反色处理
	while( srcIterStart != srcIterEnd )
	{
		 (*resIterStart)[0] = 255 - (*srcIterStart)[0];
		 (*resIterStart)[1] = 255 - (*srcIterStart)[1];
		 (*resIterStart)[2] = 255 - (*srcIterStart)[2];
		 // 迭代器递增
		 srcIterStart++;
		 resIterStart++;
	}
	return tempImage;
}
 
int main()
{
        // 装载图像
        cv::Mat srcImage = cv::imread("F:/material/images/P0028-flower-02.jpg");
        cv::Mat dstImage;
 
        if (!srcImage.data)
                return -1;
        cv::imshow("srcImage", srcImage);
 
		dstImage = srcImage.clone();
 
        dstImage = inverseColor4(srcImage);
 
        cv::imshow("dstImage", dstImage);
 
        cv::waitKey(0);
        return 0;
}
