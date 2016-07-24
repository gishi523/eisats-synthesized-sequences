#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <opencv2/opencv.hpp>

namespace {

	std::string numberString(int n)
	{
		std::stringstream ss;
		ss << std::setw(3) << std::setfill('0') << n;
		return ss.str();
	}

	cv::Mat readRaw(const std::string& filename)
	{
		std::ifstream ifs(filename, std::ios::binary);
		if (ifs.fail())
			return cv::Mat();

		std::string line;
		while (getline(ifs, line) && line[0] == '#') {}

		int width, height, depth;
		sscanf(line.c_str(), "%d %d %d", &width, &height, &depth);
		CV_Assert(depth == 32);

		cv::Mat img(height, width, CV_32F);
		ifs.read((char *)img.data, sizeof(float) * width * height);

		return img;
	}

	void readEgoMotion(const std::string& filename, cv::Matx33d& R, cv::Matx31d& t)
	{
		std::ifstream ifs(filename);
		if (ifs.fail())
			return;

		std::string line;
		while (getline(ifs, line) && line[0] == '#') {}

		sscanf(line.c_str(), "%lf %lf %lf %lf", &R(0, 0), &R(0, 1), &R(0, 2), &t(0)); getline(ifs, line);
		sscanf(line.c_str(), "%lf %lf %lf %lf", &R(1, 0), &R(1, 1), &R(1, 2), &t(1)); getline(ifs, line);
		sscanf(line.c_str(), "%lf %lf %lf %lf", &R(2, 0), &R(2, 1), &R(2, 2), &t(2));
	}

	void drawOpticalFlow(cv::Mat& img, const cv::Mat1f& flowx, const cv::Mat1f& flowy)
	{
		float maxnorm = 0;
		for (int i = 0; i < flowx.rows; ++i)
		{
			for (int j = 0; j < flowx.cols; ++j)
			{
				float x = flowx(i, j);
				float y = flowy(i, j);
				maxnorm = std::max(maxnorm, sqrtf(x * x + y * y));
			}
		}

		img.create(flowx.size(), CV_8UC3);
		for (int i = 0; i < flowx.rows; ++i)
		{
			for (int j = 0; j < flowx.cols; ++j)
			{
				float x = flowx(i, j);
				float y = flowy(i, j);

				// Convert flow angle to hue
				float angle = atan2f(y, x);
				if (angle < 0.f) angle += (float)(2 * CV_PI);
				angle /= (float)(2 * CV_PI);
				uchar hue = static_cast<uchar>(180 * angle);

				// Convert flow norm to saturation
				float norm = sqrtf(x * x + y * y) / maxnorm;
				uchar sat = static_cast<uchar>(255 * norm);

				img.at<cv::Vec3b>(i, j) = cv::Vec3b(hue, sat, 255);
			}
		}

		cv::cvtColor(img, img, cv::COLOR_HSV2BGR);
	}

} /* namespace */

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << " [data path]" << std::endl;
		return -1;
	}

	std::string dataPath(argv[1]);

	for (int frameNo = 2; ; ++frameNo)
	{
		// Read left image
		cv::Mat leftImg = cv::imread(dataPath + "/colour-left-S2/img_c0_" + numberString(frameNo) + ".ppm");
		if (leftImg.empty())
			break;

		// Read optical flow map
		cv::Mat flowx = readRaw(dataPath + "/flowX-S2/flowU_from_" + numberString(frameNo - 1) + "_to_" + numberString(frameNo) + ".raw");
		cv::Mat flowy = readRaw(dataPath + "/flowY-S2/flowV_from_" + numberString(frameNo - 1) + "_to_" + numberString(frameNo) + ".raw");
		if (flowx.empty() || flowy.empty())
			break;

		// Read disparity map
		cv::Mat disp = readRaw(dataPath + "/disparityGT-S2/stereo_" + numberString(frameNo) + ".raw");

		// Read ego-motion
		cv::Matx33d R;
		cv::Matx31d t, r;
		readEgoMotion(dataPath + "/egoMotion/from_" + numberString(frameNo - 1) + "_to_" + numberString(frameNo) + ".txt", R, t);
		cv::Rodrigues(R, r);

		// Draw and display image
		cv::Mat flowImg, dispImg;
		drawOpticalFlow(flowImg, flowx, flowy);
		cv::normalize(disp, dispImg, 0.0, 1.0, cv::NORM_MINMAX);

		cv::putText(leftImg, "Rotation: " + std::to_string(r(0)) + " " + std::to_string(r(1)) + " " + std::to_string(r(2)),
			cv::Point(10, 20), 0, 0.5, cv::Scalar(0, 255, 255));
		cv::putText(leftImg, "Translation: " + std::to_string(t(0)) + " " + std::to_string(t(1)) + " " + std::to_string(t(2)),
			cv::Point(10, 40), 0, 0.5, cv::Scalar(0, 255, 255));

		cv::imshow("Left image", 16 * leftImg);
		cv::imshow("Flow image", flowImg);
		cv::imshow("Disp image", dispImg);

		char c = cv::waitKey(100);
		if (c == 27) // ESC
			break;
	}

	return 0;
}
