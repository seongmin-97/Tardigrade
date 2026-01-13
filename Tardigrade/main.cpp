#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

int main() {
    // 1. Eigen 테스트: 3x3 단위 행렬 생성 및 출력
    std::cout << "--- Eigen Test ---" << std::endl;
    Eigen::Matrix3d m = Eigen::Matrix3d::Identity();
    m(0, 2) = 2.5;
    std::cout << "Identity Matrix with a twist:\n" << m << "\n\n";

    // 2. OpenCV 테스트: 빈 이미지 생성 및 텍스트 쓰기
    std::cout << "--- OpenCV Test ---" << std::endl;
    // 400x600 크기의 3채널(BGR) 파란색 배경 이미지 생성
    cv::Mat image = cv::Mat(400, 600, CV_8UC3, cv::Scalar(255, 0, 0));

    if (image.empty()) {
        std::cerr << "OpenCV Error: Could not create image." << std::endl;
        return -1;
    }

    // 이미지에 텍스트 넣기
    cv::putText(image, "Tardigrade Engine Test", cv::Point(50, 200),
        cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 2);

    std::cout << "OpenCV window will open now..." << std::endl;

    // 창 띄우기
    cv::imshow("Hello Tardigrade!", image);

    // 3. 통합 시나리오: OpenCV 픽셀 데이터를 Eigen으로 읽기 (맛보기)
    // 이미지 중앙의 한 점(BGR)을 Eigen 벡터에 담아보기
    cv::Vec3b pixel = image.at<cv::Vec3b>(200, 300);
    Eigen::Vector3d eigenPixel(pixel[0], pixel[1], pixel[2]);
    std::cout << "Center pixel color (Eigen Vector): \n" << eigenPixel << std::endl;

    // 아무 키나 누를 때까지 대기
    cv::waitKey(0);

    return 0;
}