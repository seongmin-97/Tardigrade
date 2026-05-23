#pragma once
#include <string>
#include <vector>
#include <random>
#include <filesystem>
#include <unordered_set>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include <opencv2/opencv.hpp>

#include "Tensor.hpp"

namespace tardigrade::data
{
    /**
     * @brief 데이터 로딩 전략
     *
     * EAGER: 전체 데이터를 RAM에 적재 (중소규모 데이터셋, 빠른 학습)
     * LAZY:  매 접근 시 디스크에서 읽음 (대규모 데이터셋, 메모리 절약)
     */
    enum class LoadStrategy
    {
        EAGER,
        LAZY
    };

    /// 이미지 리사이즈 목표 크기 (0이면 원본 유지)
    struct MatSize
    {
        int row;
        int col;
    };

    namespace fs = std::filesystem;

    /**
     * @brief DataLoader — 데이터 저장/로딩에 집중하는 클래스
     *
     * 디렉토리 구조에서 이미지 데이터셋을 로딩하고,
     * 셔플/배치 접근 등 학습 파이프라인에 필요한 기능을 제공한다.
     *
     * 디렉토리 규약: rootDir/{label}/*.{jpg,png,...}
     */
    class DataLoader
    {
    public:
        DataLoader(LoadStrategy strategy = LoadStrategy::EAGER);

        /**
         * @brief 이미지 데이터셋 로딩
         * @param rootDir 데이터셋 루트 디렉토리 (하위에 라벨별 폴더)
         * @param target  리사이즈 목표 크기 ({0,0}이면 원본 유지)
         * @param flag    OpenCV imread 플래그
         */
        void LoadImageDataset(const std::string& rootDir,
                              MatSize target = {0, 0},
                              int flag = cv::IMREAD_GRAYSCALE);

        // 접근자
        size_t GetDataSize() const;
        Tensor GetData(size_t index) const;
        int GetLabel(size_t index) const;

        // 배치 반환
        Tensor GetBatch(size_t startIdx, size_t batchSize) const;
        std::vector<int> GetLabelBatch(size_t startIdx, size_t batchSize) const;

        // 셔플 (외부 RNG 주입으로 재현성 확보)
        void Shuffle(std::mt19937& rng);

    private:
        /// 단일 이미지 → Tensor 변환 (정규화 포함: [0,255] → [0.0, 1.0])
        Tensor ReadImage(const std::string& path, MatSize target, int flag) const;

        LoadStrategy m_strategy;
        std::vector<Tensor> m_data;
        std::vector<std::string> m_paths;
        std::vector<int> m_labels;
        MatSize m_targetSize;
        int m_readFlag;

        static inline const std::unordered_set<std::string> IMAGE_EXTENSIONS =
            { ".jpg", ".png", ".jpeg", ".bmp", ".JPG", ".PNG" };
    };
}