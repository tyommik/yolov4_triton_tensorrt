#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

//#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include "math_constants.h"
#include "NvInfer.h"

#define MAX_ANCHORS 6

#define CHECK(status)                                           \
    do {                                                        \
        auto ret = status;                                      \
        if (ret != 0) {                                         \
            std::cerr << "Cuda failure in file '" << __FILE__   \
                      << "' line " << __LINE__                  \
                      << ": " << ret << std::endl;              \
            abort();                                            \
        }                                                       \
    } while (0)

#define ASSERT(assertion)                                        \
    do {                                                        \
        if (!(assertion)) {                                     \
            std::cerr << "#assertion " << __FILE__ << ","       \
                      << __LINE__ << std::endl;                 \
            abort();                                            \
        }                                                       \
    } while(0)

namespace Yolo
{
    static constexpr float IGNORE_THRESH = 0.01f;

    struct alignas(float) Detection {
        float bbox[4];  // x, y, w, h
        float det_confidence;
        float class_id;
        float class_confidence;
    };
}

namespace nvinfer1
{
    class YoloPluginDynamic: public IPluginV2DynamicExt
    {
        public:
            YoloPluginDynamic(const std::string name, int yolo_width, int yolo_height, int num_anchors, float* anchors, int num_classes, int input_width, int input_height, float scale_x_y, int new_coords);

            YoloPluginDynamic(const std::string name, const void* data, size_t length);
            YoloPluginDynamic() = delete;

            // IPluginV2DynamicExt Methods

            IPluginV2DynamicExt* clone() const override;

            DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) override;

            bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;

            void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) override {};

            size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const override { return 0; }

            int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

            // IPluginV2Ext Methods

            DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

            // IPluginV2 Methods

            const char* getPluginType() const override;
            const char* getPluginVersion() const override;
            int getNbOutputs() const override { return 1; }
            int initialize() override;
            void terminate() override;
            size_t getSerializationSize() const override;
            void serialize(void* buffer) const override;
            void destroy() override;
            void setPluginNamespace(const char* pluginNamespace) override;
            const char* getPluginNamespace() const override;

        private:
            void forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int batchSize = 1);

            const std::string mLayerName;
            std::string mPluginNamespace;

            int mThreadCount = 64;
            int mYoloWidth, mYoloHeight, mNumAnchors;
            float mAnchorsHost[MAX_ANCHORS * 2];
            float *mAnchors;  // allocated on GPU
            int mNumClasses;
            int mInputWidth, mInputHeight;
            float mScaleXY;
            int mNewCoords = 0;

        protected:
            // To prevent compiler warnings.
            using IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
            using IPluginV2DynamicExt::configurePlugin;
            using IPluginV2DynamicExt::enqueue;
            using IPluginV2DynamicExt::getOutputDimensions;
            using IPluginV2DynamicExt::getWorkspaceSize;
            using IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
            using IPluginV2DynamicExt::supportsFormat;
    };

    class YoloPluginDynamicCreator : public IPluginCreator
    {
        public:
            YoloPluginDynamicCreator();

            const char* getPluginName() const override;

            const char* getPluginVersion() const override;

            const PluginFieldCollection* getFieldNames() override;

            IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

            IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

            void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

            const char* getPluginNamespace() const override { return mNamespace.c_str(); }

        private:
            static PluginFieldCollection mFC;
            static std::vector<nvinfer1::PluginField> mPluginAttributes;
            std::string mNamespace;
    };
};

#endif
