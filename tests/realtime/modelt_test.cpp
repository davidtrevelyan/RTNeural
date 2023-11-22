#include <gmock/gmock.h>

#include "../functional/load_csv.hpp"
#include "../functional/test_configs.hpp"
#include "assertions.hpp"

#include <RTNeural.h>

namespace
{
using TestType = double;
using namespace RTNeural;

template <typename T, typename StaticModelType>
void checkRealTimeSafety(const TestConfig& test)
{
    std::ifstream jsonStream(std::string { RTNEURAL_ROOT_DIR } + test.model_file, std::ifstream::binary);
    StaticModelType static_model;
    static_model.parseJson(jsonStream, true);
    static_model.reset();

    jsonStream.seekg(0);
    auto dynamic_model = RTNeural::json_parser::parseJson<T>(jsonStream, true);
    dynamic_model->reset();

    T input[] = { 0.5f };
    EXPECT_REAL_TIME_SAFE(static_model.forward(input));
    EXPECT_REAL_TIME_SAFE(dynamic_model->forward(input));
}
}

TEST(TestTemplatedModels, doesNotCauseRADSanErrorWithDense)
{
    using ModelType = ModelT<TestType, 1, 1,
        DenseT<TestType, 1, 8>,
        TanhActivationT<TestType, 8>,
        DenseT<TestType, 8, 8>,
        ReLuActivationT<TestType, 8>,
        DenseT<TestType, 8, 8>,
        ELuActivationT<TestType, 8>,
        DenseT<TestType, 8, 8>,
        SoftmaxActivationT<TestType, 8>,
        DenseT<TestType, 8, 1>>;

    checkRealTimeSafety<TestType, ModelType>(tests.at("dense"));
}

TEST(TestTemplatedModels, doesNotCauseRADSanErrorWithConv1D)
{
    using ModelType = ModelT<TestType, 1, 1,
        DenseT<TestType, 1, 8>,
        TanhActivationT<TestType, 8>,
        Conv1DT<TestType, 8, 4, 3, 1, true>,
        TanhActivationT<TestType, 4>,
        BatchNorm1DT<TestType, 4>,
        PReLUActivationT<TestType, 4>,
        Conv1DT<TestType, 4, 4, 1, 1>,
        TanhActivationT<TestType, 4>,
        Conv1DT<TestType, 4, 4, 3, 2>,
        TanhActivationT<TestType, 4>,
        BatchNorm1DT<TestType, 4, false>,
        PReLUActivationT<TestType, 4>,
        DenseT<TestType, 4, 1>,
        SigmoidActivationT<TestType, 1>>;

    checkRealTimeSafety<TestType, ModelType>(tests.at("conv1d"));
}

TEST(TestTemplatedModels, doesNotCauseRADSanErrorWithGRU)
{
    using ModelType = ModelT<TestType, 1, 1,
        DenseT<TestType, 1, 8>,
        TanhActivationT<TestType, 8>,
        GRULayerT<TestType, 8, 8>,
        DenseT<TestType, 8, 8>,
        SigmoidActivationT<TestType, 8>,
        DenseT<TestType, 8, 1>>;

    checkRealTimeSafety<TestType, ModelType>(tests.at("gru"));
}

TEST(TestTemplatedModels, doesNotCauseRADSanErrorWithGRU1D)
{
    using ModelType = ModelT<TestType, 1, 1,
        GRULayerT<TestType, 1, 8>,
        DenseT<TestType, 8, 8>,
        SigmoidActivationT<TestType, 8>,
        DenseT<TestType, 8, 1>>;

    checkRealTimeSafety<TestType, ModelType>(tests.at("gru_1d"));
}

TEST(TestTemplatedModels, doesNotCauseRADSanErrorWithLSTM)
{
    using ModelType = ModelT<TestType, 1, 1,
        DenseT<TestType, 1, 8>,
        TanhActivationT<TestType, 8>,
        LSTMLayerT<TestType, 8, 8>,
        DenseT<TestType, 8, 1>>;

    checkRealTimeSafety<TestType, ModelType>(tests.at("lstm"));
}

TEST(TestTemplatedModels, doesNotCauseRADSanErrorWithLSTM1D)
{
    using ModelType = ModelT<TestType, 1, 1,
        LSTMLayerT<TestType, 1, 8>,
        DenseT<TestType, 8, 1>>;

    checkRealTimeSafety<TestType, ModelType>(tests.at("lstm_1d"));
}
