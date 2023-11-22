#include <gmock/gmock.h>

#include "load_csv.hpp"
#include <RTNeural.h>

namespace
{
template <typename T>
std::vector<std::vector<T>> loadFile2D(std::ifstream& stream)
{
    std::vector<std::vector<T>> vec;

    std::string line;
    if(stream.is_open())
    {
        while(std::getline(stream, line))
        {
            std::vector<T> lineVec;
            std::string num;
            for(auto ch : line)
            {
                if(ch == ',')
                {
                    lineVec.push_back(static_cast<T>(std::stod(num)));
                    num.clear();
                    continue;
                }

                num.push_back(ch);
            }

            lineVec.push_back(static_cast<T>(std::stod(num)));
            vec.push_back(lineVec);
        }

        stream.close();
    }

    return RTNeural::torch_helpers::detail::transpose(vec);
}

template <typename T>
void expectNear(T const& expected, T const& actual)
{
    EXPECT_THAT(
        static_cast<double>(expected),
        testing::DoubleNear(static_cast<double>(actual), 1e-6));
}

template <typename T>
void testTorchConv1DModel()
{
    const auto model_file = std::string { RTNEURAL_ROOT_DIR } + "models/conv1d_torch.json";
    std::ifstream jsonStream(model_file, std::ifstream::binary);
    nlohmann::json modelJson;
    jsonStream >> modelJson;

    RTNeural::ModelT<T, 1, 12, RTNeural::Conv1DT<T, 1, 12, 5, 1>> model;
    RTNeural::torch_helpers::loadConv1D<T>(modelJson, "", model.template get<0>());
    model.reset();

    std::ifstream modelInputsFile { std::string { RTNEURAL_ROOT_DIR } + "test_data/conv1d_torch_x_python.csv" };
    const auto inputs = load_csv::loadFile<T>(modelInputsFile);
    std::vector<std::array<T, 12>> outputs {};
    outputs.resize(inputs.size(), {});

    for(size_t i = 0; i < inputs.size(); ++i)
    {
        model.forward(&inputs[i]);
        std::copy(model.getOutputs(), model.getOutputs() + 12, outputs[i].begin());
    }

    std::ifstream modelOutputsFile { std::string { RTNEURAL_ROOT_DIR } + "test_data/conv1d_torch_y_python.csv" };
    const auto expected_y = loadFile2D<T>(modelOutputsFile);

    for(size_t n = 0; n < expected_y.size(); ++n)
    {
        for(size_t j = 0; j < outputs[n].size(); ++j)
        {
            expectNear(outputs[n + 4][j], expected_y[n][j]);
        }
    }
}
}

TEST(TestTorchConv1D, modelOutputMatchesPythonImplementationForFloats)
{
    testTorchConv1DModel<float>();
}

TEST(TestTorchConv1D, modelOutputMatchesPythonImplementationForDoubles)
{
    testTorchConv1DModel<double>();
}
