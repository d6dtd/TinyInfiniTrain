#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */
    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "文件打开失败: " << path;

    auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    int magic = BytesToType<int>(header, 0); // ?
    int version = BytesToType<int>(header, 4);
    int num_toks = BytesToType<int>(header, 8);

    auto it = kTypeMap.find(magic);
    CHECK(it != kTypeMap.end()) << "不支持的文件版本: " << version;
    auto tok_type = it->second;
    auto tok_size = kTypeToSize.at(tok_type);
    auto dtype = kTypeToDataType.at(tok_type);
    auto token_bytes = ReadSeveralBytesFromIfstream(num_toks * tok_size, &ifs);
    // 计算样本数
    size_t num_samples = num_toks / sequence_length;
    std::vector<int64_t> dims = {static_cast<int64_t>(num_samples), static_cast<int64_t>(sequence_length)};
    // 构造 tensor

    TinyShakespeareFile file;
    file.dims = dims;
    file.type = tok_type;
    file.tensor = infini_train::Tensor(dims, DataType::kINT64); // 不是kINT64会报错
    for (int i = 0; i < num_samples * sequence_length; i++) {
        int64_t val = 0;
        if (tok_type == TinyShakespeareType::kUINT16) {
            val = static_cast<int64_t>(BytesToType<uint16_t>(token_bytes, i * tok_size));
        } else if (tok_type == TinyShakespeareType::kUINT32) {
            val = static_cast<int64_t>(BytesToType<uint32_t>(token_bytes, i * tok_size));
        }
        std::memcpy(static_cast<int64_t *>(file.tensor.DataPtr()) + i, &val, sizeof(int64_t));
    }
    return file;
}

} // namespace
TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)), sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)), num_samples_(text_file_.dims[0]) {
    // const变量，只能列表初始化
}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
