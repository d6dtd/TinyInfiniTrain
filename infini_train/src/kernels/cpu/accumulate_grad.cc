#include <cstddef>
#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================
    float beta1_pow = std::pow(beta1, t);
    float beta2_pow = std::pow(beta2, t);

    for (int64_t idx = 0; idx < grad->NumElements(); ++idx) {
        auto L = static_cast<const float *>(grad->DataPtr())[idx];
        auto &mm = static_cast<float *>(m->DataPtr())[idx];
        auto &vv = static_cast<float *>(v->DataPtr())[idx];
        mm = beta1 * mm + (1 - beta1) * L;
        vv = beta2 * vv + (1 - beta2) * L * L;
        float m_hat = mm / (1 - beta1_pow);
        float v_hat = vv / (1 - beta2_pow);
        static_cast<float *>(param->DataPtr())[idx] -= learning_rate * m_hat / std::sqrt(v_hat + eps);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
