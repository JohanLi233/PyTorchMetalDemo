#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "metal-cpp/SingleHeader/Metal.hpp"
#include "CustomFill.metal"

#include <torch/extension.h>
#include <iostream>

static inline const MTL::Buffer *getMTLBufferStorage(const torch::Tensor &tensor)
{
  return reinterpret_cast<const MTL::Buffer *>(tensor.storage().data()); // Cast to MTL::Buffer*
}

torch::Tensor custom_fill(torch::Tensor input, float fill_val)
{
  // 确保输入张量在MPS设备上
  TORCH_CHECK(input.device().is_mps(), "输入必须是MPS张量");
  TORCH_CHECK(input.dtype() == torch::kFloat, "输入张量必须是 float 类型");

  // 创建 Metal 设备
  MTL::Device *device = MTL::CreateSystemDefaultDevice();
  if (!device)
  {
    throw std::runtime_error("无法创建 MTL 设备");
  }

  // 获取张量的形状和大小信息
  auto shape = input.sizes();
  int64_t numElements = input.numel();

  // 创建 Metal 缓冲区
  MTL::Buffer *inputBuffer = device->newBuffer(numElements * sizeof(float), MTL::ResourceStorageModeShared);
  MTL::Buffer *fillValBuffer = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
  MTL::Buffer *sizeBuffer = device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);
  if (!inputBuffer || !fillValBuffer || !sizeBuffer)
  {
    throw std::runtime_error("无法创建缓冲区");
  }

  // 将输入数据复制到 Metal 缓冲区
  memcpy(inputBuffer->contents(), input.data_ptr(), numElements * sizeof(float));

  // 将填充值复制到缓冲区
  float *fillValPtr = static_cast<float *>(fillValBuffer->contents());
  *fillValPtr = fill_val;

  // 将元素数量复制到大小缓冲区
  uint *sizePtr = static_cast<uint *>(sizeBuffer->contents());
  *sizePtr = static_cast<uint>(numElements);

  // 加载 Metal 内核
  NS::Error *error = nullptr;
  MTL::Library *library = device->newLibrary(NS::String::string(CUSTOM_KERNEL, NS::UTF8StringEncoding), nullptr, &error);
  if (!library)
  {
    throw std::runtime_error(std::string("无法创建内核库，错误: ") + error->localizedDescription()->utf8String());
  }

  MTL::Function *function = library->newFunction(NS::String::string("custom_fill", NS::UTF8StringEncoding));
  MTL::ComputePipelineState *pipelineState = device->newComputePipelineState(function, &error);
  if (!pipelineState)
  {
    throw std::runtime_error(std::string("无法创建计算管线状态，错误: ") + error->localizedDescription()->utf8String());
  }

  // 创建命令队列和命令缓冲区
  MTL::CommandQueue *commandQueue = device->newCommandQueue();
  MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

  encoder->setComputePipelineState(pipelineState);
  encoder->setBuffer(inputBuffer, 0, 0);   // data
  encoder->setBuffer(fillValBuffer, 0, 1); // fill_val
  encoder->setBuffer(sizeBuffer, 0, 2);    // data_size

  // 设置线程组和线程数
  const uint threadsPerThreadgroup = 1024;
  const uint threadgroups = (numElements + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

  encoder->dispatchThreadgroups(MTL::Size(threadgroups, 1, 1), MTL::Size(threadsPerThreadgroup, 1, 1));
  encoder->endEncoding();

  // 提交命令缓冲区并等待完成
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  // 将结果复制回 mps
  torch::Tensor outputTensor = torch::from_blob(inputBuffer->contents(), shape, torch::kFloat).clone().to(input.device());

  // 清理资源
  library->release();
  function->release();
  pipelineState->release();
  commandQueue->release();
  inputBuffer->release();
  fillValBuffer->release();
  sizeBuffer->release();
  commandBuffer->release();
  encoder->release();
  device->release();

  return outputTensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("custom_fill", &custom_fill, "Custom fill function for MPS");
}
