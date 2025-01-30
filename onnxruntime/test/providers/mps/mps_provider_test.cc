// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/mps/mps_execution_provider.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

TEST(MPSExecutionProviderTest, MetadataTest) {
  auto provider = std::make_unique<MPSExecutionProvider>();
  EXPECT_TRUE(provider != nullptr);
  ASSERT_EQ(provider->GetOrtDeviceByMemType(OrtMemTypeDefault).Type(), OrtDevice::GPU);
}

}  // namespace test
}  // namespace onnxruntime
