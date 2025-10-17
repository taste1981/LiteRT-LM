// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/conversation/model_data_processor/qwen3_data_processor.h"

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/qwen3_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<ModelDataProcessor>> Qwen3DataProcessor::Create(
    Qwen3DataProcessorConfig config) {
  return absl::WrapUnique(new Qwen3DataProcessor(config));
}

absl::StatusOr<nlohmann::ordered_json>
Qwen3DataProcessor::MessageToTemplateInput(
    const nlohmann::ordered_json& message) const {
  if (message["content"].is_array()) {
    const auto& content = message["content"];
    if (content.size() == 1 && content[0].contains("text")) {
      auto result = nlohmann::ordered_json::object(
          {{"role", message["role"]}, {"content", content[0]["text"]}});
      return result;
    }
  }
  return message;
}

absl::StatusOr<std::vector<InputData>>
Qwen3DataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt,
    const nlohmann::ordered_json& messages,
    const Qwen3DataProcessorArguments& args) {
  std::vector<InputData> input_data;
  input_data.emplace_back(InputText(rendered_template_prompt));
  return input_data;
}

absl::StatusOr<Message> Qwen3DataProcessor::ToMessageImpl(
    const Responses& responses, const Qwen3DataProcessorArguments& args) {
  ASSIGN_OR_RETURN(absl::string_view response_text,
                   responses.GetResponseTextAt(0));
  nlohmann::ordered_json content;
  content = nlohmann::ordered_json::array(
      {{{"type", "text"}, {"text", std::string(response_text)}}});
  return nlohmann::ordered_json::object(
      {{"role", "assistant"}, {"content", content}});
}

}  // namespace litert::lm
