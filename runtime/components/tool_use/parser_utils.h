// Copyright 2025 The Google AI Edge Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_PARSER_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_PARSER_UTILS_H_

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json

namespace litert::lm {

// A struct to hold the text and tool call strings.
struct TextAndToolCallStrings {
  absl::string_view text;
  absl::string_view tool_calls;
};

// The syntax type of the tool calls.
enum class SyntaxType {
  kUnknown = 0,
  kPython = 1,
  kJson = 2,
};

// Maps from string to SyntaxType.
SyntaxType GetSyntaxType(absl::string_view syntax_type);

// Extracts text and code blocks from a string. A code block is delimited
// by `code_fence_start` and `code_fence_end`.
//
// Args:
//   `response_str`: The raw string response from the model.
//   `code_fence_start`: The string marking the beginning of the code block.
//   `code_fence_end`: The string marking the end of the code block.
//   `escape_fence_strings`: If true, regex special characters within the
//     fence strings will be escaped using RE2::QuoteMeta. Set to false if the
//     fence strings already contain valid regex patterns.
// Returns:
//   A TextAndToolCallStrings struct. `text` contains the portion of
//   `response_str` *before* the `code_fence_start`. `tool_calls` contains
//   the portion of `response_str` *between* the start and end fences.
//   If the pattern is not found, behavior depends:
//     - If `code_fence_start` is not found at all, the entire `response_str`
//       is returned in `text`, and `tool_calls` is empty.
//     - If `code_fence_start` is found but `code_fence_end` is not (or the
//       regex match fails), the text before the start fence is returned in
//       `text`, and the text *after* the start fence is returned in
//       `tool_calls`.
TextAndToolCallStrings ExtractTextAndToolCallStrings(
    absl::string_view response_str, absl::string_view code_fence_start,
    absl::string_view code_fence_end, bool escape_fence_strings = true);

// Parses a string into text and tool calls.
//
// Args:
//   `response_str`: The raw string response from the model.
//   `code_fence_start`: The string marking the beginning of the code block.
//   `code_fence_end`: The string marking the end of the code block.
//   `response_role`: The role to assign to the response content (e.g.,
//      "model").
//   `syntax_type`: The syntax type of the tool calls.
//   `escape_fence_strings`: If true, regex special characters
//      within the fence strings will be escaped.
//   `tool_code_regex`: A regex with a capture group used to filter each line
//      of the tool call string.
//
// Returns:
//   A JSON array of content parts. Each content part has type "text" or
//   "tool_call".
absl::StatusOr<nlohmann::ordered_json> ParseTextAndToolCalls(
    absl::string_view response_str, absl::string_view code_fence_start,
    absl::string_view code_fence_end, SyntaxType syntax_type,
    bool escape_fence_strings = true, absl::string_view tool_code_regex = "");

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_PARSER_UTILS_H_
