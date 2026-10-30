#pragma once

#include "duckdb/parser/parser_extension.hpp"

namespace duckdb {

// The PEG parser supplies the unparsed tail, which can contain later SQL
// statements. Pass exactly one statement to the Lance grammar and claim only
// its tokens. Token text retains quotes and escapes; dots remain adjacent so
// the existing qualified-name and field-path grammar sees a single path.
inline ParserExtensionParseResult
ParseLanceStatement(ParserExtensionInfo *info,
                    const vector<SimpleToken> &tokens,
                    ParserExtensionParseResult (*parse)(ParserExtensionInfo *,
                                                        const string &)) {
  string statement;
  idx_t consumed = 0;
  string previous;
  for (const auto &token : tokens) {
    if (token.text == ";") {
      break;
    }
    if (!statement.empty() && token.text != "." && previous != "." &&
        !(token.type == TokenType::NUMBER_LITERAL &&
          (previous == "-" || previous == "+"))) {
      statement += " ";
    }
    statement += token.text;
    previous = token.text;
    consumed++;
  }
  auto result = parse(info, statement);
  if (result.type == ParserExtensionResultType::PARSE_SUCCESSFUL) {
    result.consumed_tokens = NumericCast<int64_t>(consumed);
  } else if (result.type ==
             ParserExtensionResultType::DISPLAY_EXTENSION_ERROR) {
    result.consumed_tokens = -1;
  }
  return result;
}

} // namespace duckdb
