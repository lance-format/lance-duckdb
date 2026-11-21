#include "duckdb.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_ffi.hpp"
#include "lance_resolver.hpp"
#include "lance_table_entry.hpp"

#include <algorithm>

namespace duckdb {

namespace {

// Split on tabs and keep empty fields: the tags column is empty for untagged
// versions.
vector<vector<string>> ParseVersionRows(const char *ptr, idx_t expected_cols,
                                        const char *what) {
  if (!ptr) {
    throw IOException(string("Failed to read Lance ") + what +
                      LanceFormatErrorSuffix());
  }
  string joined = ptr;
  lance_free_string(ptr);

  vector<vector<string>> rows;
  for (auto &line : StringUtil::Split(joined, '\n')) {
    if (line.empty()) {
      continue;
    }
    vector<string> fields;
    idx_t start = 0;
    while (true) {
      auto tab = line.find('\t', start);
      if (tab == string::npos) {
        fields.push_back(line.substr(start));
        break;
      }
      fields.push_back(line.substr(start, tab - start));
      start = tab + 1;
    }
    if (fields.size() != expected_cols) {
      throw IOException(string("Malformed Lance ") + what + " row");
    }
    rows.push_back(std::move(fields));
  }
  return rows;
}

Value MicrosToTimestampTz(const string &micros) {
  if (micros.empty()) {
    return Value(LogicalType::TIMESTAMP_TZ);
  }
  return Value::TIMESTAMPTZ(timestamp_tz_t(std::stoll(micros)));
}

enum VersionColumn : column_t { VERSION_ID, VERSION_TIMESTAMP, VERSION_TAGS };

Value VersionColumnValue(column_t column, const vector<string> &fields) {
  switch (column) {
  case VERSION_ID:
    return Value::BIGINT(std::stoll(fields[VERSION_ID]));
  case VERSION_TIMESTAMP:
    return MicrosToTimestampTz(fields[VERSION_TIMESTAMP]);
  case VERSION_TAGS: {
    vector<Value> tags;
    if (!fields[VERSION_TAGS].empty()) {
      for (auto &tag : StringUtil::Split(fields[VERSION_TAGS], ',')) {
        tags.emplace_back(tag);
      }
    }
    return Value::LIST(LogicalType::VARCHAR, std::move(tags));
  }
  default:
    throw InternalException("Unexpected lance_versions column ID");
  }
}

struct LanceVersionsBindData final : public FunctionData {
  LanceVersionsBindData(LanceDatasetReference dataset_ref_p, bool tags_p)
      : dataset_ref(std::move(dataset_ref_p)), tags(tags_p) {}

  LanceDatasetReference dataset_ref;
  bool tags;

  unique_ptr<FunctionData> Copy() const override {
    return make_uniq<LanceVersionsBindData>(dataset_ref, tags);
  }

  bool Equals(const FunctionData &other_p) const override {
    auto &other = other_p.Cast<LanceVersionsBindData>();
    return dataset_ref == other.dataset_ref && tags == other.tags;
  }
};

struct LanceVersionsGlobalState final : public GlobalTableFunctionState {
  vector<vector<Value>> rows;
  idx_t offset = 0;
};

unique_ptr<FunctionData> LanceVersionsBind(ClientContext &context,
                                           TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types,
                                           vector<Identifier> &names, bool tags,
                                           const char *fn_name) {
  auto dataset_ref =
      ResolveLanceDataset(context, input.inputs[0],
                          LanceResolvePolicy::FALLBACK_TO_PATH, fn_name)
          .Reference();
  if (tags) {
    return_types = {LogicalType::VARCHAR, LogicalType::BIGINT,
                    LogicalType::VARCHAR, LogicalType::TIMESTAMP_TZ,
                    LogicalType::TIMESTAMP_TZ};
    names = {"tag", "version", "branch", "created_at", "updated_at"};
  } else {
    return_types = {LogicalType::BIGINT, LogicalType::TIMESTAMP_TZ,
                    LogicalType::LIST(LogicalType::VARCHAR)};
    names = {"version", "timestamp", "tags"};
  }
  return make_uniq<LanceVersionsBindData>(std::move(dataset_ref), tags);
}

unique_ptr<FunctionData> LanceListVersionsBind(ClientContext &context,
                                               TableFunctionBindInput &input,
                                               vector<LogicalType> &types,
                                               vector<Identifier> &names) {
  return LanceVersionsBind(context, input, types, names, false,
                           "lance_versions");
}

unique_ptr<FunctionData> LanceListTagsBind(ClientContext &context,
                                           TableFunctionBindInput &input,
                                           vector<LogicalType> &types,
                                           vector<Identifier> &names) {
  return LanceVersionsBind(context, input, types, names, true, "lance_tags");
}

unique_ptr<GlobalTableFunctionState>
LanceVersionsInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
  auto &bind_data = input.bind_data->Cast<LanceVersionsBindData>();
  string display_uri;
  auto target = bind_data.dataset_ref.Resolve(context);
  void *dataset = target.Open(context, display_uri);
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + display_uri +
                      LanceFormatErrorSuffix());
  }

  auto state = make_uniq<LanceVersionsGlobalState>();
  try {
    LanceInterruptScope interrupt_scope(context);
    if (bind_data.tags) {
      for (auto &f :
           ParseVersionRows(lance_dataset_list_tags(dataset), 5, "tags")) {
        state->rows.push_back(
            {Value(f[0]), Value::BIGINT(std::stoll(f[1])),
             f[2].empty() ? Value(LogicalType::VARCHAR) : Value(f[2]),
             MicrosToTimestampTz(f[3]), MicrosToTimestampTz(f[4])});
      }
    } else {
      const auto &columns = input.column_ids;
      auto requested = [&](VersionColumn column) {
        return std::find(columns.begin(), columns.end(), column) !=
               columns.end();
      };
      auto rows = ParseVersionRows(
          lance_dataset_list_versions(dataset, requested(VERSION_TIMESTAMP),
                                      requested(VERSION_TAGS)),
          3, "versions");
      for (auto &fields : rows) {
        vector<Value> projected;
        for (auto column : columns) {
          projected.push_back(VersionColumnValue(column, fields));
        }
        state->rows.push_back(std::move(projected));
      }
    }
  } catch (...) {
    lance_close_dataset(dataset);
    throw;
  }
  lance_close_dataset(dataset);
  return std::move(state);
}

void LanceVersionsFunc(ClientContext &, TableFunctionInput &data,
                       DataChunk &output) {
  auto &state = data.global_state->Cast<LanceVersionsGlobalState>();
  idx_t count = 0;
  while (state.offset < state.rows.size() && count < STANDARD_VECTOR_SIZE) {
    auto &row = state.rows[state.offset++];
    for (idx_t col = 0; col < row.size(); col++) {
      output.data[col].Append(row[col]);
    }
    count++;
  }
}

} // namespace

void RegisterLanceVersions(ExtensionLoader &loader) {
  TableFunction versions("lance_versions", {LogicalType::VARCHAR},
                         LanceVersionsFunc, LanceListVersionsBind,
                         LanceVersionsInitGlobal);
  versions.projection_pushdown = true;
  loader.RegisterFunction(versions);
  loader.RegisterFunction(TableFunction("lance_tags", {LogicalType::VARCHAR},
                                        LanceVersionsFunc, LanceListTagsBind,
                                        LanceVersionsInitGlobal));
}

} // namespace duckdb
