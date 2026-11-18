#include "lance_versions.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_ffi.hpp"
#include "lance_insert.hpp"
#include "lance_resolver.hpp"
#include "lance_table_entry.hpp"

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
                    LogicalType::TIMESTAMP_TZ, LogicalType::TIMESTAMP_TZ};
    names = {"tag", "version", "created_at", "updated_at"};
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
           ParseVersionRows(lance_dataset_list_tags(dataset), 4, "tags")) {
        state->rows.push_back({Value(f[0]), Value::BIGINT(std::stoll(f[1])),
                               MicrosToTimestampTz(f[2]),
                               MicrosToTimestampTz(f[3])});
      }
    } else {
      for (auto &f : ParseVersionRows(lance_dataset_list_versions(dataset), 3,
                                      "versions")) {
        vector<Value> tag_values;
        if (!f[2].empty()) {
          for (auto &tag : StringUtil::Split(f[2], ',')) {
            tag_values.emplace_back(tag);
          }
        }
        state->rows.push_back(
            {Value::BIGINT(std::stoll(f[0])), MicrosToTimestampTz(f[1]),
             Value::LIST(LogicalType::VARCHAR, std::move(tag_values))});
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

enum class LanceVersionAdminOp : uint8_t {
  CREATE_TAG = 1,
  DELETE_TAG = 2,
  RESTORE = 3,
};

struct LanceVersionAdminBindData final : public FunctionData {
  LanceVersionAdminBindData(LanceDatasetReference dataset_ref_p,
                            LanceVersionAdminOp op_p, string tag_p,
                            bool has_version_p, int64_t version_p)
      : dataset_ref(std::move(dataset_ref_p)), op(op_p), tag(std::move(tag_p)),
        has_version(has_version_p), version(version_p) {}

  LanceDatasetReference dataset_ref;
  LanceVersionAdminOp op;
  string tag;
  bool has_version;
  int64_t version;

  unique_ptr<FunctionData> Copy() const override {
    return make_uniq<LanceVersionAdminBindData>(dataset_ref, op, tag,
                                                has_version, version);
  }

  bool Equals(const FunctionData &other_p) const override {
    auto &other = other_p.Cast<LanceVersionAdminBindData>();
    return dataset_ref == other.dataset_ref && op == other.op &&
           tag == other.tag && has_version == other.has_version &&
           version == other.version;
  }
};

struct LanceVersionAdminGlobalState final : public GlobalTableFunctionState {
  bool finished = false;
};

unique_ptr<GlobalTableFunctionState>
LanceVersionAdminInitGlobal(ClientContext &, TableFunctionInitInput &) {
  return make_uniq<LanceVersionAdminGlobalState>();
}

unique_ptr<FunctionData>
LanceVersionAdminBind(ClientContext &context, TableFunctionBindInput &input,
                      vector<LogicalType> &return_types,
                      vector<Identifier> &names, LanceVersionAdminOp op,
                      const char *fn_name) {
  auto dataset_ref =
      ResolveLanceDataset(context, input.inputs[0],
                          LanceResolvePolicy::FALLBACK_TO_PATH, fn_name)
          .Reference();

  string tag;
  bool has_version = false;
  int64_t version = 0;
  // Parameter 1 is always the tag; parameter 2 is the version when present.
  if (!input.inputs[1].IsNull()) {
    tag = input.inputs[1].GetValue<string>();
    if (tag.empty()) {
      throw BinderException("tag name must be non-empty");
    }
  }
  if (input.inputs.size() > 2 && !input.inputs[2].IsNull()) {
    has_version = true;
    version = input.inputs[2].GetValue<int64_t>();
    if (version < 1) {
      throw BinderException("version must be a positive integer");
    }
  }
  if (op == LanceVersionAdminOp::RESTORE && has_version == !tag.empty()) {
    throw BinderException("RESTORE requires exactly one of a version or a tag");
  }
  if (op != LanceVersionAdminOp::RESTORE && tag.empty()) {
    throw BinderException("tag name must be non-empty");
  }

  return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR,
                  LogicalType::VARCHAR, LogicalType::BIGINT};
  names = {"Operation", "Target", "Tag", "Version"};
  return make_uniq<LanceVersionAdminBindData>(
      std::move(dataset_ref), op, std::move(tag), has_version, version);
}

unique_ptr<FunctionData> LanceCreateTagBind(ClientContext &context,
                                            TableFunctionBindInput &input,
                                            vector<LogicalType> &types,
                                            vector<Identifier> &names) {
  return LanceVersionAdminBind(context, input, types, names,
                               LanceVersionAdminOp::CREATE_TAG,
                               "__lance_create_tag");
}

unique_ptr<FunctionData> LanceDeleteTagBind(ClientContext &context,
                                            TableFunctionBindInput &input,
                                            vector<LogicalType> &types,
                                            vector<Identifier> &names) {
  return LanceVersionAdminBind(context, input, types, names,
                               LanceVersionAdminOp::DELETE_TAG,
                               "__lance_delete_tag");
}

unique_ptr<FunctionData> LanceRestoreBind(ClientContext &context,
                                          TableFunctionBindInput &input,
                                          vector<LogicalType> &types,
                                          vector<Identifier> &names) {
  return LanceVersionAdminBind(context, input, types, names,
                               LanceVersionAdminOp::RESTORE,
                               "__lance_restore_version");
}

void LanceVersionAdminFunc(ClientContext &context, TableFunctionInput &data,
                           DataChunk &output) {
  auto &gstate = data.global_state->Cast<LanceVersionAdminGlobalState>();
  if (gstate.finished) {
    return;
  }
  gstate.finished = true;

  auto &bind_data = data.bind_data->Cast<LanceVersionAdminBindData>();
  if (bind_data.op == LanceVersionAdminOp::RESTORE &&
      LanceHasUncommittedWrites(context)) {
    throw NotImplementedException(
        "Cannot restore a Lance table while the transaction has uncommitted "
        "Lance writes");
  }

  string display_uri;
  auto target = bind_data.dataset_ref.Resolve(context);
  void *dataset = target.Open(context, display_uri);
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + display_uri +
                      LanceFormatErrorSuffix());
  }

  const char *op_name = "";
  int64_t result_version = 0;
  bool has_result_version = false;
  int32_t rc = -1;
  {
    LanceInterruptScope interrupt_scope(context);
    switch (bind_data.op) {
    case LanceVersionAdminOp::CREATE_TAG: {
      op_name = "create_tag";
      uint64_t version = 0;
      rc = 0;
      if (bind_data.has_version) {
        version = static_cast<uint64_t>(bind_data.version);
      } else {
        // Tag the latest version by default.
        rc = lance_dataset_version(dataset, &version);
      }
      if (rc == 0) {
        rc = lance_dataset_create_tag(dataset, bind_data.tag.c_str(), version);
        result_version = static_cast<int64_t>(version);
        has_result_version = true;
      }
      break;
    }
    case LanceVersionAdminOp::DELETE_TAG: {
      op_name = "drop_tag";
      uint64_t version = 0;
      if (lance_dataset_tag_version(dataset, bind_data.tag.c_str(), &version) ==
          0) {
        result_version = static_cast<int64_t>(version);
        has_result_version = true;
      }
      rc = lance_dataset_delete_tag(dataset, bind_data.tag.c_str());
      break;
    }
    case LanceVersionAdminOp::RESTORE: {
      op_name = "restore";
      uint64_t version = 0;
      rc = 0;
      if (bind_data.has_version) {
        version = static_cast<uint64_t>(bind_data.version);
      } else {
        rc =
            lance_dataset_tag_version(dataset, bind_data.tag.c_str(), &version);
      }
      if (rc == 0) {
        uint64_t new_version = 0;
        rc = lance_dataset_restore_version(dataset, version, &new_version);
        result_version = static_cast<int64_t>(new_version);
        has_result_version = true;
      }
      break;
    }
    default:
      throw InternalException("unknown version admin op");
    }
  }
  if (rc != 0) {
    // Capture the error before closing the handle can clobber it.
    auto suffix = LanceFormatErrorSuffix();
    lance_close_dataset(dataset);
    context.BlockingInterruptCheck();
    throw IOException(string("Failed to ") + op_name +
                      " on Lance dataset: " + display_uri + suffix);
  }
  lance_close_dataset(dataset);

  // Restore moves the latest version; later reads must not use a stale cache.
  if (bind_data.op == LanceVersionAdminOp::RESTORE) {
    if (target.table) {
      LanceInvalidateDatasetCacheForTable(context, *target.table);
    } else {
      LanceInvalidateDatasetCacheForPath(context, target.path);
    }
  }

  output.data[0].Append(Value(op_name));
  output.data[1].Append(Value(display_uri));
  output.data[2].Append(bind_data.tag.empty() ? Value(LogicalType::VARCHAR)
                                              : Value(bind_data.tag));
  output.data[3].Append(has_result_version ? Value::BIGINT(result_version)
                                           : Value(LogicalType::BIGINT));
}

} // namespace

TableFunction LanceInternalCreateTagTableFunction() {
  return TableFunction(
      "__lance_create_tag",
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT},
      LanceVersionAdminFunc, LanceCreateTagBind, LanceVersionAdminInitGlobal);
}

TableFunction LanceInternalDeleteTagTableFunction() {
  return TableFunction(
      "__lance_delete_tag", {LogicalType::VARCHAR, LogicalType::VARCHAR},
      LanceVersionAdminFunc, LanceDeleteTagBind, LanceVersionAdminInitGlobal);
}

TableFunction LanceInternalRestoreTableFunction() {
  return TableFunction(
      "__lance_restore_version",
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT},
      LanceVersionAdminFunc, LanceRestoreBind, LanceVersionAdminInitGlobal);
}

void RegisterLanceVersions(ExtensionLoader &loader) {
  loader.RegisterFunction(
      TableFunction("lance_versions", {LogicalType::VARCHAR}, LanceVersionsFunc,
                    LanceListVersionsBind, LanceVersionsInitGlobal));
  loader.RegisterFunction(TableFunction("lance_tags", {LogicalType::VARCHAR},
                                        LanceVersionsFunc, LanceListTagsBind,
                                        LanceVersionsInitGlobal));
  loader.RegisterFunction(LanceInternalCreateTagTableFunction());
  loader.RegisterFunction(LanceInternalDeleteTagTableFunction());
  loader.RegisterFunction(LanceInternalRestoreTableFunction());
}

} // namespace duckdb