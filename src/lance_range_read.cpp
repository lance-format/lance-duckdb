#include "duckdb.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/storage/buffer_manager.hpp"

#include "lance_common.hpp"
#include "lance_ffi.hpp"

#include <atomic>
#include <cstring>
#include <vector>

namespace duckdb {

static constexpr const char *LANCE_BLOB_SELECTOR_ROW_ADDRESS = "row_address";
static constexpr const char *LANCE_BLOB_SELECTOR_ROW_ID = "row_id";
static constexpr uint8_t LANCE_FFI_BLOB_SELECTOR_ROW_ADDRESS = 0;
static constexpr uint8_t LANCE_FFI_BLOB_SELECTOR_ROW_ID = 1;

static constexpr const char *LANCE_BLOB_REQUEST_ID_COLUMN = "request_id";
static constexpr const char *LANCE_BLOB_ROW_COLUMN = "row_value";
static constexpr const char *LANCE_BLOB_OFFSET_COLUMN = "offset";
static constexpr const char *LANCE_BLOB_LENGTH_COLUMN = "length";
static constexpr const char *LANCE_BLOB_RESULT_REQUEST_INDEX = "request_index";
static constexpr const char *LANCE_BLOB_RESULT_REQUEST_ID = "request_id";
static constexpr const char *LANCE_BLOB_RESULT_ROW_ADDRESS = "row_address";
static constexpr const char *LANCE_BLOB_RESULT_DATA = "data";
static constexpr const char *LANCE_BLOB_RESULT_VERSION = "version";

// One input chunk is processed as one or more bounded Lance request windows.
// A window never materializes more total payload bytes than the per-window
// byte budget derived from DuckDB's memory_limit, so a single input chunk
// cannot force arbitrary simultaneous materialization.
struct LanceBlobRangeReadBindData : public TableFunctionData {
  string dataset_path;
  string column;
  uint8_t selector = LANCE_FFI_BLOB_SELECTOR_ROW_ADDRESS;
  idx_t request_id_idx = DConstants::INVALID_INDEX;
  idx_t row_idx = DConstants::INVALID_INDEX;
  idx_t offset_idx = DConstants::INVALID_INDEX;
  idx_t length_idx = DConstants::INVALID_INDEX;
  LogicalType request_id_type;
  void *dataset = nullptr;
  uint64_t dataset_version = 0;
  uint64_t range_bytes_budget = 0;

  ~LanceBlobRangeReadBindData() override {
    if (dataset) {
      lance_close_dataset(dataset);
      dataset = nullptr;
    }
  }

  // The bind data pins a dataset handle and the memory_limit-derived window
  // budget. Both must be refreshed per execution so prepared statements with
  // version = NULL observe external commits and pick up SET memory_limit.
  bool SupportStatementCache() const override { return false; }
};

struct LanceBlobRangeReadGlobalState : public GlobalTableFunctionState {
  // Independent pipelines (e.g. UNION ALL branches) can execute this operator
  // concurrently, so each chunk reserves a contiguous request range with an
  // atomic counter. Lance still parallelizes the actual I/O.
  std::atomic<uint64_t> next_request_index{0};

  idx_t MaxThreads() const override { return 1; }
};

struct LanceBlobRangeReadLocalState : public LocalTableFunctionState {
  // Start of the next request window inside the current input chunk.
  idx_t next_input_row = 0;
  // First global request index owned by the current chunk.
  uint64_t chunk_request_base = 0;
};

// Owns one FFI result array so exceptions while filling the output chunk
// cannot leak the materialized payloads.
class LanceBlobRangeResultGuard {
public:
  LanceBlobRangeResultGuard() = default;
  LanceBlobRangeResultGuard(const LanceBlobRangeResultGuard &) = delete;
  LanceBlobRangeResultGuard &
  operator=(const LanceBlobRangeResultGuard &) = delete;

  ~LanceBlobRangeResultGuard() { Reset(); }

  void Reset() {
    if (results) {
      lance_free_blob_range_results(results, results_len);
      results = nullptr;
      results_len = 0;
    }
  }

  LanceBlobRangeResult *results = nullptr;
  size_t results_len = 0;
};

static bool TryParseBlobRangeUInt64(const Value &value, uint64_t &out) {
  if (value.IsNull()) {
    return false;
  }
  auto parse_signed = [&out](int64_t v) {
    if (v < 0) {
      return false;
    }
    out = NumericCast<uint64_t>(v);
    return true;
  };
  switch (value.type().id()) {
  case LogicalTypeId::TINYINT:
    return parse_signed(value.GetValue<int8_t>());
  case LogicalTypeId::SMALLINT:
    return parse_signed(value.GetValue<int16_t>());
  case LogicalTypeId::INTEGER:
    return parse_signed(value.GetValue<int32_t>());
  case LogicalTypeId::BIGINT:
    return parse_signed(value.GetValue<int64_t>());
  case LogicalTypeId::UTINYINT:
    out = value.GetValue<uint8_t>();
    return true;
  case LogicalTypeId::USMALLINT:
    out = value.GetValue<uint16_t>();
    return true;
  case LogicalTypeId::UINTEGER:
    out = value.GetValue<uint32_t>();
    return true;
  case LogicalTypeId::UBIGINT:
    out = value.GetValue<uint64_t>();
    return true;
  default:
    return false;
  }
}

static idx_t FindRequestColumn(const TableFunctionBindInput &input,
                               const char *name) {
  for (idx_t i = 0; i < input.input_table_names.size(); i++) {
    if (StringUtil::CIEquals(input.input_table_names[i].GetIdentifierName(),
                             name)) {
      return i;
    }
  }
  return DConstants::INVALID_INDEX;
}

static unique_ptr<FunctionData>
LanceBlobRangeReadBind(ClientContext &context, TableFunctionBindInput &input,
                       vector<LogicalType> &return_types,
                       vector<Identifier> &names) {
  if (input.inputs[0].IsNull()) {
    throw BinderException(
        "lance_read_blob_ranges: dataset URI must not be NULL");
  }
  if (input.inputs[1].IsNull()) {
    throw BinderException("lance_read_blob_ranges: column must not be NULL");
  }
  if (input.inputs[3].IsNull()) {
    throw BinderException("lance_read_blob_ranges: selector must not be NULL");
  }

  auto result = make_uniq<LanceBlobRangeReadBindData>();
  result->dataset_path = input.inputs[0].GetValue<string>();
  result->column = input.inputs[1].GetValue<string>();
  if (result->dataset_path.empty()) {
    throw BinderException(
        "lance_read_blob_ranges: dataset URI must not be empty");
  }
  if (result->column.empty()) {
    throw BinderException("lance_read_blob_ranges: column must not be empty");
  }

  auto selector = input.inputs[3].GetValue<string>();
  if (selector == LANCE_BLOB_SELECTOR_ROW_ADDRESS) {
    result->selector = LANCE_FFI_BLOB_SELECTOR_ROW_ADDRESS;
  } else if (selector == LANCE_BLOB_SELECTOR_ROW_ID) {
    result->selector = LANCE_FFI_BLOB_SELECTOR_ROW_ID;
  } else {
    throw BinderException(
        "lance_read_blob_ranges: selector must be 'row_address' or 'row_id', "
        "got '%s'",
        selector);
  }

  optional_idx version;
  if (!input.inputs[2].IsNull()) {
    auto version_value = input.inputs[2].GetValue<int64_t>();
    if (version_value < 1) {
      throw BinderException(
          "lance_read_blob_ranges: version must be a positive integer or "
          "NULL for the latest version");
    }
    version = optional_idx(NumericCast<idx_t>(version_value));
  } else if (result->selector == LANCE_FFI_BLOB_SELECTOR_ROW_ADDRESS) {
    // Physical row addresses cannot be reinterpreted against an open-ended
    // snapshot; a persisted address must always be bound to the version it
    // was read from.
    throw BinderException(
        "lance_read_blob_ranges: selector 'row_address' requires an explicit "
        "positive version; a NULL version for the latest snapshot is only "
        "available through 'row_id' on datasets with stable row ids");
  }

  result->request_id_idx =
      FindRequestColumn(input, LANCE_BLOB_REQUEST_ID_COLUMN);
  result->row_idx = FindRequestColumn(input, LANCE_BLOB_ROW_COLUMN);
  result->offset_idx = FindRequestColumn(input, LANCE_BLOB_OFFSET_COLUMN);
  result->length_idx = FindRequestColumn(input, LANCE_BLOB_LENGTH_COLUMN);
  const std::pair<idx_t, const char *> required_columns[] = {
      {result->request_id_idx, LANCE_BLOB_REQUEST_ID_COLUMN},
      {result->row_idx, LANCE_BLOB_ROW_COLUMN},
      {result->offset_idx, LANCE_BLOB_OFFSET_COLUMN},
      {result->length_idx, LANCE_BLOB_LENGTH_COLUMN},
  };
  for (const auto &entry : required_columns) {
    if (entry.first == DConstants::INVALID_INDEX) {
      throw BinderException(
          "lance_read_blob_ranges: request relation is missing required "
          "column '%s'",
          entry.second);
    }
  }
  const std::pair<idx_t, const char *> integral_columns[] = {
      {result->row_idx, LANCE_BLOB_ROW_COLUMN},
      {result->offset_idx, LANCE_BLOB_OFFSET_COLUMN},
      {result->length_idx, LANCE_BLOB_LENGTH_COLUMN},
  };
  for (const auto &entry : integral_columns) {
    if (!input.input_table_types[entry.first].IsIntegral()) {
      throw BinderException(
          "lance_read_blob_ranges: request column '%s' must be an integer "
          "type, got %s",
          entry.second, input.input_table_types[entry.first].ToString());
    }
  }
  result->request_id_type = input.input_table_types[result->request_id_idx];

  result->dataset =
      LanceOpenDatasetAtVersion(context, result->dataset_path, version);
  if (!result->dataset) {
    throw IOException("Failed to open Lance dataset '" + result->dataset_path +
                      "' for blob range read" + LanceFormatErrorSuffix());
  }
  uint64_t actual_version = 0;
  if (lance_dataset_version(result->dataset, &actual_version) != 0) {
    auto error_message = LanceConsumeLastError();
    lance_close_dataset(result->dataset);
    result->dataset = nullptr;
    throw IOException("Failed to read Lance dataset version for '" +
                      result->dataset_path + "': " + error_message);
  }
  result->dataset_version = actual_version;

  if (result->selector == LANCE_FFI_BLOB_SELECTOR_ROW_ID &&
      !version.IsValid()) {
    // `_rowid` is only a stable identity when the manifest enables stable row
    // ids. Otherwise it is the physical address, which must stay bound to the
    // snapshot it was read from.
    uint8_t uses_stable_row_ids = 0;
    if (lance_dataset_uses_stable_row_ids(result->dataset,
                                          &uses_stable_row_ids) != 0) {
      auto error_message = LanceConsumeLastError();
      lance_close_dataset(result->dataset);
      result->dataset = nullptr;
      throw IOException("Failed to read Lance stable row-id flag for '" +
                        result->dataset_path + "': " + error_message);
    }
    if (!uses_stable_row_ids) {
      throw BinderException(
          "lance_read_blob_ranges: selector 'row_id' with a NULL version "
          "requires a dataset with stable row ids; this dataset exposes "
          "physical row addresses as '_rowid', so pass an explicit positive "
          "version (or use selector 'row_address')");
    }
  }

  auto &buffer_manager = BufferManager::GetBufferManager(context);
  auto memory_limit = buffer_manager.GetMaxMemory();
  result->range_bytes_budget = memory_limit / 8;
  if (result->range_bytes_budget == 0) {
    throw BinderException(
        "lance_read_blob_ranges: DuckDB memory_limit is too small to "
        "materialize a blob range window");
  }

  return_types = {LogicalType::UBIGINT, result->request_id_type,
                  LogicalType::UBIGINT, LogicalType::BLOB,
                  LogicalType::UBIGINT};
  names = {LANCE_BLOB_RESULT_REQUEST_INDEX, LANCE_BLOB_RESULT_REQUEST_ID,
           LANCE_BLOB_RESULT_ROW_ADDRESS, LANCE_BLOB_RESULT_DATA,
           LANCE_BLOB_RESULT_VERSION};
  return std::move(result);
}

static unique_ptr<GlobalTableFunctionState>
LanceBlobRangeReadInitGlobal(ClientContext &, TableFunctionInitInput &) {
  return make_uniq<LanceBlobRangeReadGlobalState>();
}

static unique_ptr<LocalTableFunctionState>
LanceBlobRangeReadInitLocal(ExecutionContext &, TableFunctionInitInput &,
                            GlobalTableFunctionState *) {
  return make_uniq<LanceBlobRangeReadLocalState>();
}

static void ValidateBlobRangeValue(const Value &value, const char *column,
                                   uint64_t global_request_index,
                                   uint64_t &out_value) {
  if (value.IsNull()) {
    throw InvalidInputException(
        "lance_read_blob_ranges: request %llu has NULL '%s'",
        global_request_index, column);
  }
  if (!TryParseBlobRangeUInt64(value, out_value)) {
    throw InvalidInputException(
        "lance_read_blob_ranges: request %llu has a non-integer or negative "
        "'%s'",
        global_request_index, column);
  }
}

static OperatorResultType LanceBlobRangeReadFunc(ExecutionContext &,
                                                 TableFunctionInput &data,
                                                 DataChunk &input,
                                                 DataChunk &output) {
  if (input.size() == 0) {
    return OperatorResultType::NEED_MORE_INPUT;
  }
  auto &bind_data = data.bind_data->Cast<LanceBlobRangeReadBindData>();
  auto &global_state = data.global_state->Cast<LanceBlobRangeReadGlobalState>();
  auto &local_state = data.local_state->Cast<LanceBlobRangeReadLocalState>();

  const idx_t input_rows = input.size();
  if (local_state.next_input_row == 0) {
    local_state.chunk_request_base =
        global_state.next_request_index.fetch_add(input_rows);
  }
  idx_t window_start = local_state.next_input_row;
  if (window_start >= input_rows) {
    local_state.next_input_row = 0;
    return OperatorResultType::NEED_MORE_INPUT;
  }
  const uint64_t chunk_request_base = local_state.chunk_request_base;

  // Grow the window until the byte budget or the end of the chunk is reached.
  // A single request larger than the budget is rejected up front instead of
  // being materialized beyond the window policy.
  uint64_t window_bytes = 0;
  idx_t window_end = window_start;
  while (window_end < input_rows) {
    uint64_t length = 0;
    auto length_value =
        input.GetValue(bind_data.length_idx, NumericCast<idx_t>(window_end));
    ValidateBlobRangeValue(length_value, LANCE_BLOB_LENGTH_COLUMN,
                           chunk_request_base + window_end, length);
    if (length > bind_data.range_bytes_budget) {
      throw InvalidInputException(
          "lance_read_blob_ranges: request %llu asks for %llu bytes which "
          "exceeds the per-window materialization budget of %llu bytes "
          "(derived from DuckDB memory_limit); request a smaller range or "
          "raise memory_limit",
          chunk_request_base + window_end, length,
          bind_data.range_bytes_budget);
    }
    if (window_end > window_start &&
        window_bytes + length > bind_data.range_bytes_budget) {
      break;
    }
    window_bytes += length;
    window_end++;
  }
  D_ASSERT(window_end > window_start);

  vector<LanceBlobRangeRequest> requests;
  requests.reserve(window_end - window_start);
  for (idx_t row = window_start; row < window_end; row++) {
    LanceBlobRangeRequest request = {};
    const std::pair<idx_t, const char *> fields[] = {
        {bind_data.row_idx, LANCE_BLOB_ROW_COLUMN},
        {bind_data.offset_idx, LANCE_BLOB_OFFSET_COLUMN},
        {bind_data.length_idx, LANCE_BLOB_LENGTH_COLUMN},
    };
    uint64_t values[3] = {};
    for (idx_t field_idx = 0; field_idx < 3; field_idx++) {
      ValidateBlobRangeValue(
          input.GetValue(fields[field_idx].first, NumericCast<idx_t>(row)),
          fields[field_idx].second, chunk_request_base + row,
          values[field_idx]);
    }
    request.row = values[0];
    request.offset = values[1];
    request.length = values[2];
    requests.push_back(request);
  }

  LanceBlobRangeResultGuard result_guard;
  if (lance_read_blob_ranges(bind_data.dataset, bind_data.column.c_str(),
                             bind_data.selector, requests.data(),
                             requests.size(), &result_guard.results,
                             &result_guard.results_len) != 0) {
    throw IOException("Lance blob range read failed: " +
                      LanceConsumeLastError());
  }
  const size_t results_len = result_guard.results_len;
  const LanceBlobRangeResult *results = result_guard.results;
  if (results_len != requests.size()) {
    throw IOException("Lance blob range read returned " +
                      std::to_string(results_len) + " results for " +
                      std::to_string(requests.size()) + " requests");
  }

  auto request_index_data =
      FlatVector::GetDataMutable<uint64_t>(output.data[0]);
  auto row_address_data = FlatVector::GetDataMutable<uint64_t>(output.data[2]);
  auto version_data = FlatVector::GetDataMutable<uint64_t>(output.data[4]);
  SelectionVector request_id_sel(NumericCast<idx_t>(results_len));
  for (idx_t result_idx = 0; result_idx < NumericCast<idx_t>(results_len);
       result_idx++) {
    const auto &result = results[result_idx];
    request_index_data[result_idx] =
        chunk_request_base + window_start + result.request_index;
    row_address_data[result_idx] = result.row_address;
    version_data[result_idx] = bind_data.dataset_version;
    // Lance reports the request index inside this window; translate it back to
    // the position inside the input chunk before slicing request ids.
    request_id_sel.set_index(
        result_idx, NumericCast<sel_t>(window_start + result.request_index));
    if (result.is_null) {
      FlatVector::SetNull(output.data[3], result_idx, true);
    } else {
      auto value = StringVector::AddStringOrBlob(
          output.data[3],
          result.data_len == 0
              ? ""
              : reinterpret_cast<const char *>(result.data_ptr),
          result.data_len);
      FlatVector::GetDataMutable<string_t>(output.data[3])[result_idx] = value;
    }
  }
  output.data[1].Slice(input.data[bind_data.request_id_idx], request_id_sel,
                       NumericCast<idx_t>(results_len));
  // The physical table-in-out operator synchronizes child vector sizes via
  // SetChildCardinality after this call, so only the logical count is set here.
  output.SetCardinalityUnsafe(NumericCast<idx_t>(results_len));

  if (window_end >= input_rows) {
    local_state.next_input_row = 0;
    return OperatorResultType::NEED_MORE_INPUT;
  }
  local_state.next_input_row = window_end;
  return OperatorResultType::HAVE_MORE_OUTPUT;
}

void RegisterLanceRangeRead(ExtensionLoader &loader) {
  TableFunction range_read(
      "lance_read_blob_ranges",
      FunctionSignature()
          .AddPositionalOnly("dataset", LogicalType::VARCHAR)
          .AddPositionalOnly("column", LogicalType::VARCHAR)
          .AddPositionalOnly("version", LogicalType::BIGINT)
          .AddPositionalOnly("selector", LogicalType::VARCHAR)
          .AddPositionalOnly("requests", LogicalType::TABLE),
      nullptr, LanceBlobRangeReadBind);
  range_read.in_out_function = LanceBlobRangeReadFunc;
  range_read.init_global = LanceBlobRangeReadInitGlobal;
  range_read.init_local = LanceBlobRangeReadInitLocal;
  loader.RegisterFunction(std::move(range_read));
}

} // namespace duckdb
