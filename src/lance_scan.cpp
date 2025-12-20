#include "duckdb.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension_util.hpp"

#include <cstring>
#include <mutex>

extern "C" {
void *lance_open_dataset(const char *path);
void lance_close_dataset(void *dataset);

void *lance_get_schema(void *dataset);
void lance_free_schema(void *schema);
int32_t lance_schema_to_arrow(void *schema, ArrowSchema *out_schema);

void *lance_create_stream(void *dataset);
void *lance_stream_next(void *stream);
void lance_close_stream(void *stream);

void lance_free_batch(void *batch);
int64_t lance_batch_num_rows(void *batch);
int32_t lance_batch_to_arrow(void *batch, ArrowArray *out_array, ArrowSchema *out_schema);
}

namespace duckdb {

struct LanceScanBindData : public TableFunctionData {
	string file_path;
	void *dataset = nullptr;
	ArrowSchemaWrapper schema_root;
	ArrowTableType arrow_table;

	~LanceScanBindData() override {
		if (dataset) {
			lance_close_dataset(dataset);
		}
	}
};

struct LanceScanGlobalState : public GlobalTableFunctionState {
	std::mutex lock;
	void *stream = nullptr;
	bool finished = false;
	idx_t lines_read = 0;

	~LanceScanGlobalState() override {
		if (stream) {
			lance_close_stream(stream);
		}
	}
};

static unique_ptr<FunctionData> LanceScanBind(ClientContext &context, TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw InvalidInputException("lance_scan requires a dataset root path");
	}

	auto result = make_uniq<LanceScanBindData>();
	result->file_path = input.inputs[0].GetValue<string>();

	result->dataset = lance_open_dataset(result->file_path.c_str());
	if (!result->dataset) {
		throw IOException("Failed to open Lance dataset: " + result->file_path);
	}

	auto *schema_handle = lance_get_schema(result->dataset);
	if (!schema_handle) {
		throw IOException("Failed to get schema from Lance dataset: " + result->file_path);
	}

	if (lance_schema_to_arrow(schema_handle, &result->schema_root.arrow_schema) != 0) {
		lance_free_schema(schema_handle);
		throw IOException("Failed to export Lance schema to Arrow C Data Interface");
	}
	lance_free_schema(schema_handle);

	auto &config = DBConfig::GetConfig(context);
	ArrowTableFunction::PopulateArrowTableType(config, result->arrow_table, result->schema_root, names, return_types);
	return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> LanceScanInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto state = make_uniq_base<GlobalTableFunctionState, LanceScanGlobalState>();
	auto &bind_data = input.bind_data->Cast<LanceScanBindData>();
	auto &scan_state = state->Cast<LanceScanGlobalState>();

	scan_state.stream = lance_create_stream(bind_data.dataset);
	if (!scan_state.stream) {
		throw IOException("Failed to create Lance stream");
	}
	return state;
}

static unique_ptr<LocalTableFunctionState> LanceScanLocalInit(ExecutionContext &context, TableFunctionInitInput &input,
                                                              GlobalTableFunctionState *global_state) {
	auto chunk = make_uniq<ArrowArrayWrapper>();
	return make_uniq_base<LocalTableFunctionState, ArrowScanLocalState>(std::move(chunk), context.client);
}

static void LanceScanFunc(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	if (!data.local_state) {
		return;
	}

	auto &bind_data = data.bind_data->Cast<LanceScanBindData>();
	auto &global_state = data.global_state->Cast<LanceScanGlobalState>();
	auto &scan_state = data.local_state->Cast<ArrowScanLocalState>();

	std::lock_guard<std::mutex> lock(global_state.lock);
	if (global_state.finished) {
		return;
	}

	if (scan_state.chunk_offset >= NumericCast<idx_t>(scan_state.chunk->arrow_array.length)) {
		auto *batch = lance_stream_next(global_state.stream);
		if (!batch) {
			global_state.finished = true;
			return;
		}

		auto new_chunk = make_shared_ptr<ArrowArrayWrapper>();
		ArrowSchema tmp_schema;
		memset(&tmp_schema, 0, sizeof(tmp_schema));

		if (lance_batch_to_arrow(batch, &new_chunk->arrow_array, &tmp_schema) != 0) {
			lance_free_batch(batch);
			throw IOException("Failed to export Lance RecordBatch to Arrow C Data Interface");
		}

		lance_free_batch(batch);

		if (tmp_schema.release) {
			tmp_schema.release(&tmp_schema);
		}

		scan_state.chunk = std::move(new_chunk);
		scan_state.Reset();
	}

	auto remaining = NumericCast<idx_t>(scan_state.chunk->arrow_array.length) - scan_state.chunk_offset;
	auto output_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, remaining);
	output.SetCardinality(output_size);

	ArrowTableFunction::ArrowToDuckDB(scan_state, bind_data.arrow_table.GetColumns(), output, global_state.lines_read);
	global_state.lines_read += output_size;
	scan_state.chunk_offset += output.size();

	output.Verify();
}

void RegisterLanceScan(DatabaseInstance &instance) {
	TableFunction lance_scan("lance_scan", {LogicalType::VARCHAR}, LanceScanFunc, LanceScanBind, LanceScanInitGlobal,
	                         LanceScanLocalInit);
	lance_scan.projection_pushdown = false;
	lance_scan.filter_pushdown = false;
	ExtensionUtil::RegisterFunction(instance, lance_scan);
}

} // namespace duckdb
