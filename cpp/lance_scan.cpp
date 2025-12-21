#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"

#include <memory>
#include <mutex>

// FFI declarations for Rust functions
extern "C" {
    // Dataset operations
    void* lance_open_dataset(const char* path);
    void lance_close_dataset(void* dataset);
    
    // Schema operations
    void* lance_get_schema(void* dataset);
    void lance_free_schema(void* schema);
    int64_t lance_schema_num_fields(void* schema);
    const char* lance_schema_field_name(void* schema, int64_t index);
    const char* lance_schema_field_type(void* schema, int64_t index);
    
    // Stream operations
    void* lance_create_stream(void* dataset);
    void* lance_stream_next(void* stream);
    void lance_close_stream(void* stream);
    void lance_free_batch(void* batch);
    int64_t lance_batch_num_rows(void* batch);
    
    // Arrow C Data Interface
    int32_t lance_batch_to_arrow(void* batch, ArrowArray* out_array, ArrowSchema* out_schema);
    
    // Fallback: Direct data access functions (kept for compatibility)
    int64_t lance_batch_get_int64_column(void* batch, int64_t col_idx, int64_t* out_data);
    int64_t lance_batch_get_float64_column(void* batch, int64_t col_idx, double* out_data);
    const char* lance_batch_get_string_value(void* batch, int64_t col_idx, int64_t row_idx);
}

namespace duckdb {

struct LanceScanBindData : public TableFunctionData {
    string file_path;
    vector<string> column_names;
    vector<LogicalType> column_types;
    void* dataset = nullptr;
    
    ~LanceScanBindData() {
        if (dataset) {
            lance_close_dataset(dataset);
        }
    }
};

struct LanceScanGlobalState : public GlobalTableFunctionState {
    mutex lock;
    void* stream = nullptr;
    void* current_batch = nullptr;
    bool finished = false;
    
    ~LanceScanGlobalState() {
        if (current_batch) {
            lance_free_batch(current_batch);
        }
        if (stream) {
            lance_close_stream(stream);
        }
    }
};

struct LanceScanLocalState : public LocalTableFunctionState {};

static unique_ptr<FunctionData> LanceScanBind(ClientContext &context, TableFunctionBindInput &input,
                                               vector<LogicalType> &return_types, vector<string> &names) {
    auto bind_data = make_uniq<LanceScanBindData>();
    
    // Get the file path from the input
    if (input.inputs.empty() || input.inputs[0].IsNull()) {
        throw InvalidInputException("lance_scan requires a file path");
    }
    bind_data->file_path = input.inputs[0].GetValue<string>();
    
    // Open the Lance dataset
    bind_data->dataset = lance_open_dataset(bind_data->file_path.c_str());
    if (!bind_data->dataset) {
        throw IOException("Failed to open Lance dataset: " + bind_data->file_path);
    }
    
    // Get schema from the dataset
    void* schema = lance_get_schema(bind_data->dataset);
    if (!schema) {
        throw IOException("Failed to get schema from Lance dataset");
    }
    
    // Extract column names and types
    int64_t num_fields = lance_schema_num_fields(schema);
    for (int64_t i = 0; i < num_fields; i++) {
        const char* field_name = lance_schema_field_name(schema, i);
        const char* field_type = lance_schema_field_type(schema, i);
        
        bind_data->column_names.push_back(string(field_name));
        
        // Map Lance/Arrow types to DuckDB types
        // For now, use simplified mapping
        string type_str(field_type);
        LogicalType duckdb_type;
        
        if (type_str == "int32" || type_str == "int64") {
            duckdb_type = LogicalType::BIGINT;
        } else if (type_str == "float" || type_str == "double") {
            duckdb_type = LogicalType::DOUBLE;
        } else if (type_str == "string" || type_str == "utf8") {
            duckdb_type = LogicalType::VARCHAR;
        } else if (type_str == "bool") {
            duckdb_type = LogicalType::BOOLEAN;
        } else {
            // Default to VARCHAR for unknown types
            duckdb_type = LogicalType::VARCHAR;
        }
        
        bind_data->column_types.push_back(duckdb_type);
    }
    
    lance_free_schema(schema);
    
    // Set return types and names
    return_types = bind_data->column_types;
    names = bind_data->column_names;
    
    return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> LanceScanInit(ClientContext &context, TableFunctionInitInput &input) {
    auto state = make_uniq_base<GlobalTableFunctionState, LanceScanGlobalState>();
    auto &bind_data = input.bind_data->Cast<LanceScanBindData>();
    auto &scan_state = state->Cast<LanceScanGlobalState>();
    
    // Create a stream for reading batches
    scan_state.stream = lance_create_stream(bind_data.dataset);
    if (!scan_state.stream) {
        throw IOException("Failed to create Lance stream");
    }
    
    return state;
}

static unique_ptr<LocalTableFunctionState> LanceScanLocalInit(ExecutionContext &context, TableFunctionInitInput &input,
                                                              GlobalTableFunctionState *global_state) {
    return make_uniq_base<LocalTableFunctionState, LanceScanLocalState>();
}

static void LanceScanFunc(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
    auto &bind_data = data.bind_data->Cast<LanceScanBindData>();
    auto &global_state = data.global_state->Cast<LanceScanGlobalState>();
    
    lock_guard<mutex> lock(global_state.lock);
    
    if (global_state.finished) {
        return;
    }
    
    // Read next batch from stream
    if (!global_state.current_batch) {
        global_state.current_batch = lance_stream_next(global_state.stream);
        if (!global_state.current_batch) {
            global_state.finished = true;
            return;
        }
    }
    
    // Get the number of rows in the batch
    idx_t num_rows = lance_batch_num_rows(global_state.current_batch);
    if (num_rows == 0) {
        global_state.finished = true;
        return;
    }
    
    output.SetCardinality(num_rows);
    
    // Try to use Arrow C Data Interface for better performance
    ArrowArray arrow_array;
    ArrowSchema arrow_schema;
    bool use_arrow_interface = false;
    
    if (lance_batch_to_arrow(global_state.current_batch, &arrow_array, &arrow_schema) == 0) {
        // Successfully exported to Arrow C Data Interface
        // TODO: Convert from Arrow to DuckDB using the C Data Interface
        // For now, we'll fall back to the type-specific functions
        // In a production implementation, we would use DuckDB's Arrow conversion utilities
        use_arrow_interface = false;  // Disabled until we integrate with DuckDB's Arrow utilities
        
        // Clean up Arrow structures if we're not using them
        if (arrow_array.release) {
            arrow_array.release(&arrow_array);
        }
        if (arrow_schema.release) {
            arrow_schema.release(&arrow_schema);
        }
    }
    
    // Fallback to type-specific functions
    if (!use_arrow_interface) {
        for (idx_t col = 0; col < bind_data.column_types.size(); col++) {
            auto &vec = output.data[col];
            auto &type = bind_data.column_types[col];
            
            if (type == LogicalType::BIGINT) {
                auto data = FlatVector::GetData<int64_t>(vec);
                int64_t result = lance_batch_get_int64_column(global_state.current_batch, col, data);
                if (result < 0) {
                    // Failed to get data, fill with default values
                    for (idx_t row = 0; row < num_rows; row++) {
                        data[row] = 0;
                    }
                }
            } else if (type == LogicalType::DOUBLE) {
                auto data = FlatVector::GetData<double>(vec);
                int64_t result = lance_batch_get_float64_column(global_state.current_batch, col, data);
                if (result < 0) {
                    // Failed to get data, fill with default values
                    for (idx_t row = 0; row < num_rows; row++) {
                        data[row] = 0.0;
                    }
                }
            } else if (type == LogicalType::VARCHAR) {
                auto data = FlatVector::GetData<string_t>(vec);
                for (idx_t row = 0; row < num_rows; row++) {
                    const char* str_value = lance_batch_get_string_value(global_state.current_batch, col, row);
                    if (str_value) {
                        data[row] = StringVector::AddString(vec, str_value);
                    } else {
                        // Set empty string for null values
                        data[row] = StringVector::AddString(vec, "");
                    }
                }
            } else {
                // Unsupported type, fill with default values
                for (idx_t row = 0; row < num_rows; row++) {
                    FlatVector::SetNull(vec, row, true);
                }
            }
        }
    }
    
    // Mark batch as consumed
    lance_free_batch(global_state.current_batch);
    global_state.current_batch = nullptr;
}

void RegisterLanceScan(ExtensionLoader &loader) {
    TableFunction lance_scan("lance_scan", {LogicalType::VARCHAR}, LanceScanFunc, LanceScanBind, 
                            LanceScanInit, LanceScanLocalInit);
    
    lance_scan.name = "lance_scan";
    lance_scan.projection_pushdown = false;  // TODO: implement later
    lance_scan.filter_pushdown = false;      // TODO: implement later
    
    loader.RegisterFunction(lance_scan);
}

} // namespace duckdb
