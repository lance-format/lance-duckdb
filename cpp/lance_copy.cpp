#include "duckdb.hpp"
#include "duckdb/function/copy_function.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_copy_function_info.hpp"
// Arrow integration headers would go here when available

#include <memory>

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
    
    // Writer operations
    void* lance_create_writer(const char* path, void* arrow_schema);
    void lance_write_batch(void* writer, void* arrow_batch);
    void lance_finish_writer(void* writer);
    void lance_close_writer(void* writer);
    
    // Schema conversion
    void* lance_duckdb_to_arrow_schema(const char** names, const char** types, int64_t num_fields);
}

namespace duckdb {

struct LanceCopyBindData : public FunctionData {
    string file_path;
    vector<string> column_names;
    vector<LogicalType> column_types;
    void* writer = nullptr;
    
    ~LanceCopyBindData() {
        if (writer) {
            lance_close_writer(writer);
        }
    }
    
    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<LanceCopyBindData>();
        result->file_path = file_path;
        result->column_names = column_names;
        result->column_types = column_types;
        result->writer = nullptr; // Don't copy the writer
        return std::move(result);
    }
    
    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<LanceCopyBindData>();
        return file_path == other.file_path;
    }
};

struct LanceCopyGlobalState : public GlobalFunctionData {
    mutex lock;
    idx_t total_rows = 0;
};

struct LanceCopyLocalState : public LocalFunctionData {
    unique_ptr<DataChunk> chunk;
};

// COPY TO functions (write Lance files)
static unique_ptr<FunctionData> LanceCopyToBind(ClientContext &context, CopyFunctionBindInput &input,
                                                const vector<string> &names, const vector<LogicalType> &sql_types) {
    auto bind_data = make_uniq<LanceCopyBindData>();
    bind_data->file_path = input.info.file_path;
    bind_data->column_names = names;
    bind_data->column_types = sql_types;
    
    // Create Arrow schema from DuckDB schema
    vector<const char*> name_ptrs;
    vector<const char*> type_ptrs;
    vector<string> type_strings;
    
    for (size_t i = 0; i < names.size(); i++) {
        name_ptrs.push_back(names[i].c_str());
        
        // Convert DuckDB type to Arrow type string
        string arrow_type;
        switch (sql_types[i].id()) {
            case LogicalTypeId::BIGINT:
                arrow_type = "int64";
                break;
            case LogicalTypeId::INTEGER:
                arrow_type = "int32";
                break;
            case LogicalTypeId::DOUBLE:
                arrow_type = "double";
                break;
            case LogicalTypeId::FLOAT:
                arrow_type = "float";
                break;
            case LogicalTypeId::VARCHAR:
                arrow_type = "utf8";
                break;
            case LogicalTypeId::BOOLEAN:
                arrow_type = "bool";
                break;
            default:
                arrow_type = "utf8";  // Default to string
                break;
        }
        type_strings.push_back(arrow_type);
        type_ptrs.push_back(type_strings.back().c_str());
    }
    
    void* arrow_schema = lance_duckdb_to_arrow_schema(name_ptrs.data(), type_ptrs.data(), names.size());
    
    // Create Lance writer
    bind_data->writer = lance_create_writer(bind_data->file_path.c_str(), arrow_schema);
    if (!bind_data->writer) {
        throw IOException("Failed to create Lance writer for: " + bind_data->file_path);
    }
    
    return std::move(bind_data);
}

static unique_ptr<GlobalFunctionData> LanceCopyToInitGlobal(ClientContext &context, FunctionData &bind_data,
                                                            const string &file_path) {
    return make_uniq_base<GlobalFunctionData, LanceCopyGlobalState>();
}

static unique_ptr<LocalFunctionData> LanceCopyToInitLocal(ExecutionContext &context, FunctionData &bind_data) {
    auto local_state = make_uniq_base<LocalFunctionData, LanceCopyLocalState>();
    local_state->Cast<LanceCopyLocalState>().chunk = make_uniq<DataChunk>();
    return local_state;
}

static void LanceCopyToSink(ExecutionContext &context, FunctionData &bind_data_p, GlobalFunctionData &gstate,
                           LocalFunctionData &lstate, DataChunk &input) {
    auto &bind_data = bind_data_p.Cast<LanceCopyBindData>();
    auto &global_state = gstate.Cast<LanceCopyGlobalState>();
    
    if (input.size() == 0) {
        return;
    }
    
    lock_guard<mutex> lock(global_state.lock);
    
    // Convert DataChunk to Arrow format and write
    // This is simplified - actual implementation needs proper Arrow conversion
    // For now, we'll need to implement the conversion in the Rust side
    
    // TODO: Implement DataChunk to Arrow conversion
    // void* arrow_batch = ConvertToArrow(input);
    // lance_write_batch(bind_data.writer, arrow_batch);
    
    global_state.total_rows += input.size();
}

static void LanceCopyToCombine(ExecutionContext &context, FunctionData &bind_data, GlobalFunctionData &gstate,
                               LocalFunctionData &lstate) {
    // Combine local state into global state if needed
}

static void LanceCopyToFinalize(ClientContext &context, FunctionData &bind_data_p, GlobalFunctionData &gstate) {
    auto &bind_data = bind_data_p.Cast<LanceCopyBindData>();
    auto &global_state = gstate.Cast<LanceCopyGlobalState>();
    
    // Finalize the Lance writer
    if (bind_data.writer) {
        lance_finish_writer(bind_data.writer);
    }
}

// COPY FROM functions (read Lance files)
static unique_ptr<FunctionData> LanceCopyFromBind(ClientContext &context, CopyFunctionBindInput &input,
                                                  vector<string> &expected_names,
                                                  vector<LogicalType> &expected_types) {
    auto bind_data = make_uniq<LanceCopyBindData>();
    bind_data->file_path = input.info.file_path;
    
    // Open dataset to get schema
    void* dataset = lance_open_dataset(bind_data->file_path.c_str());
    if (!dataset) {
        throw IOException("Failed to open Lance dataset: " + bind_data->file_path);
    }
    
    // Get schema and populate expected_names and expected_types
    void* schema = lance_get_schema(dataset);
    int64_t num_fields = lance_schema_num_fields(schema);
    
    for (int64_t i = 0; i < num_fields; i++) {
        expected_names.push_back(lance_schema_field_name(schema, i));
        
        string field_type(lance_schema_field_type(schema, i));
        LogicalType duckdb_type;
        
        if (field_type == "int64") {
            duckdb_type = LogicalType::BIGINT;
        } else if (field_type == "int32") {
            duckdb_type = LogicalType::INTEGER;
        } else if (field_type == "double") {
            duckdb_type = LogicalType::DOUBLE;
        } else if (field_type == "float") {
            duckdb_type = LogicalType::FLOAT;
        } else if (field_type == "utf8" || field_type == "string") {
            duckdb_type = LogicalType::VARCHAR;
        } else if (field_type == "bool") {
            duckdb_type = LogicalType::BOOLEAN;
        } else {
            duckdb_type = LogicalType::VARCHAR;
        }
        
        expected_types.push_back(duckdb_type);
    }
    
    lance_free_schema(schema);
    lance_close_dataset(dataset);
    
    return std::move(bind_data);
}

static void LanceCopyFromFunction(ExecutionContext &context, 
                                  DataChunk &output, 
                                  FunctionData &bind_data_p) {
    // This would implement the actual reading logic
    // For now, set cardinality to 0 to indicate end of data
    output.SetCardinality(0);
}

void RegisterLanceCopy(ExtensionLoader &loader) {
    CopyFunction lance_copy("lance");
    
    // COPY TO functions
    lance_copy.copy_to_bind = LanceCopyToBind;
    lance_copy.copy_to_initialize_global = LanceCopyToInitGlobal;
    lance_copy.copy_to_initialize_local = LanceCopyToInitLocal;
    lance_copy.copy_to_sink = LanceCopyToSink;
    lance_copy.copy_to_combine = LanceCopyToCombine;
    lance_copy.copy_to_finalize = LanceCopyToFinalize;
    
    // COPY FROM functions - TODO: implement later
    // lance_copy.copy_from_bind = LanceCopyFromBind;
    // lance_copy.copy_from_function = LanceCopyFromFunction;
    
    lance_copy.extension = "lance";
    
    loader.RegisterFunction(lance_copy);
}

} // namespace duckdb
