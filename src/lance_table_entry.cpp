#include "lance_table_entry.hpp"
#include "lance_arrow_compat.hpp"
#include "lance_blob.hpp"
#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_ffi.hpp"

#include "duckdb/common/arrow/schema_metadata.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/constraints/not_null_constraint.hpp"
#include "duckdb/parser/expression/cast_expression.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/parsed_data/comment_on_column_info.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/transaction/transaction_context.hpp"

#include <algorithm>
#include <cstring>
#include <memory>

namespace duckdb {

static string LanceHexEncode(const string &bytes) {
  static constexpr char kHexDigits[] = "0123456789ABCDEF";
  string result;
  result.reserve(bytes.size() * 2);
  for (auto byte : bytes) {
    result.push_back(kHexDigits[(static_cast<uint8_t>(byte) >> 4) & 0x0F]);
    result.push_back(kHexDigits[static_cast<uint8_t>(byte) & 0x0F]);
  }
  return result;
}

static void RestoreColumnDefault(ClientContext &context,
                                 const ArrowSchema &field,
                                 ColumnDefinition &column) {
  ArrowSchemaMetadata metadata(field.metadata);
  auto sql = metadata.GetOption("duckdb_default_expr");
  if (sql.empty()) {
    return;
  }
  auto expressions = Parser(context).ParseExpressionList(sql);
  if (expressions.size() != 1) {
    throw InvalidInputException(
        "Lance column has an invalid stored DEFAULT expression");
  }
  column.SetDefaultValue(std::move(expressions[0]));
}

LanceTableSchema LanceTableSchema::FromArrow(ClientContext &context,
                                             ArrowSchema &schema,
                                             ArrowTableSchema &arrow_table) {
  LanceTableSchema result;
  result.coerced_columns = LanceCoerceArrowSchemaForDuckDB(&schema);
  ArrowTableFunction::PopulateArrowTableSchema(context, arrow_table, schema);
  const auto names = arrow_table.GetNames();
  const auto types = arrow_table.GetTypes();
  if (names.size() != types.size()) {
    throw InternalException(
        "Arrow table schema returned mismatched names/types sizes");
  }
  for (idx_t i = 0; i < names.size(); i++) {
    auto &field = *schema.children[i];
    ColumnDefinition column(Identifier(names[i]), types[i]);
    RestoreColumnDefault(context, field, column);
    result.columns.AddColumn(std::move(column));
    result.blob_columns.push_back(LanceArrowFieldContainsBlob(field));
    // Only genuinely Blob v2 fields accept the LANCE_BLOB_INPUT write type.
    // Plain binary, legacy Blob, and Arrow extension-typed BLOB columns keep
    // DuckDB's ordinary BLOB cast path.
    if (types[i].id() == LogicalTypeId::BLOB) {
      ArrowSchemaMetadata metadata(field.metadata);
      auto extension = metadata.GetOption("ARROW:extension:name");
      if (StringUtil::CIEquals(extension, "lance.blob.v2")) {
        result.writable_blob_v2_columns.push_back(names[i]);
      }
    }
    if ((field.flags & ARROW_FLAG_NULLABLE) == 0) {
      result.constraints.push_back(
          make_uniq<NotNullConstraint>(LogicalIndex(i)));
    }
  }
  return result;
}

LanceTableSchema LanceTableSchema::FromDataset(ClientContext &context,
                                               void *dataset) {
  std::unique_ptr<void, decltype(&lance_free_schema)> handle(
      lance_get_schema(dataset), lance_free_schema);
  if (!handle) {
    throw IOException("Failed to get schema from Lance dataset" +
                      LanceFormatErrorSuffix());
  }
  ArrowSchemaWrapper root;
  root.arrow_schema = {};
  if (lance_schema_to_arrow(handle.get(), &root.arrow_schema) != 0) {
    throw IOException(
        "Failed to export Lance schema to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  ArrowTableSchema arrow_table;
  return FromArrow(context, root.arrow_schema, arrow_table);
}

LanceTableEntry::LanceTableEntry(Catalog &catalog, SchemaCatalogEntry &schema,
                                 CreateTableInfo &info, string dataset_uri)
    : TableCatalogEntry(catalog, schema, info),
      columns(std::move(info.columns)), dataset_uri(std::move(dataset_uri)) {
  timestamp = 0;
}

static string
LanceNamespaceDisplayUri(const LanceNamespaceTableConfig &config) {
  if (!config.display_uri.empty()) {
    return config.display_uri;
  }
  if (config.IsDirectory()) {
    return config.root + "/" + config.table_id;
  }
  return config.endpoint + "/" + config.table_id;
}

LanceTableEntry::LanceTableEntry(Catalog &catalog, SchemaCatalogEntry &schema,
                                 CreateTableInfo &info,
                                 LanceNamespaceTableConfig config)
    : TableCatalogEntry(catalog, schema, info),
      columns(std::move(info.columns)),
      dataset_uri(LanceNamespaceDisplayUri(config)),
      namespace_config(
          make_uniq<LanceNamespaceTableConfig>(std::move(config))) {
  timestamp = 0;
}

static unordered_map<string, string> ParseTsvKvs(const char *ptr) {
  unordered_map<string, string> out;
  if (!ptr) {
    return out;
  }

  string joined = ptr;
  lance_free_string(ptr);

  for (auto &line : StringUtil::Split(joined, '\n')) {
    if (line.empty()) {
      continue;
    }
    auto parts = StringUtil::Split(line, '\t');
    if (parts.size() != 2) {
      continue;
    }
    out[std::move(parts[0])] = std::move(parts[1]);
  }
  return out;
}

static void LoadLanceColumnComments(void *dataset, ColumnList &columns) {
  for (idx_t i = 0; i < columns.LogicalColumnCount(); i++) {
    auto &column = columns.GetColumnMutable(LogicalIndex(i));
    auto *metadata =
        lance_dataset_list_field_metadata(dataset, column.Name().c_str());
    if (!metadata) {
      throw IOException("Failed to list field metadata from Lance dataset" +
                        LanceFormatErrorSuffix());
    }
    auto kvs = ParseTsvKvs(metadata);
    auto it = kvs.find("comment");
    if (it != kvs.end()) {
      column.SetComment(Value(it->second));
    }
  }
}

static unique_ptr<CatalogEntry>
BuildUpdatedLanceTableEntry(ClientContext &context, const LanceTableEntry &base,
                            bool internal) {
  string display_uri;
  void *dataset = LanceOpenDatasetForTable(context, base, display_uri);
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + display_uri +
                      LanceFormatErrorSuffix());
  }

  auto &catalog = base.catalog;
  auto &schema = base.schema;

  CreateTableInfo create_info(schema, base.name);
  create_info.internal = internal;
  create_info.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;

  vector<string> coerced;
  vector<bool> blob_columns;
  vector<string> writable_blob_v2_columns;
  try {
    auto live = LanceTableSchema::FromDataset(context, dataset);
    create_info.columns = std::move(live.columns);
    create_info.constraints = std::move(live.constraints);
    coerced = std::move(live.coerced_columns);
    blob_columns = std::move(live.blob_columns);
    writable_blob_v2_columns = std::move(live.writable_blob_v2_columns);
    LoadLanceColumnComments(dataset, create_info.columns);
  } catch (...) {
    lance_close_dataset(dataset);
    throw;
  }

  unique_ptr<LanceTableEntry> entry;
  if (base.IsNamespaceBacked()) {
    entry = make_uniq<LanceTableEntry>(catalog, schema, create_info,
                                       base.NamespaceConfig());
  } else {
    entry = make_uniq<LanceTableEntry>(catalog, schema, create_info,
                                       base.DatasetUri());
  }
  entry->SetCoercedColumnNames(std::move(coerced));
  entry->SetBlobColumns(std::move(blob_columns));
  entry->SetWritableBlobV2Columns(std::move(writable_blob_v2_columns));

  auto *table_md = lance_dataset_list_table_metadata(dataset);
  if (!table_md) {
    lance_close_dataset(dataset);
    throw IOException("Failed to list table metadata from Lance dataset" +
                      LanceFormatErrorSuffix());
  }
  auto table_kvs = ParseTsvKvs(table_md);
  auto it = table_kvs.find("comment");
  if (it != table_kvs.end()) {
    entry->comment = Value(it->second);
  }

  lance_close_dataset(dataset);
  return unique_ptr_cast<LanceTableEntry, CatalogEntry>(std::move(entry));
}

static void ValidateAlterColumnTypeTarget(const LogicalType &type) {
  switch (type.id()) {
  case LogicalTypeId::BOOLEAN:
  case LogicalTypeId::TINYINT:
  case LogicalTypeId::UTINYINT:
  case LogicalTypeId::SMALLINT:
  case LogicalTypeId::USMALLINT:
  case LogicalTypeId::INTEGER:
  case LogicalTypeId::UINTEGER:
  case LogicalTypeId::BIGINT:
  case LogicalTypeId::UBIGINT:
  case LogicalTypeId::FLOAT:
  case LogicalTypeId::DOUBLE:
  case LogicalTypeId::DATE:
  case LogicalTypeId::TIME:
  case LogicalTypeId::TIMESTAMP:
  case LogicalTypeId::TIMESTAMP_TZ:
  case LogicalTypeId::VARCHAR:
  case LogicalTypeId::BLOB:
  case LogicalTypeId::ARRAY:
    return;
  default:
    break;
  }
  throw NotImplementedException(
      "Lance ALTER COLUMN TYPE only supports a limited set of DuckDB types "
      "(BOOLEAN, integer/floating, DATE/TIME/TIMESTAMP, VARCHAR, BLOB, "
      "ARRAY).");
}

static bool IsImplicitCastUsingExpression(const ParsedExpression &expr,
                                          const string &column_name,
                                          const LogicalType &target_type) {
  if (expr.GetExpressionClass() != ExpressionClass::CAST) {
    return false;
  }
  auto &cast_expr = expr.Cast<CastExpression>();
  if (cast_expr.IsTryCast()) {
    return false;
  }

  LogicalType expression_cast_type;
  try {
    expression_cast_type = UnboundType::TryDefaultBind(cast_expr.TargetType());
  } catch (...) {
    return false;
  }
  if (expression_cast_type != target_type) {
    return false;
  }
  if (cast_expr.Child().GetExpressionClass() != ExpressionClass::COLUMN_REF) {
    return false;
  }
  auto &col_ref = cast_expr.Child().Cast<ColumnRefExpression>();
  if (col_ref.ColumnNames().empty()) {
    return false;
  }
  // Accept qualified references and match the leaf name against target column.
  return col_ref.ColumnNames().back() == column_name;
}

unique_ptr<CatalogEntry>
LanceTableEntry::Rebuild(ClientContext &context) const {
  return BuildUpdatedLanceTableEntry(context, *this, internal);
}

unique_ptr<CatalogEntry> LanceTableEntry::AlterEntry(ClientContext &context,
                                                     AlterInfo &info) {
  if (!context.transaction.IsAutoCommit()) {
    throw NotImplementedException(
        "Lance DDL does not support explicit transactions yet");
  }

  string display_uri;
  void *dataset = LanceOpenDatasetForTable(context, *this, display_uri);
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + display_uri +
                      LanceFormatErrorSuffix());
  }

  unique_ptr<CatalogEntry> result;
  switch (info.type) {
  case AlterType::ALTER_TABLE: {
    auto &alter = info.Cast<AlterTableInfo>();
    switch (alter.alter_table_type) {
    case AlterTableType::ADD_COLUMN: {
      auto &add = info.Cast<AddColumnInfo>();
      vector<string> names{add.new_column.Name().GetIdentifierName()};
      vector<LogicalType> types{add.new_column.Type()};

      ArrowSchemaWrapper new_schema_root;
      memset(&new_schema_root.arrow_schema, 0,
             sizeof(new_schema_root.arrow_schema));
      auto props = context.GetClientProperties();
      ArrowConverter::ToArrowSchema(&new_schema_root.arrow_schema, types, names,
                                    props);
      LanceNormalizeArrowListFieldNames(&new_schema_root.arrow_schema);

      vector<string> expressions;
      if (add.new_column.HasDefaultValue()) {
        if (types[0].id() == LogicalTypeId::BLOB) {
          // Lance's DataFusion planner has no BLOB type. Fold the DuckDB
          // default with the native binder/evaluator, then hand the exact
          // bytes to the existing expression path as a hex literal. Only
          // constant-foldable BLOB defaults are supported: URI ingestion is a
          // DML value contract, not a stored default.
          try {
            auto default_binder = Binder::CreateBinder(context);
            vector<unique_ptr<Expression>> bound_defaults;
            default_binder->BindDefaultValue(
                add.new_column, bound_defaults,
                catalog.GetName().GetIdentifierName(),
                schema.name.GetIdentifierName());
            auto &bound_default = *bound_defaults[0];
            if (!bound_default.IsFoldable() || bound_default.IsVolatile()) {
              throw NotImplementedException(
                  "ALTER TABLE ADD COLUMN BLOB DEFAULT only supports "
                  "constant-foldable values, not columns or volatile "
                  "functions");
            }
            Value default_value;
            if (!ExpressionExecutor::TryEvaluateScalar(context, bound_default,
                                                       default_value)) {
              throw NotImplementedException(
                  "ALTER TABLE ADD COLUMN BLOB DEFAULT could not be folded to "
                  "a constant value");
            }
            if (default_value.IsNull()) {
              expressions.push_back("NULL");
            } else {
              auto bytes = default_value.GetValueUnsafe<string_t>().GetString();
              expressions.push_back("X'" + LanceHexEncode(bytes) + "'");
            }
          } catch (...) {
            lance_close_dataset(dataset);
            throw;
          }
        } else {
          expressions.push_back(add.new_column.DefaultValue().ToString());
        }
      }
      vector<const char *> expr_ptrs;
      expr_ptrs.reserve(expressions.size());
      for (auto &e : expressions) {
        expr_ptrs.push_back(e.c_str());
      }
      auto blob_flags = LanceBlobFlagsForTypes(types);

      auto rc = lance_dataset_add_columns(
          dataset, &new_schema_root.arrow_schema,
          expr_ptrs.empty() ? nullptr : expr_ptrs.data(), expr_ptrs.size(), 0,
          blob_flags.empty() ? nullptr : blob_flags.data(), blob_flags.size());
      if (rc != 0) {
        lance_close_dataset(dataset);
        throw IOException("Failed to add column to Lance dataset: " +
                          display_uri + LanceFormatErrorSuffix());
      }

      // Best-effort persistence of DuckDB column defaults in Lance field
      // metadata. Re-open the dataset after schema evolution so the new field
      // is visible to the metadata updater.
      lance_close_dataset(dataset);
      dataset = nullptr;
      if (add.new_column.HasDefaultValue()) {
        string reopen_uri;
        dataset = LanceOpenDatasetForTable(context, *this, reopen_uri);
        if (dataset) {
          auto default_expr = add.new_column.DefaultValue().ToString();
          (void)lance_dataset_update_field_metadata(
              dataset, add.new_column.Name().c_str(), "duckdb_default_expr",
              default_expr.c_str());
          lance_close_dataset(dataset);
          dataset = nullptr;
        }
      } else {
        // The dataset was already closed above, keep going.
      }
      LanceInvalidateDatasetCacheForTable(context, *this);
      return BuildUpdatedLanceTableEntry(context, *this, internal);
      break;
    }
    case AlterTableType::REMOVE_COLUMN: {
      auto &drop = info.Cast<RemoveColumnInfo>();
      if (drop.if_column_exists && !ColumnExists(drop.removed_column)) {
        lance_close_dataset(dataset);
        result = nullptr;
        break;
      }
      const char *cols[1] = {drop.removed_column.c_str()};
      auto rc = lance_dataset_drop_columns(dataset, cols, 1);
      if (rc != 0) {
        lance_close_dataset(dataset);
        throw IOException("Failed to drop column from Lance dataset: " +
                          display_uri + LanceFormatErrorSuffix());
      }
      lance_close_dataset(dataset);
      LanceInvalidateDatasetCacheForTable(context, *this);
      return BuildUpdatedLanceTableEntry(context, *this, internal);
      break;
    }
    case AlterTableType::RENAME_COLUMN: {
      auto &rename = info.Cast<RenameColumnInfo>();
      auto rc = lance_dataset_alter_columns_rename(
          dataset, rename.old_name.c_str(), rename.new_name.c_str());
      if (rc != 0) {
        lance_close_dataset(dataset);
        throw IOException("Failed to rename column in Lance dataset: " +
                          display_uri + LanceFormatErrorSuffix());
      }
      lance_close_dataset(dataset);
      LanceInvalidateDatasetCacheForTable(context, *this);
      return BuildUpdatedLanceTableEntry(context, *this, internal);
      break;
    }
    case AlterTableType::ALTER_COLUMN_TYPE: {
      auto &cast = info.Cast<ChangeColumnTypeInfo>();
      if (cast.column_path.size() != 1) {
        lance_close_dataset(dataset);
        throw NotImplementedException(
            "Lance ALTER COLUMN does not support nested field paths");
      }
      ValidateAlterColumnTypeTarget(cast.target_type);
      if (!cast.expression ||
          !IsImplicitCastUsingExpression(
              *cast.expression, cast.GetColumnName().GetIdentifierName(),
              cast.target_type)) {
        lance_close_dataset(dataset);
        throw NotImplementedException(
            "Lance ALTER COLUMN TYPE only supports implicit USING (i.e., a "
            "simple CAST of the original column)");
      }

      ArrowSchemaWrapper new_type_schema;
      memset(&new_type_schema.arrow_schema, 0,
             sizeof(new_type_schema.arrow_schema));
      auto props = context.GetClientProperties();
      ArrowConverter::ToArrowSchema(
          &new_type_schema.arrow_schema, {cast.target_type},
          {cast.GetColumnName().GetIdentifierName()}, props);
      LanceNormalizeArrowListFieldNames(&new_type_schema.arrow_schema);

      auto rc = lance_dataset_alter_columns_cast(
          dataset, cast.GetColumnName().GetIdentifierName().c_str(),
          &new_type_schema.arrow_schema);
      if (rc != 0) {
        lance_close_dataset(dataset);
        throw IOException("Failed to change column type in Lance dataset: " +
                          display_uri + LanceFormatErrorSuffix());
      }
      lance_close_dataset(dataset);
      LanceInvalidateDatasetCacheForTable(context, *this);
      return BuildUpdatedLanceTableEntry(context, *this, internal);
      break;
    }
    case AlterTableType::SET_NOT_NULL: {
      auto &nn = info.Cast<SetNotNullInfo>();
      if (nn.column_path.size() != 1) {
        lance_close_dataset(dataset);
        throw NotImplementedException(
            "Lance ALTER COLUMN does not support nested field paths");
      }
      auto rc = lance_dataset_alter_columns_set_nullable(
          dataset, nn.column_path[0].c_str(), 0);
      if (rc != 0) {
        lance_close_dataset(dataset);
        throw IOException("Failed to set NOT NULL in Lance dataset: " +
                          display_uri + LanceFormatErrorSuffix());
      }
      lance_close_dataset(dataset);
      LanceInvalidateDatasetCacheForTable(context, *this);
      return BuildUpdatedLanceTableEntry(context, *this, internal);
      break;
    }
    case AlterTableType::DROP_NOT_NULL: {
      auto &nn = info.Cast<DropNotNullInfo>();
      if (nn.column_path.size() != 1) {
        lance_close_dataset(dataset);
        throw NotImplementedException(
            "Lance ALTER COLUMN does not support nested field paths");
      }
      auto rc = lance_dataset_alter_columns_set_nullable(
          dataset, nn.column_path[0].c_str(), 1);
      if (rc != 0) {
        lance_close_dataset(dataset);
        throw IOException("Failed to drop NOT NULL in Lance dataset: " +
                          display_uri + LanceFormatErrorSuffix());
      }
      lance_close_dataset(dataset);
      LanceInvalidateDatasetCacheForTable(context, *this);
      return BuildUpdatedLanceTableEntry(context, *this, internal);
      break;
    }
    default:
      lance_close_dataset(dataset);
      throw NotImplementedException(
          "ALTER TABLE operation not supported for Lance tables");
    }
    break;
  }
  case AlterType::SET_COLUMN_COMMENT: {
    auto &comment = info.Cast<SetColumnCommentInfo>();
    const char *comment_ptr = nullptr;
    string comment_str;
    if (!comment.comment_value.IsNull()) {
      comment_str = comment.comment_value.DefaultCastAs(LogicalType::VARCHAR)
                        .GetValue<string>();
      comment_ptr = comment_str.c_str();
    }
    auto rc = lance_dataset_update_field_metadata(
        dataset, comment.column_name.c_str(), "comment", comment_ptr);
    if (rc != 0) {
      lance_close_dataset(dataset);
      throw IOException("Failed to update column comment in Lance dataset: " +
                        display_uri + LanceFormatErrorSuffix());
    }
    lance_close_dataset(dataset);
    LanceInvalidateDatasetCacheForTable(context, *this);
    return BuildUpdatedLanceTableEntry(context, *this, internal);
    break;
  }
  default:
    lance_close_dataset(dataset);
    throw NotImplementedException("ALTER is not supported for Lance tables");
  }

  return result;
}

unique_ptr<CatalogEntry> LanceTableEntry::Copy(ClientContext &context) const {
  (void)context;
  auto &catalog = this->catalog;
  auto &schema = this->schema;
  auto create_info = make_uniq<CreateTableInfo>(schema, name);
  create_info->temporary = temporary;
  create_info->internal = internal;
  create_info->comment = comment;
  create_info->tags = tags;
  create_info->columns = columns.Copy();
  for (auto &c : constraints) {
    create_info->constraints.push_back(c->Copy());
  }
  unique_ptr<LanceTableEntry> copy;
  if (IsNamespaceBacked()) {
    copy = make_uniq<LanceTableEntry>(catalog, schema, *create_info,
                                      NamespaceConfig());
  } else {
    copy =
        make_uniq<LanceTableEntry>(catalog, schema, *create_info, dataset_uri);
  }
  copy->SetCoercedColumnNames(coerced_column_names);
  copy->SetBlobColumns(blob_column_flags);
  copy->SetWritableBlobV2Columns(writable_blob_v2_columns);
  return unique_ptr_cast<LanceTableEntry, CatalogEntry>(std::move(copy));
}

bool LanceTableEntry::IsWritableBlobV2Column(
    const ColumnDefinition &column) const {
  if (column.Type().id() != LogicalTypeId::BLOB) {
    return false;
  }
  auto name = column.Name().GetIdentifierName();
  return std::find(writable_blob_v2_columns.begin(),
                   writable_blob_v2_columns.end(),
                   name) != writable_blob_v2_columns.end();
}

LogicalType LanceTableEntry::GetExpectedTypeForInsert(
    const ColumnDefinition &column) const {
  if (IsWritableBlobV2Column(column)) {
    return LanceBlobInputType();
  }
  return TableCatalogEntry::GetExpectedTypeForInsert(column);
}

unique_ptr<Expression> LanceTableEntry::GetDefaultExpressionForColumn(
    ClientContext &context, const LogicalType &input_type,
    const LogicalType &result_type, ColumnBinding binding,
    const Expression &constant_value) const {
  if (result_type.id() == LogicalTypeId::BLOB && LanceIsBlobInput(input_type)) {
    // The value is already the Blob write input; casting it back to the
    // declared BLOB read type would lose bytes/reference/ingest intent on
    // DEFAULT and omitted-column paths.
    return make_uniq<BoundColumnRefExpression>(input_type, binding);
  }
  return TableCatalogEntry::GetDefaultExpressionForColumn(
      context, input_type, result_type, binding, constant_value);
}

// Collect the logical column indexes carrying NOT NULL constraints. Both the
// entry side (constraints built from Arrow flags during discovery/rebuild)
// and the live side (Arrow flags of the coerced schema) are produced by the
// same population pipeline, so the resulting index sets are comparable.
static vector<idx_t> NotNullIndexesFromConstraints(
    const vector<unique_ptr<Constraint>> &constraints) {
  vector<idx_t> indexes;
  for (auto &constraint : constraints) {
    if (constraint->type == ConstraintType::NOT_NULL) {
      indexes.push_back(constraint->Cast<NotNullConstraint>().index.index);
    }
  }
  std::sort(indexes.begin(), indexes.end());
  return indexes;
}

// External writers can evolve the dataset schema behind this catalog entry
// (same-connection ALTERs rebuild the entry via AlterEntry, but external
// commits do not). DuckDB binds column ids against this entry's columns
// while scans and writers resolve state through the live dataset schema;
// serving a mismatched pair would mislabel columns, read the wrong fields,
// or apply stale write gates. Fail closed instead: replace the stale entry
// in the metadata cache, retaining the old generation for executing queries,
// so the next access serves the entry rebuilt from the current schema, and
// surface an explicit error for this statement.
//
// The comparison covers, beyond top-level names/types:
// - the coerced-column state: Arrow types that the reader-boundary layer
//   maps to the same DuckDB type (e.g. float16 and float32 both surface as
//   FLOAT) are indistinguishable by name/type, yet the entry's
//   coerced-column list gates writes (INSERT/UPDATE/MERGE reject coerced
//   columns);
// - the nullability state: external ALTER SET/DROP NOT NULL changes only
//   the Arrow flags, which the entry mirrors as NOT NULL constraints;
// - stored defaults, which native INSERT/UPDATE binding reads from this entry.
// All compared states come from the same population pipeline over the
// (entry-build vs live) schema, so the comparison is exact.
bool LanceTableEntry::MatchesLiveSchemaState(
    const LanceTableSchema &live) const {
  if (columns.LogicalColumnCount() != live.columns.LogicalColumnCount() ||
      CoercedColumnNames() != live.coerced_columns ||
      writable_blob_v2_columns != live.writable_blob_v2_columns ||
      NotNullIndexesFromConstraints(constraints) !=
          NotNullIndexesFromConstraints(live.constraints)) {
    return false;
  }
  for (idx_t i = 0; i < columns.LogicalColumnCount(); i++) {
    auto &column = columns.GetColumn(LogicalIndex(i));
    auto &live_column = live.columns.GetColumn(LogicalIndex(i));
    if (column.Name().GetIdentifierName() !=
            live_column.Name().GetIdentifierName() ||
        column.Type() != live_column.Type() ||
        column.HasDefaultValue() != live_column.HasDefaultValue() ||
        (column.HasDefaultValue() &&
         !column.DefaultValue().Equals(live_column.DefaultValue()))) {
      return false;
    }
  }
  return true;
}

void LanceTableEntry::ValidateLiveSchemaOrReplace(ClientContext &context,
                                                  const LanceTableSchema &live,
                                                  const string &display_uri) {
  if (MatchesLiveSchemaState(live)) {
    return;
  }
  // The query pins this generation until QueryEnd, so reading
  // members of `this` for the error message below remains safe. The heal is
  // best-effort (fail-open); the error is thrown regardless so this
  // statement never runs against a mismatched binding.
  (void)LanceTryReplaceStaleTableEntry(context, *this);
  throw CatalogException(
      "Lance table \"%s\" was changed externally: the dataset schema at "
      "'%s' no longer matches the catalog entry. The entry has been "
      "refreshed from the current schema - please re-run the query.",
      name, display_uri);
}

LanceTableSchema LanceTableEntry::FetchLiveSchema(ClientContext &context,
                                                  string &out_display_uri) {
  auto entry =
      LanceGetOrOpenDatasetEntryForTable(context, *this, out_display_uri);
  auto *dataset = entry ? entry->Handle() : nullptr;
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + out_display_uri +
                      LanceFormatErrorSuffix());
  }
  return LanceTableSchema::FromDataset(context, dataset);
}

bool LanceTableEntry::IsSchemaStale(ClientContext &context) {
  string display_uri;
  return !MatchesLiveSchemaState(FetchLiveSchema(context, display_uri));
}

void LanceTableEntry::VerifySchemaFreshness(ClientContext &context) {
  string display_uri;
  auto live = FetchLiveSchema(context, display_uri);
  ValidateLiveSchemaOrReplace(context, live, display_uri);
}

} // namespace duckdb
