#pragma once

#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"

namespace duckdb {

// Internal table functions behind the version administration statements:
//   ALTER TABLE <t> CREATE TAG <name> [AT VERSION <n>]
//   ALTER TABLE <t> DROP TAG <name>
//   ALTER TABLE <t> RESTORE TO VERSION <n> | TAG <name>
// Parameters: (target, tag, version) for CREATE TAG and RESTORE, (target, tag)
// for DROP TAG. A NULL version means "latest" for CREATE TAG; RESTORE takes
// exactly one of version or tag.
TableFunction LanceInternalCreateTagTableFunction();
TableFunction LanceInternalDeleteTagTableFunction();
TableFunction LanceInternalRestoreTableFunction();

} // namespace duckdb
