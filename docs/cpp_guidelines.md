# DuckDB C++ Implementation Guidelines

Unless DuckDB/Lance requirements dictate otherwise, follow the defaults below. These are guidelines, not immutable constraints.

## C++ Guidelines

- Do not use `malloc`, prefer the use of smart pointers. Keywords `new` and `delete` are a code smell.
- Strongly prefer the use of `unique_ptr` over `shared_ptr`, only use `shared_ptr` if you absolutely have to.
- Use `const` whenever possible.
- Do not import namespaces (e.g. `using std`).
- All functions in source files in the core (`src` directory) should be part of the `duckdb` namespace.
- When overriding a virtual method, avoid repeating `virtual` and always use `override` or `final`.
- Use `[u]int8_t`/`[u]int16_t`/`[u]int32_t`/`[u]int64_t` instead of `int`, `long`, `uint` etc. Use `idx_t` instead of `size_t` for offsets/indices/counts of any kind.
- Prefer using references over pointers as arguments.
- Use `const` references for arguments of non-trivial objects (e.g. `std::vector`, ...).
- Use C++11 for loops when possible: `for (const auto& item : items) {...}`.
- Use braces for `if` statements and loops. Avoid single-line `if` statements and loops, especially nested ones.
- Class layout should follow this structure:

```cpp
class MyClass {
public:
	MyClass();

	int my_public_variable;

public:
	void MyFunction();

private:
	void MyPrivateFunction();

private:
	int my_private_variable;
};
```

- Avoid unnamed magic numbers. Use named variables stored in a `constexpr`.
- Return early and avoid deep nested branches.
- Do not include commented-out code blocks in pull requests.

## Error Handling

- Use exceptions only when an error terminates a query (e.g. parser error, table not found).
- For expected/non-fatal errors, prefer return values over exceptions.
- Try to add test cases that trigger exceptions. If an exception cannot be easily triggered by tests, it may be an assertion instead.
- Use `D_ASSERT` for debug-only assertions (stripped in release builds). Use `assert` only for invariants that must also be checked in release.
- Assertions should never be triggerable by user input.
- Avoid assertions without context like `D_ASSERT(a > b + 3);` unless accompanied by clear context/comments.
- Assert liberally, and make clear (with concise comments when needed) what invariant failed.

## Naming Conventions

- Choose descriptive names. Avoid single-letter variable names.
- Files: lowercase with underscores, e.g. `abstract_operator.cpp`.
- Types (classes, structs, enums, typedefs, `using`): `CamelCase` starting with uppercase, e.g. `BaseColumn`.
- Variables: lowercase with underscores, e.g. `chunk_size`.
- Functions: `CamelCase` starting with uppercase, e.g. `GetChunk`.
- Avoid `i`, `j`, etc. in nested loops. Prefer names like `column_idx`, `check_idx`.
- In non-nested loops, `i` is permissible as iterator index.
- These rules are partially enforced by `clang-tidy`.
