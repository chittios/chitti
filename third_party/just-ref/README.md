<img src="logo/just-logo-full.png" alt="logo image" />

# just - JS on Rust

A ground-up implementation of an ES6 JavaScript engine written in Rust, featuring a PEG parser, a tree-walking interpreter, two bytecode VMs (stack-based and register-based), and a Cranelift-powered native JIT compiler.

This is an academic/experimental project rather than a production-ready engine.

[![Video Overview](https://img.youtube.com/vi/tbqD0wiAeDA/maxresdefault.jpg)](https://youtu.be/tbqD0wiAeDA)

## Overview

- **Parser** — An ES6 grammar coded in [Pest](https://pest.rs/) PEG, following the [ECMAScript 2015 specification](https://262.ecma-international.org/6.0/). The AST conforms to the [ESTree](https://github.com/estree/estree) specification (visualize similar trees at [astexplorer.net](https://astexplorer.net/)).
- **Three execution backends** — A tree-walking interpreter, a bytecode compiler + VM (in both stack and register flavours), and a Cranelift native-code JIT for numeric-heavy paths.
- **Plugin architecture** — Built-in and plugin-provided objects (`Math`, `JSON`, `String`, …) are exposed via a **super-global scope** that resolves objects lazily at runtime through pluggable resolvers.

## Building & Testing

```bash
# Build
cargo build

# Run all tests (451 tests across 8 suites)
cargo test

# Run specific test suites
cargo test --package just --lib parser::unit_tests   # Parser unit tests
cargo test --test test_integration                    # Interpreter integration tests
cargo test --test test_jit                            # Stack-based VM + JIT tests
cargo test --test test_reg_jit                        # Register VM + Cranelift JIT tests
cargo test --test test_std_lib                        # Standard library tests

# Run benchmarks (Interpreter vs JIT comparison)
cargo bench

# Or run the benchmark binary directly
cargo run --release --bin benchmark
```

## CLI Usage

The `just` binary provides a command-line interface for executing JavaScript:

```bash
# Build the CLI
cargo build --release --bin just

# Execute a JavaScript file
./target/release/just script.js

# Evaluate JavaScript code directly
./target/release/just -e "var x = 5 + 3; x"
# Output: JsValue::Number(Integer(8))

./target/release/just --eval "function factorial(n) { if (n <= 1) return 1; return n * factorial(n - 1); } factorial(5)"
# Output: JsValue::Number(Integer(120))

# Start interactive REPL
./target/release/just
# > var x = 10;
# > var y = 20;
# > x + y
# JsValue::Number(Integer(30))
# > .exit
```

**REPL Commands:**
- `.exit` or `.quit` — Exit the REPL
- `Ctrl+D` — Exit the REPL (EOF)

**Built-in Support:**
The CLI uses the tree-walking interpreter with full super-global scope integration. Built-in objects like `Math`, `console`, `JSON`, etc. are resolved lazily through the super-global scope:

```bash
./target/release/just -e "console.log('Hello!'); Math.abs(-42)"
# Output: Hello!
#         JsValue::Number(Integer(42))
```

## Architecture

```
src/
├── lib.rs                          # Crate root: exposes parser + runner modules
├── parser/
│   ├── js_grammar.pest             # Pest PEG grammar (ES6 spec)
│   ├── api.rs                      # Parser API: source → Pest pairs → AST
│   ├── ast.rs                      # AST node types (ESTree-compliant)
│   ├── static_semantics.rs         # Static analysis (bound names, scoping flags)
│   └── util.rs                     # Formatting helpers
└── runner/
    ├── eval/                       # Tree-walking interpreter
    │   ├── expression.rs           # Expression evaluation (operators, calls, objects)
    │   ├── statement.rs            # Statement execution (loops, control flow, scoping)
    │   ├── function.rs             # Function call mechanics
    │   └── types.rs                # Completion records, references
    ├── jit/                        # Bytecode compilation & VMs
    │   ├── mod.rs                  # Public API: compile → execute orchestration
    │   ├── compiler.rs             # AST → stack-based bytecode compiler
    │   ├── bytecode.rs             # Stack-based opcode definitions + Chunk
    │   ├── vm.rs                   # Stack-based bytecode VM
    │   ├── reg_compiler.rs         # AST → 3-address register bytecode compiler
    │   ├── reg_bytecode.rs         # Register opcode definitions + RegChunk
    │   ├── reg_vm.rs               # Register-based bytecode VM
    │   └── reg_jit.rs              # Cranelift native JIT (numeric fast-path)
    ├── ds/                         # Runtime data structures (ES6 spec types)
    │   ├── value.rs                # JsValue, JsNumberType
    │   ├── object.rs               # Object model (ordinary + function objects)
    │   ├── object_property.rs      # Property descriptors, PropertyKey
    │   ├── function_object.rs      # Function object internals
    │   ├── env_record.rs           # Environment records (declarative, function, global)
    │   ├── lex_env.rs              # Lexical environment chain
    │   ├── execution_context.rs    # Execution context stack
    │   ├── realm.rs                # Code realm + well-known intrinsics
    │   ├── error.rs                # JErrorType (TypeError, ReferenceError, …)
    │   ├── heap.rs                 # Heap allocation tracking with optional limits
    │   ├── symbol.rs               # Symbol primitives
    │   └── operations/             # Abstract operations (type conversion, comparison)
    ├── plugin/                     # Plugin architecture
    │   ├── types.rs                # EvalContext, super-global integration, BuiltInFn, BuiltInObject
    │   ├── resolver.rs             # PluginResolver trait (lazy dynamic object resolution)
    │   ├── super_global.rs         # SuperGlobalEnvironment (resolver chain + caching)
    │   ├── core_resolver.rs        # CorePluginResolver (BuiltInRegistry adapter)
    │   ├── registry.rs             # BuiltInRegistry: built-in object/method definitions + plugin loading
    │   └── config.rs               # Plugin configuration file parsing
    └── std_lib/                    # Built-in object implementations
        ├── core.rs                 # Registers all built-ins into the registry
        ├── console.rs              # console.log/error/warn/info
        ├── math.rs                 # Math object (constants + 35 methods)
        ├── json.rs                 # JSON.parse/stringify
        ├── number.rs               # Number constants + methods
        ├── string.rs               # String.prototype methods
        ├── array.rs                # Array.prototype methods
        ├── object.rs               # Object static methods
        └── error.rs                # Error type constructors

tests/
├── test_eval.rs                    # Tree-walking interpreter tests
├── test_integration.rs             # End-to-end interpreter integration tests
├── test_jit.rs                     # Stack-based VM + JIT tests
├── test_reg_jit.rs                 # Register VM + Cranelift JIT tests
└── test_std_lib.rs                 # Standard library built-in tests

benches/
└── benchmark_runner.rs             # Interpreter vs JIT performance comparison
```

### Execution Pipelines

The engine provides three execution paths, selectable at the API level:

```
                                ┌──────────────────────────────────────┐
                                │          JavaScript Source            │
                                └──────────────────┬───────────────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │   Pest Parser    │
                                          │  (js_grammar.pest)│
                                          └────────┬────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │    AST (ESTree)  │
                                          └──┬─────┬─────┬──┘
                                             │     │     │
                        ┌────────────────────┘     │     └────────────────────┐
                        │                          │                          │
               ┌────────▼────────┐       ┌────────▼────────┐       ┌────────▼────────┐
               │  Tree-Walking   │       │  Stack Compiler  │       │  Reg Compiler   │
               │  Interpreter    │       │  (compiler.rs)   │       │ (reg_compiler.rs)│
               │  (eval/)        │       └────────┬────────┘       └───┬─────────┬───┘
               └────────┬────────┘                │                    │         │
                        │               ┌────────▼────────┐    ┌─────▼───┐ ┌───▼──────┐
                        │               │  Stack-based VM  │    │ Reg VM  │ │Cranelift │
                        │               │  (vm.rs)         │    │(reg_vm) │ │  JIT     │
                        │               └────────┬────────┘    └────┬────┘ │(reg_jit) │
                        │                        │                  │      └───┬──────┘
                        ▼                        ▼                  ▼          ▼
                                          JsValue Result
```

**1. Tree-Walking Interpreter** (`runner::eval`)
- Directly walks the AST, evaluating each node recursively.
- Full JS feature support: objects, functions, closures, `try`/`catch`, generators, `for-in`/`for-of`.
- Uses `EvalContext` with lexical environment chains for variable resolution.

**2. Stack-Based Bytecode VM** (`runner::jit::compiler` → `runner::jit::vm`)
- Single-pass AST compiler emits flat stack-based bytecode (`OpCode` instructions with a `Chunk`).
- The VM uses an operand stack and inline caches for variable/property lookups.
- Built-in/plugin objects are resolved through the `EvalContext` **super-global scope** (lazy resolution via plugin resolvers) rather than being preloaded into the JS global scope.
- Supports user-defined function declarations/expressions, calls, and closures, and method calls (`CallMethod`) including built-ins like `Math.abs`.

**3. Register-Based Bytecode VM + Cranelift JIT** (`runner::jit::reg_compiler` → `runner::jit::reg_vm` / `runner::jit::reg_jit`)
- Single-pass AST compiler emits 3-address register bytecode (`RegOpCode` instructions with `dst`, `src1`, `src2`, `imm` fields).
- **Register VM** (`reg_vm.rs`): interprets register bytecode with inline caches. Full JS value support.
- **Cranelift JIT** (`reg_jit.rs`): compiles register bytecode to native x86_64 machine code via [Cranelift](https://cranelift.dev/). Operates on `f64` registers for numeric-heavy code. Pre-scans bytecode and bails out to the register VM for unsupported operations (object property access, function calls, `typeof`, etc.).
- The `execute_reg_jit_or_vm` function implements the tiered strategy: try JIT first, fall back to register VM on bail.

### Key Design Decisions

- **JsValue** — A tagged enum (`Undefined | Null | Boolean | String | Symbol | Number | Object`) where `Number` is further split into `Integer(i64) | Float(f64) | NaN | PositiveInfinity | NegativeInfinity` for precise spec-compliant arithmetic.
- **Environment Records** — ES6-compliant declarative, function, and global environment records with lexical environment chains (`Rc<RefCell<LexEnvironment>>`).
- **Completion Records** — Statement evaluation returns `Completion { type, value, target }` to propagate `return`, `break`, `continue`, `throw`, and `yield` through the call stack.
- **Inline Caches** — Both VMs cache variable lookups (`EnvCacheEntry` keyed by environment version) and property accesses to avoid repeated scope-chain walks.
- **Super-Global Scope (lazy built-ins/plugins)** — Name lookup walks lexical scopes → global → **super-global**. The super-global is read-only to JS code and resolves names dynamically by querying a chain of `PluginResolver`s in registration order. Resolved values are cached, so plugins are invoked only on first use.
- **Plugin Registry** — `BuiltInRegistry` still defines built-in objects as `BuiltInObject` entries containing `HashMap<String, BuiltInFn>` method maps, but it is consumed via a resolver adapter (`CorePluginResolver`) rather than being directly consulted by the stack VM.
- **Heap Tracking** — An optional `Heap` with configurable memory limits tracks allocations for resource-constrained environments.

### Super-Global Scope (Lazy Built-ins & Plugins)

The runtime models a special scope *below* the JS global environment called the **super-global**:

- **Lookup order**
  - **lexical scopes** (block/function)
  - **global scope**
  - **super-global scope** (dynamic; queried on demand)
- **Read-only to JS**
  - JS code cannot create bindings in super-global.
  - It can still shadow super-global names by declaring local/global variables with the same name.
- **Lazy resolution**
  - When a name reaches super-global, the runtime asks each registered `PluginResolver`:
    - `has_binding(name)` (cheap probe)
    - `resolve(name, ctx)` (materialize value; cached after first resolution)

The intended implication is that hundreds of potential APIs do **not** need to be preloaded at startup; plugins are only consulted when code actually references their objects.

## Performance

Benchmarks comparing the tree-walking interpreter against the stack-based bytecode VM (run with `cargo bench`):

| Benchmark | Interpreter | JIT | Speedup |
|---|---:|---:|---:|
| Fibonacci (n=20) | 22.59ms | 3.32ms | **6.80x** |
| Loop Sum (10K iter) | 614.53ms | 114.28ms | **5.38x** |
| Nested Loops (100×100) | 566.00ms | 116.08ms | **4.88x** |
| Bitwise Ops (1K) | 320.74ms | 62.60ms | **5.12x** |
| Conditionals (1K) | 368.58ms | 121.85ms | **3.02x** |
| While Loop (5K) | 311.30ms | 56.96ms | **5.47x** |
| Arithmetic (1K) | 368.66ms | 81.72ms | **4.51x** |
| Factorial (n=12) | 44.81ms | 8.79ms | **5.10x** |
| Prime Sieve (<100) | 49.34ms | 12.84ms | **3.84x** |
| GCD (100 iter) | 66.27ms | 11.08ms | **5.98x** |
| **Total** | **2.73s** | **589.51ms** | **4.64x** |

## Project Status

### Parser (Mostly Complete)

The parser supports most ES6 syntax:

**Literals** — Strings (escape sequences, unicode), numbers (decimal, hex `0x`, binary `0b`, octal `0o`, floats, scientific notation), booleans, null, regular expressions, template literals.

**Expressions** — Identifiers, object/array literals, function/arrow/class/generator expressions, member expressions (dot and bracket), call expressions, unary/binary/logical/conditional/sequence expressions, assignment (including compound `+=`, `-=`, etc.), update (`++`, `--`), spread `...`, destructuring patterns, `super`, `new.target`.

**Statements** — `var`/`let`/`const`, blocks, `if`/`else`, `while`, `do-while`, `for`, `for-in`, `for-of`, `switch`/`case`, `try`/`catch`/`finally`, `throw`, `return`, `break`, `continue`, function/class/generator declarations.

**Not Supported** — Async/await (ES2017), labeled statements.

### Interpreter & VMs

**Interpreter (tree-walking)** — All literal types, unary/binary/bitwise/logical/comparison operators, `typeof`, conditional and sequence expressions, update expressions, assignment (including destructuring with defaults/rest), type coercion, `var`/`let`/`const` with proper scoping, object/array creation, property access (dot, bracket, computed), getters/setters, `new` expressions, classes + inheritance, `delete`, `in`/`instanceof`, spread in calls/arrays, function declarations and calls, closures, `if`/`else`, `while`, `do-while`, `for`, `for-in`, `for-of`, `switch`/`case` with fall-through, `break`/`continue`, `try`/`catch`/`finally`, `throw`, generators and `yield`.

**Bytecode VMs (stack + register)** — Core arithmetic, control flow, variables, and property access are present. The **stack VM** supports user-defined function objects + calls + closures and method calls (including built-in/plugin dispatch via the super-global). Several ES6 features are still not yet compiled/executed in bytecode, including: `new` expressions, classes/inheritance, `delete`, `in`/`instanceof`, getters/setters, spread in calls/arrays, and destructuring assignment.

**Not Yet Implemented (global)** — `eval()`.

### Built-in Objects

| Object | Coverage |
|---|---|
| **console** | `log`, `error`, `warn`, `info` |
| **Math** | All ES6 constants + 35 methods |
| **JSON** | `parse`, `stringify` |
| **Number** | All constants + `isNaN`, `isFinite`, `isInteger`, `isSafeInteger`, `parseFloat`, `parseInt`, `toString`, `toFixed`, `toExponential`, `toPrecision` |
| **String** | `charAt`, `charCodeAt`, `substring`, `slice`, `indexOf`, `lastIndexOf`, `includes`, `startsWith`, `endsWith`, `split`, `trim`, `trimStart`, `trimEnd`, `toUpperCase`, `toLowerCase`, `repeat`, `padStart`, `padEnd`, `replace`, `concat`, `fromCharCode` |
| **Array** | `push`, `pop`, `shift`, `unshift`, `splice`, `reverse`, `sort`, `slice`, `concat`, `indexOf`, `includes`, `join`, `forEach`, `map`, `filter`, `reduce`, `find`, `every`, `some`, `isArray` |
| **Object** | `toString`, `valueOf`, `hasOwnProperty`, `keys`, `values`, `entries`, `assign` |
| **Error types** | `Error`, `TypeError`, `ReferenceError`, `SyntaxError`, `RangeError`, `EvalError`, `URIError` |

### Test Suite

| Suite | Tests | Description |
|---|---:|---|
| Parser unit tests | 70 | Grammar and AST construction |
| Integration tests | 118 | End-to-end interpreter scenarios |
| Interpreter super-global | 21 | Built-in resolution, constructors & custom plugins |
| JIT tests | 52 | Stack-based VM + bytecode compiler |
| Register JIT tests | 41 | Register VM + Cranelift JIT |
| Standard library | 76 | Built-in object methods |
| Eval tests | 73 | Expression and statement evaluation |
| **Total** | **451** | |

## Dependencies

| Crate | Purpose |
|---|---|
| `pest` / `pest_derive` | PEG parser generator for the ES6 grammar |
| `cranelift` / `cranelift-jit` / `cranelift-module` / `cranelift-native` | Native code generation for the JIT backend |
| `lazy_static` | Lazy-initialized global constants (symbols, property keys) |
| `uuid` | Unique identifiers for internal object tracking |

## License

This project is licensed under the MIT License.
