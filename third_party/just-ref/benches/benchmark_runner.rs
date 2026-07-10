/// Benchmark runner for the JavaScript interpreter.
///
/// Compares performance against Node.js for reference.

extern crate just_engine;

use just_engine::parser::JsParser;
use just_engine::runner::eval::statement::execute_statement;
use just_engine::runner::jit;
use just_engine::runner::plugin::registry::BuiltInRegistry;
use just_engine::runner::plugin::types::EvalContext;
use just_engine::runner::ds::value::{JsValue, JsNumberType};
use std::time::{Duration, Instant};

/// Run a benchmark and return the execution time.
fn run_benchmark(name: &str, code: &str, iterations: u32) -> Duration {
    let ast = JsParser::parse_to_ast_from_str(code)
        .expect(&format!("Failed to parse benchmark: {}", name));

    let start = Instant::now();

    for _ in 0..iterations {
        let mut ctx = EvalContext::new();
        for stmt in &ast.body {
            let _ = execute_statement(stmt, &mut ctx);
        }
    }

    start.elapsed()
}

/// Run a JIT benchmark and return the execution time.
fn run_benchmark_jit(name: &str, code: &str, iterations: u32) -> Duration {
    let ast = JsParser::parse_to_ast_from_str(code)
        .expect(&format!("Failed to parse benchmark: {}", name));

    let chunk = jit::compile(&ast);

    let start = Instant::now();

    for _ in 0..iterations {
        let mut ctx = EvalContext::new();
        ctx.install_core_builtins(BuiltInRegistry::with_core());
        let _ = jit::execute(&chunk, ctx);
    }

    start.elapsed()
}

/// Get the result of running code.
fn run_and_get_var(code: &str, var_name: &str) -> JsValue {
    let ast = JsParser::parse_to_ast_from_str(code).unwrap();
    let mut ctx = EvalContext::new();
    for stmt in &ast.body {
        let _ = execute_statement(stmt, &mut ctx);
    }
    ctx.get_binding(var_name).unwrap_or(JsValue::Undefined)
}

/// Get the result of running code via JIT.
fn run_and_get_var_jit(code: &str, var_name: &str) -> JsValue {
    let ast = JsParser::parse_to_ast_from_str(code).unwrap();
    let (_, mut ctx) = jit::compile_and_run_with_ctx(&ast);
    ctx.get_binding(var_name).unwrap_or(JsValue::Undefined)
}

// ============================================================================
// Benchmark definitions
// ============================================================================

const BENCH_FIBONACCI: &str = r#"
var n = 20;
var a = 0;
var b = 1;
for (var i = 0; i < n; i = i + 1) {
    var temp = a;
    a = b;
    b = temp + b;
}
"#;

const BENCH_LOOP_SUM: &str = r#"
var sum = 0;
for (var i = 0; i < 10000; i = i + 1) {
    sum = sum + i;
}
"#;

const BENCH_NESTED_LOOPS: &str = r#"
var count = 0;
for (var i = 0; i < 100; i = i + 1) {
    for (var j = 0; j < 100; j = j + 1) {
        count = count + 1;
    }
}
"#;

const BENCH_BITWISE: &str = r#"
var result = 0;
for (var i = 0; i < 1000; i = i + 1) {
    result = (result ^ i) & 0xFFFF;
}
"#;

const BENCH_CONDITIONALS: &str = r#"
var count = 0;
for (var i = 0; i < 1000; i = i + 1) {
    if (i % 2 === 0) {
        count = count + 1;
    } else {
        count = count + 2;
    }
}
"#;

const BENCH_WHILE_LOOP: &str = r#"
var i = 0;
var sum = 0;
while (i < 5000) {
    sum = sum + i;
    i = i + 1;
}
"#;

const BENCH_ARITHMETIC: &str = r#"
var result = 0;
for (var i = 1; i < 1000; i = i + 1) {
    result = result + i * 2 - i / 2;
}
"#;

const BENCH_FACTORIAL: &str = r#"
var n = 12;
var result = 1;
for (var i = 2; i <= n; i = i + 1) {
    result = result * i;
}
"#;

const BENCH_PRIME_SIEVE: &str = r#"
var count = 0;
for (var n = 2; n < 100; n = n + 1) {
    var isPrime = true;
    for (var i = 2; i * i <= n; i = i + 1) {
        if (n % i === 0) {
            isPrime = false;
            break;
        }
    }
    if (isPrime) {
        count = count + 1;
    }
}
"#;

const BENCH_GCD: &str = r#"
var result = 0;
for (var k = 0; k < 100; k = k + 1) {
    var a = 48;
    var b = 18;
    while (b !== 0) {
        var temp = b;
        b = a % b;
        a = temp;
    }
    result = result + a;
}
"#;

fn main() {
    println!("=======================================================");
    println!("  Just JavaScript Engine - Performance Benchmarks");
    println!("  Tree-Walking Interpreter vs Bytecode JIT");
    println!("=======================================================\n");

    let benchmarks: Vec<(&str, &str, u32)> = vec![
        ("Fibonacci (n=20)", BENCH_FIBONACCI, 1000),
        ("Loop Sum (10K iterations)", BENCH_LOOP_SUM, 100),
        ("Nested Loops (100x100)", BENCH_NESTED_LOOPS, 100),
        ("Bitwise Operations (1K)", BENCH_BITWISE, 500),
        ("Conditionals (1K)", BENCH_CONDITIONALS, 500),
        ("While Loop (5K)", BENCH_WHILE_LOOP, 100),
        ("Arithmetic (1K)", BENCH_ARITHMETIC, 500),
        ("Factorial (n=12)", BENCH_FACTORIAL, 5000),
        ("Prime Sieve (<100)", BENCH_PRIME_SIEVE, 200),
        ("GCD (100 iterations)", BENCH_GCD, 200),
    ];

    println!("{:<30} {:>14} {:>14} {:>10}", "Benchmark", "Interpreter", "JIT", "Speedup");
    println!("{}", "-".repeat(70));

    let mut total_interp = Duration::ZERO;
    let mut total_jit = Duration::ZERO;

    for (name, code, iterations) in &benchmarks {
        let interp_dur = run_benchmark(name, code, *iterations);
        let jit_dur = run_benchmark_jit(name, code, *iterations);
        total_interp += interp_dur;
        total_jit += jit_dur;

        let speedup = interp_dur.as_secs_f64() / jit_dur.as_secs_f64();

        println!(
            "{:<30} {:>12.2?} {:>12.2?} {:>9.2}x",
            name, interp_dur, jit_dur, speedup
        );
    }

    println!("{}", "-".repeat(70));
    let total_speedup = total_interp.as_secs_f64() / total_jit.as_secs_f64();
    println!(
        "{:<30} {:>12.2?} {:>12.2?} {:>9.2}x",
        "TOTAL", total_interp, total_jit, total_speedup
    );

    // Verify correctness
    println!("\n=======================================================");
    println!("  Correctness Verification");
    println!("=======================================================\n");

    let verifications: Vec<(&str, &str, &str, i64)> = vec![
        ("Fibonacci", BENCH_FIBONACCI, "a", 6765),
        ("Loop Sum", BENCH_LOOP_SUM, "sum", 49995000),
        ("Nested Loops", BENCH_NESTED_LOOPS, "count", 10000),
        ("Factorial", BENCH_FACTORIAL, "result", 479001600),
        ("Prime Count", BENCH_PRIME_SIEVE, "count", 25),
    ];

    println!("{:<20} {:>12} {:>12} {:>12}", "Test", "Expected", "Interp", "JIT");
    println!("{}", "-".repeat(58));

    for (name, code, var, expected) in verifications {
        let interp_result = run_and_get_var(code, var);
        let jit_result = run_and_get_var_jit(code, var);

        let interp_val = match interp_result {
            JsValue::Number(JsNumberType::Integer(n)) => n,
            _ => -1,
        };
        let jit_val = match jit_result {
            JsValue::Number(JsNumberType::Integer(n)) => n,
            _ => -1,
        };

        let i_status = if interp_val == expected { "✓" } else { "✗" };
        let j_status = if jit_val == expected { "✓" } else { "✗" };
        println!(
            "{:<20} {:>12} {:>4} {:>7} {:>4} {:>7}",
            name, expected, i_status, interp_val, j_status, jit_val
        );
    }

    println!("\n=======================================================");
    println!("  Reference: Node.js Equivalent Commands");
    println!("=======================================================\n");

    println!("To compare with Node.js, run these commands:\n");
    println!("# Fibonacci benchmark");
    println!("node -e \"let start=Date.now(); for(let k=0;k<1000;k++){{let a=0,b=1;for(let i=0;i<20;i++){{let t=a;a=b;b=t+b}}}}; console.log(Date.now()-start+'ms')\"\n");
    println!("# Loop Sum benchmark");
    println!("node -e \"let start=Date.now(); for(let k=0;k<100;k++){{let sum=0;for(let i=0;i<10000;i++)sum+=i}}; console.log(Date.now()-start+'ms')\"\n");
}
