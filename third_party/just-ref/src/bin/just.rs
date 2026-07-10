//! CLI wrapper for the just JavaScript engine.
//!
//! Usage:
//!   just <file.js>              # Execute a JavaScript file
//!   just -e "code"              # Evaluate JavaScript code
//!   just                        # Start REPL (interactive mode)

use just_engine::parser::JsParser;
use just_engine::runner::plugin::registry::BuiltInRegistry;
use just_engine::runner::plugin::types::EvalContext;
use just_engine::runner::eval::statement::execute_statement;
use just_engine::runner::ds::value::JsValue;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    match args.len() {
        1 => {
            // No arguments: start REPL
            run_repl();
        }
        2 => {
            let arg = &args[1];
            // Check for help flags
            if arg == "-h" || arg == "--help" {
                print_usage();
                process::exit(0);
            }
            // Single argument: execute file
            run_file(arg);
        }
        3 if args[1] == "-e" || args[1] == "--eval" => {
            // -e flag: evaluate expression
            let code = &args[2];
            eval_code(code);
        }
        _ => {
            print_usage();
            process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("just - JavaScript Engine");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  just <file.js>              Execute a JavaScript file");
    eprintln!("  just -e \"code\"              Evaluate JavaScript code");
    eprintln!("  just --eval \"code\"          Evaluate JavaScript code");
    eprintln!("  just                        Start REPL (interactive mode)");
}

fn run_file(filename: &str) {
    let source = match fs::read_to_string(filename) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", filename, e);
            process::exit(1);
        }
    };

    let ast = match JsParser::parse_to_ast_from_str(&source) {
        Ok(program) => program,
        Err(e) => {
            eprintln!("Parse error: {:?}", e);
            process::exit(1);
        }
    };

    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());

    for stmt in &ast.body {
        match execute_statement(stmt, &mut ctx) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Runtime error: {:?}", e);
                process::exit(1);
            }
        }
    }
}

fn eval_code(code: &str) {
    let ast = match JsParser::parse_to_ast_from_str(code) {
        Ok(program) => program,
        Err(e) => {
            eprintln!("Parse error: {:?}", e);
            process::exit(1);
        }
    };

    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());

    let mut last_value: Option<JsValue> = None;
    for stmt in &ast.body {
        match execute_statement(stmt, &mut ctx) {
            Ok(completion) => {
                last_value = completion.value;
            }
            Err(e) => {
                eprintln!("Runtime error: {:?}", e);
                process::exit(1);
            }
        }
    }

    // Print the last value if it's not undefined
    if let Some(val) = last_value {
        if !matches!(val, JsValue::Undefined) {
            println!("{:?}", val);
        }
    }
}

fn run_repl() {
    println!("just v0.1.0 - JavaScript Engine");
    println!("Type JavaScript code and press Enter. Type .exit to quit.");
    println!();

    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush().unwrap();

        let mut input = String::new();
        match stdin.read_line(&mut input) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }

        let input = input.trim();

        // Handle special commands
        if input == ".exit" || input == ".quit" {
            break;
        }

        if input.is_empty() {
            continue;
        }

        // Try to parse and execute
        let ast = match JsParser::parse_to_ast_from_str(input) {
            Ok(program) => program,
            Err(e) => {
                eprintln!("Parse error: {:?}", e);
                continue;
            }
        };

        for stmt in &ast.body {
            match execute_statement(stmt, &mut ctx) {
                Ok(completion) => {
                    // Print non-undefined values
                    if let Some(val) = completion.value {
                        if !matches!(val, JsValue::Undefined) {
                            println!("{:?}", val);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Runtime error: {:?}", e);
                }
            }
        }
    }

    println!("Goodbye!");
}
