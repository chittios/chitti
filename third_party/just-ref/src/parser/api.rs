// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::parser::ast::StatementType::BreakStatement;
use crate::parser::ast::{
    AssignmentOperator, AssignmentPropertyData, AstBuilderValidationErrorType, BinaryOperator,
    BlockStatementData, CatchClauseData, ClassBodyData, ClassData, ClassFieldData, StaticBlockData, DeclarationType,
    ExpressionOrSpreadElement, ExpressionOrSuper, ExpressionPatternType, ExpressionType,
    ExtendedNumberLiteralType, FormalParameters, ForIteratorData, FunctionBodyData,
    FunctionBodyOrExpression, FunctionData, HasMeta, IdentifierData, JsError, JsErrorType,
    LiteralData, LiteralType, LogicalOperator, MemberExpressionType, Meta, MethodDefinitionData,
    MethodDefinitionKind, NumberLiteralType, PatternOrExpression, PatternType, ProgramData,
    PropertyData, PropertyKind, StatementType, SwitchCaseData, TemplateElementData,
    TemplateLiteralData, UnaryOperator, UpdateOperator, VariableDeclarationData,
    VariableDeclarationKind, VariableDeclarationOrExpression, VariableDeclarationOrPattern,
    VariableDeclaratorData,
};
use crate::parser::static_semantics::Semantics;
use crate::parser::util::TAB_WIDTH;
use pest::iterators::{Pair, Pairs};
use pest::Parser;
use pest_derive::Parser;
use core::borrow::Borrow;
use hashbrown::HashMap;
use alloc::rc::Rc;
#[cfg(feature = "std")]
use std::time::Instant;

/// JavaScript parser using PEG grammar.
///
/// Parses ES6 JavaScript source code into an ESTree-compliant AST.
/// The parser is built on [Pest](https://pest.rs/) and follows the
/// [ECMAScript 2015 specification](https://262.ecma-international.org/6.0/).
///
/// # Examples
///
/// ```
/// use just::parser::JsParser;
///
/// let code = "var x = 5 + 3;";
/// let ast = JsParser::parse_to_ast_from_str(code).unwrap();
/// assert_eq!(ast.body.len(), 1);
/// ```
#[derive(Parser)]
#[grammar = "parser/js_grammar.pest"] // relative to src
pub struct JsParser;

type JsRuleError = JsError<Rule>;

fn pair_to_string(pair: Pair<Rule>, level: usize) -> Vec<String> {
    let mut tree = vec![];
    let span = pair.as_span();
    let rule_name = format!(
        "{:?} => ({},{}) #{:?}",
        pair.as_rule(),
        span.start(),
        span.end(),
        span.as_str()
    );
    let mut string_pads = String::with_capacity(level * TAB_WIDTH);
    for _ in 1..level * TAB_WIDTH + 1 {
        string_pads.push('_');
    }
    tree.push(format!("{}{}", string_pads, rule_name));
    for child_pair in pair.into_inner() {
        tree.append(pair_to_string(child_pair, level + 1).as_mut());
    }
    tree
}

impl JsParser {
    /// Parse JavaScript source into a debug token tree.
    ///
    /// Returns a formatted string representation of the parse tree,
    /// useful for debugging the parser grammar.
    ///
    /// # Examples
    ///
    /// ```
    /// use just::parser::JsParser;
    ///
    /// let code = "var x = 5;";
    /// let tree = JsParser::parse_to_token_tree(code).unwrap();
    /// assert!(tree.contains("script"));
    /// ```
    #[cfg(feature = "std")]
    pub fn parse_to_token_tree(script: &str) -> Result<String, String> {
        let mut tree = vec![];
        let start = Instant::now();
        let result = Self::parse(Rule::script, script);
        let end = Instant::now();
        let total_time = end.saturating_duration_since(start);
        println!("Actual parse time is {}ms", total_time.as_millis());

        match result {
            Ok(pairs) => {
                for pair in pairs {
                    tree.push(pair_to_string(pair, 0).join("\n"));
                }
            }
            Err(err) => {
                return Err(format!("Parse error due to {:?}", err));
            }
        }
        Ok(tree.join("\n"))
    }

    /// Parse JavaScript source into an AST.
    ///
    /// Takes an `Rc<String>` to allow efficient sharing of the source text
    /// across AST nodes for error reporting and formatting.
    ///
    /// # Examples
    ///
    /// ```
    /// use just::parser::JsParser;
    /// use alloc::rc::Rc;
    ///
    /// let code = Rc::new("var x = 5 + 3;".to_string());
    /// let ast = JsParser::parse_to_ast(code).unwrap();
    /// assert_eq!(ast.body.len(), 1);
    /// ```
    pub fn parse_to_ast(script: Rc<String>) -> Result<ProgramData, JsRuleError> {
        Self::parse_to_ast_mode(script, false)
    }

    /// Parse JavaScript source into an AST, treating the top-level program as
    /// strict-mode code when `strict` is set (as if a `"use strict"` directive
    /// prologue were present). Nested `"use strict"` directives in function
    /// bodies are honoured regardless. Strict mode makes Annex-B legacy octal
    /// integer/escape literals a Syntax Error (see `check_strict_legacy_octal`).
    pub fn parse_to_ast_mode(
        script: Rc<String>,
        strict: bool,
    ) -> Result<ProgramData, JsRuleError> {
        let result = Self::parse(Rule::script, &script);
        match result {
            Ok(pairs) => {
                check_strict_legacy_octal(pairs.clone(), strict, &script)?;
                build_ast_from_script(pairs, &script)
            }
            Err(err) => {
                return Err(JsRuleError {
                    kind: JsErrorType::ParserValidation(err.clone()),
                    message: format!("Parse error due to \n{}", err),
                });
            }
        }
    }

    /// Parse JavaScript source string into an AST.
    ///
    /// Convenience method that wraps the source in an `Rc<String>`.
    /// This is the most commonly used parsing method.
    ///
    /// # Examples
    ///
    /// ```
    /// use just::parser::JsParser;
    ///
    /// let code = "function add(a, b) { return a + b; }";
    /// let ast = JsParser::parse_to_ast_from_str(code).unwrap();
    /// assert_eq!(ast.body.len(), 1);
    /// ```
    pub fn parse_to_ast_from_str(script: &str) -> Result<ProgramData, JsRuleError> {
        Self::parse_to_ast(Rc::new(script.to_string()))
    }

    /// Like [`parse_to_ast_from_str`], but forces top-level strict mode when
    /// `strict` is set — used by hosts (e.g. the test262 runner) that know a
    /// script is strict from out-of-band metadata (`onlyStrict`) rather than an
    /// in-source `"use strict"` directive.
    pub fn parse_to_ast_from_str_strict(
        script: &str,
        strict: bool,
    ) -> Result<ProgramData, JsRuleError> {
        Self::parse_to_ast_mode(Rc::new(script.to_string()), strict)
    }

    /// Parse JavaScript and return a formatted AST string.
    ///
    /// Useful for debugging and visualizing the AST structure.
    ///
    /// # Examples
    ///
    /// ```
    /// use just::parser::JsParser;
    ///
    /// let code = "var x = 5;";
    /// let formatted = JsParser::parse_to_ast_formatted_string(code).unwrap();
    /// assert!(formatted.contains("VariableDeclaration"));
    /// ```
    pub fn parse_to_ast_formatted_string(script: &str) -> Result<String, JsRuleError> {
        let result = Self::parse_to_ast_from_str(script)?;
        Ok(result.to_formatted_string(script))
    }

    /// Parse a numeric literal string.
    ///
    /// Supports decimal, hexadecimal (0x), binary (0b), octal (0o),
    /// floating point, and scientific notation.
    ///
    /// # Arguments
    ///
    /// * `s` - The numeric string to parse
    /// * `is_error_on_empty` - Whether to return an error on empty string
    ///
    /// # Examples
    ///
    /// ```
    /// use just::parser::JsParser;
    ///
    /// let hex = "0xFF".to_string();
    /// let result = JsParser::parse_numeric_string(&hex, true).unwrap();
    /// ```
    pub fn parse_numeric_string(
        s: &String,
        is_error_on_empty: bool,
    ) -> Result<ExtendedNumberLiteralType, JsRuleError> {
        let result = Self::parse(Rule::string_numeric_literal, s);
        Ok(match result {
            Ok(mut pairs) => {
                if let Some(pair) = pairs.next() {
                    match pair.as_rule() {
                        Rule::str_numeric_literal => build_ast_from_str_numeric_literal(pair)?,
                        _ => return Err(get_unexpected_error("parse_numeric_string", &pair)),
                    }
                } else {
                    if is_error_on_empty {
                        return Err(JsRuleError {
                            kind: JsErrorType::ParserGeneralError,
                            message: "Got empty string".to_string(),
                        });
                    } else {
                        ExtendedNumberLiteralType::Std(NumberLiteralType::IntegerLiteral(0))
                    }
                }
            }
            Err(err) => {
                return Err(JsRuleError {
                    kind: JsErrorType::ParserValidation(err.clone()),
                    message: format!("Parse error due to \n{}", err),
                });
            }
        })
    }
}

fn get_unexpected_error(src: &'static str, pair: &Pair<Rule>) -> JsRuleError {
    get_unexpected_error_with_rule(src, &pair.as_rule())
}

fn get_unexpected_error_with_rule(src: &'static str, rule: &Rule) -> JsRuleError {
    let message = format!("Unexpected state reached in the parser at \"{:?}\". This indicates internal logic error in the parser.", rule);
    JsRuleError {
        message,
        kind: JsErrorType::Unexpected(src),
    }
}

fn get_validation_error(
    error: String,
    kind: AstBuilderValidationErrorType,
    pair: &Pair<Rule>,
    script: &Rc<String>,
) -> JsRuleError {
    get_validation_error_with_meta(error, kind, get_meta(pair, script))
}

fn get_validation_error_with_meta(
    error: String,
    kind: AstBuilderValidationErrorType,
    meta: Meta,
) -> JsRuleError {
    let message = format!("Parsing error encountered: {}", error);
    JsRuleError {
        message,
        kind: JsErrorType::AstBuilderValidation(kind, meta),
    }
}

fn get_meta(pair: &Pair<Rule>, script: &Rc<String>) -> Meta {
    Meta {
        start_index: pair.as_span().start(),
        end_index: pair.as_span().end(),
        script: script.clone(),
    }
}

/// Helper to safely get the first inner pair with proper error handling.
/// Use this instead of `.into_inner().next().unwrap()` for safer error propagation.
fn expect_inner<'a>(
    pair: Pair<'a, Rule>,
    context: &str,
    script: &Rc<String>,
) -> Result<Pair<'a, Rule>, JsRuleError> {
    let meta = get_meta(&pair, script);
    pair.into_inner().next().ok_or_else(|| {
        get_validation_error_with_meta(
            format!("Expected inner token in {}", context),
            AstBuilderValidationErrorType::SyntaxError,
            meta,
        )
    })
}

fn build_ast_from_script(
    pairs: Pairs<Rule>,
    script: &Rc<String>,
) -> Result<ProgramData, JsRuleError> {
    let mut instructions = vec![];
    let mut end: usize = 0;
    for pair in pairs {
        let meta = get_meta(&pair, script);
        if meta.end_index > end {
            end = meta.end_index;
        }
        match pair.as_rule() {
            Rule::EOI => { /* Do nothing */ }
            Rule::statement_list => {
                // There should be only one pair and it should be statement_list (except EOI)
                let (sl, sl_s) = build_ast_from_statement_list(pair, script)?;
                if sl_s.contains_unpaired_continue.is_true() {
                    return Err(get_validation_error_with_meta(
                        "Invalid 'continue' statement".to_string(),
                        AstBuilderValidationErrorType::SyntaxError,
                        meta,
                    ));
                }
                if sl_s.contains_unpaired_break.is_true() {
                    return Err(get_validation_error_with_meta(
                        "Invalid 'break' statement".to_string(),
                        AstBuilderValidationErrorType::SyntaxError,
                        meta,
                    ));
                }
                validate_lexically_declared_names_have_no_duplicates_and_also_not_present_in_var_declared_names(&sl_s)?;
                //sl_s.top_level_lexically_declared_names should be lexically_declared_names for this production
                //sl_s.top_level_var_declared_names should be var_declared_names for this production
                instructions = sl;
            }
            _ => return Err(get_unexpected_error("build_ast_from_script", &pair)),
        };
    }
    Ok(ProgramData {
        meta: Meta {
            start_index: 0,
            end_index: end,
            script: script.clone(),
        },
        body: instructions,
    })
}

fn build_ast_from_declaration(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(DeclarationType, Semantics), JsRuleError> {
    let inner_pair = expect_inner(pair, "build_ast_from_declaration", script)?;
    Ok(match inner_pair.as_rule() {
        Rule::hoistable_declaration | Rule::hoistable_declaration__yield => {
            let (h, _h_s) = build_ast_from_hoistable_declaration(inner_pair, script)?;
            (h, Semantics::new_empty()) // empty top_level_lexically_declared_names
        }
        Rule::class_declaration | Rule::class_declaration__yield => {
            let (c, c_s) = build_ast_from_class_declaration(inner_pair, script)?;
            let mut s = Semantics::new_empty();
            s.top_level_lexically_declared_names = c_s.bound_names;
            (DeclarationType::ClassDeclaration(c), s)
        }
        Rule::lexical_declaration__in | Rule::lexical_declaration__in_yield => {
            let (ld, ld_s) = build_ast_from_lexical_declaration(inner_pair, script)?;
            let mut s = Semantics::new_empty();
            s.top_level_lexically_declared_names = ld_s.bound_names;
            (DeclarationType::VariableDeclaration(ld), s)
        }
        _ => {
            return Err(get_unexpected_error(
                "build_ast_from_declaration",
                &inner_pair,
            ))
        }
    })
}

fn build_ast_from_lexical_declaration(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(VariableDeclarationData, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut s = Semantics::new_empty();
    let mut declarations = vec![];
    let mut inner_iter = pair.into_inner();
    let let_or_const_pair = inner_iter.next().unwrap();
    let is_let = let_or_const_pair.as_str() == "let";
    let binding_list_pair = inner_iter.next().unwrap();
    for lexical_binding_pair in binding_list_pair.into_inner() {
        let lexical_binding_meta = get_meta(&lexical_binding_pair, script);
        let (l, l_s) = build_ast_from_lexical_binding_or_variable_declaration_or_binding_element(
            lexical_binding_pair,
            script,
        )?;
        if !is_let {
            // init is mandatory when it is constant declaration assigning to binding_identifier
            if let PatternType::PatternWhichCanBeExpression(ExpressionPatternType::Identifier(id)) =
                l.id.borrow()
            {
                if l.init.is_none() {
                    return Err(get_validation_error_with_meta(
                        format!(
                            "Initializer not provided for constant declaration: {}",
                            id.name
                        ),
                        AstBuilderValidationErrorType::SyntaxError,
                        lexical_binding_meta,
                    ));
                }
            }
        }
        s.merge(l_s);
        declarations.push(l);
    }
    // Look for duplicates in BoundNames
    let mut found = HashMap::new();
    for n in &s.bound_names {
        if n.name == "let" {
            return Err(get_validation_error_with_meta(
                format!("Illegal name found: {}", n),
                AstBuilderValidationErrorType::SyntaxError,
                n.meta.clone(),
            ));
        } else if found.contains_key(&n.name) {
            return Err(get_validation_error_with_meta(
                format!("Duplicate declaration found: {}", n),
                AstBuilderValidationErrorType::SyntaxError,
                n.meta.clone(),
            ));
        } else {
            found.insert(n.name.clone(), true);
        }
    }
    Ok((
        VariableDeclarationData {
            meta,
            declarations,
            kind: if is_let {
                VariableDeclarationKind::Let
            } else {
                VariableDeclarationKind::Const
            },
        },
        s,
    ))
}

fn build_ast_from_hoistable_declaration(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(DeclarationType, Semantics), JsRuleError> {
    let inner_pair = pair.into_inner().next().unwrap();
    Ok(match inner_pair.as_rule() {
        Rule::generator_declaration | Rule::generator_declaration__yield => {
            let (f, s) =
                build_ast_from_generator_declaration_or_generator_expression(inner_pair, script)?;
            (DeclarationType::FunctionOrGeneratorDeclaration(f), s)
        }
        Rule::function_declaration | Rule::function_declaration__yield => {
            let (f, s) =
                build_ast_from_function_declaration_or_function_expression(inner_pair, script)?;
            (DeclarationType::FunctionOrGeneratorDeclaration(f), s)
        }
        _ => {
            return Err(get_unexpected_error(
                "build_ast_from_hoistable_declaration",
                &inner_pair,
            ))
        }
    })
}

fn build_ast_from_generator_declaration_or_generator_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(FunctionData, Semantics), JsRuleError> {
    let is_generator_declaration = match pair.as_rule() {
        Rule::generator_declaration | Rule::generator_declaration__yield => true,
        _ => false,
    };
    let meta = get_meta(&pair, script);
    let mut pair_iter = pair.into_inner();
    let first_pair = pair_iter.next().unwrap();
    let (f_name, mut s, formal_parameters_pair) =
        if matches!(
            first_pair.as_rule(),
            Rule::binding_identifier | Rule::binding_identifier__yield
        ) {
            let (bi, mut bi_s) = get_binding_identifier_data(first_pair, script)?;
            if !is_generator_declaration {
                // In case of generator_expression we ignore the binding_identifier
                bi_s.bound_names = vec![];
            }
            (Some(bi), bi_s, pair_iter.next().unwrap())
        } else {
            (None, Semantics::new_empty(), first_pair)
        };
    let (args, args_s) = build_ast_from_formal_parameters(formal_parameters_pair, script)?;
    if args_s.contains_yield_expression.is_true() {
        return Err(get_validation_error_with_meta(
            "Cannot reference 'yield' in parameters".to_string(),
            AstBuilderValidationErrorType::SyntaxError,
            meta.clone(),
        ));
    }
    if args_s.contains_super_property.is_true() {
        return Err(get_validation_error_with_meta(
            "Cannot invoke 'super'".to_string(),
            AstBuilderValidationErrorType::SyntaxError,
            meta.clone(),
        ));
    }
    // We first have generator_body then function_body__yield in it
    let function_body_pair = pair_iter.next().unwrap().into_inner().next().unwrap();
    let (f_body, f_body_s) = build_ast_from_function_body(function_body_pair, script)?;
    if f_body_s.contains_super_property.is_true() {
        return Err(get_validation_error_with_meta(
            "Cannot invoke 'super'".to_string(),
            AstBuilderValidationErrorType::SyntaxError,
            meta.clone(),
        ));
    }
    if f_body_s.has_direct_super.is_true() {
        return Err(get_validation_error_with_meta(
            "Invalid reference to 'super'".to_string(),
            AstBuilderValidationErrorType::SyntaxError,
            meta.clone(),
        ));
    }
    validate_bound_names_have_no_duplicates_and_also_not_present_in_var_declared_names_or_lexically_declared_names(&args_s.bound_names, &vec![], &f_body_s.lexically_declared_names)?;
    let body = Box::new(f_body);
    s.merge(args_s);
    // .merge(f_body_s); bound_names from f_body_s are not returned
    Ok((
        FunctionData {
            meta,
            id: f_name,
            body,
            params: FormalParameters::new(args),
            generator: true,
            is_async: false,
        },
        s,
    ))
}

fn build_ast_from_function_declaration_or_function_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(FunctionData, Semantics), JsRuleError> {
    let is_function_declaration = match pair.as_rule() {
        Rule::function_declaration | Rule::function_declaration__yield => true,
        _ => false,
    };
    let meta = get_meta(&pair, script);
    // ChittiOS: the grammar allows an optional `async` prefix before `function`.
    let is_async = pair.as_str().trim_start().starts_with("async");
    let mut pair_iter = pair.into_inner();
    let first_pair = pair_iter.next().unwrap();
    let (f_name, mut s, formal_parameters_pair) =
        if first_pair.as_rule() == Rule::binding_identifier {
            let (bi, mut bi_s) = get_binding_identifier_data(first_pair, script)?;
            if !is_function_declaration {
                // In case of function_expression we ignore the binding_identifier
                bi_s.bound_names = vec![];
            }
            (Some(bi), bi_s, pair_iter.next().unwrap())
        } else {
            (None, Semantics::new_empty(), first_pair)
        };
    let formal_parameters_meta = get_meta(&formal_parameters_pair, script);
    let (args, args_s) = build_ast_from_formal_parameters(formal_parameters_pair, script)?;
    if args_s.contains_super_call.is_true() || args_s.contains_super_property.is_true() {
        return Err(get_validation_error_with_meta(
            "Cannot invoke 'super'".to_string(),
            AstBuilderValidationErrorType::SyntaxError,
            formal_parameters_meta,
        ));
    }
    let function_body_pair = pair_iter.next().unwrap();
    let function_body_meta = get_meta(&function_body_pair, script);
    let (f_body, f_body_s) = build_ast_from_function_body(function_body_pair, script)?;
    if f_body_s.contains_super_call.is_true() || f_body_s.contains_super_property.is_true() {
        return Err(get_validation_error_with_meta(
            "Cannot invoke 'super'".to_string(),
            AstBuilderValidationErrorType::SyntaxError,
            function_body_meta,
        ));
    }
    validate_bound_names_have_no_duplicates_and_also_not_present_in_var_declared_names_or_lexically_declared_names(&args_s.bound_names, &vec![], &f_body_s.lexically_declared_names,)?;
    let body = Box::new(f_body);
    s.merge(args_s);
    // .merge(f_body_s); bound_names from f_body_s are not returned
    Ok((
        FunctionData {
            meta,
            id: f_name,
            body,
            params: FormalParameters::new(args),
            generator: false,
            is_async,
        },
        s,
    ))
}

fn build_ast_from_formal_parameters(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(Vec<PatternType>, Semantics), JsRuleError> {
    let mut args: Vec<PatternType> = vec![];
    let mut s = Semantics::new_empty();
    for param in pair.into_inner() {
        let meta = get_meta(&param, script);
        args.push(match param.as_rule() {
            Rule::function_rest_parameter | Rule::function_rest_parameter__yield => {
                let binding_rest_element = param.into_inner().next().unwrap();
                let (argument, arg_s) =
                    build_ast_from_binding_rest_argument(binding_rest_element, script)?;
                s.merge(arg_s);
                PatternType::RestElement {
                    meta,
                    argument: Box::new(argument),
                }
            }
            Rule::formal_parameter | Rule::formal_parameter__yield => {
                // formal_parameter contains binding_element
                let binding_element = param.into_inner().next().unwrap();
                let (pattern, pattern_s) = build_ast_from_binding_element(binding_element, script)?;
                s.merge(pattern_s);
                pattern
            }
            _ => {
                return Err(get_unexpected_error(
                    "build_ast_from_formal_parameters",
                    &param,
                ))
            }
        });
    }
    Ok((args, s))
}

/// Build AST from a single formal parameter (used for setter property_set_parameter_list).
fn build_ast_from_single_formal_parameter(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(Vec<PatternType>, Semantics), JsRuleError> {
    let mut args: Vec<PatternType> = vec![];
    let mut s = Semantics::new_empty();

    match pair.as_rule() {
        Rule::formal_parameter | Rule::formal_parameter__yield => {
            // formal_parameter contains binding_element
            let binding_element = pair.into_inner().next().unwrap();
            let (pattern, pattern_s) = build_ast_from_binding_element(binding_element, script)?;
            s.merge(pattern_s);
            args.push(pattern);
        }
        _ => {
            return Err(get_unexpected_error(
                "build_ast_from_single_formal_parameter",
                &pair,
            ))
        }
    }
    Ok((args, s))
}

fn build_ast_from_generator_body(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(FunctionBodyData, Semantics), JsRuleError> {
    build_ast_from_function_body(pair.into_inner().next().unwrap(), script)
}

fn build_ast_from_function_body(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(FunctionBodyData, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    // An empty function body (`function f(){}`) produces no `statement_list`
    // pair (the grammar's `statement_list?` is absent), so `next()` is `None`.
    // Return an empty body rather than panicking.
    let statement_list_pair = match pair.into_inner().next() {
        Some(p) => p,
        None => {
            return Ok((
                FunctionBodyData {
                    meta,
                    body: Vec::new(),
                },
                Semantics::new_empty(),
            ));
        }
    };
    let statement_list_meta = get_meta(&statement_list_pair, script);
    let (statements, statements_s) = build_ast_from_statement_list(statement_list_pair, script)?;
    if statements_s.contains_unpaired_continue.is_true() {
        return Err(get_validation_error_with_meta(
            "Invalid 'continue' statement".to_string(),
            AstBuilderValidationErrorType::SyntaxError,
            statement_list_meta,
        ));
    }
    if statements_s.contains_unpaired_break.is_true() {
        return Err(get_validation_error_with_meta(
            "Invalid 'break' statement".to_string(),
            AstBuilderValidationErrorType::SyntaxError,
            statement_list_meta,
        ));
    }
    validate_lexically_declared_names_have_no_duplicates_and_also_not_present_in_var_declared_names(&statements_s)?;
    let mut s = Semantics::new_empty();
    s.lexically_declared_names = statements_s.top_level_lexically_declared_names;
    s.var_declared_names = statements_s.top_level_var_declared_names;
    Ok((
        FunctionBodyData {
            meta,
            body: statements,
        },
        s,
    ))
}

/// Deepest nesting the AST builder will follow before refusing.
///
/// The builder is recursive descent, so nesting in the *source* becomes
/// recursion on the **kernel task stack** — 256 KiB, with frames this large.
/// Real minified code reaches depths hand-written code never does: a chain of
/// 320 `else if`s is one nested `IfStatement` per link, and www.google.com's
/// search page ships exactly that. Overflowing the stack does not raise a Rust
/// error, it takes a synchronous exception whose handler faults again — the
/// machine stops with no output and no Ctrl+C, which is what a page must never
/// be able to do. 256 is far deeper than any legible source and shallow enough
/// to survive with room to spare.
const MAX_AST_DEPTH: u32 = 256;

/// Current builder depth. The parse runs on one task, so a plain counter is
/// enough; [`DepthGuard`] restores it on every exit path including `?`.
static AST_DEPTH: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

struct DepthGuard;

impl DepthGuard {
    /// Enter one level, or `Err` past [`MAX_AST_DEPTH`].
    fn enter(pair: &Pair<Rule>, script: &Rc<String>) -> Result<Self, JsRuleError> {
        let d = AST_DEPTH.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        AST_DEPTH_MAX.fetch_max(d + 1, core::sync::atomic::Ordering::Relaxed);
        if d >= MAX_AST_DEPTH {
            AST_DEPTH.fetch_sub(1, core::sync::atomic::Ordering::Relaxed);
            return Err(get_validation_error(
                alloc::format!("expression or statement nested deeper than {MAX_AST_DEPTH}"),
                AstBuilderValidationErrorType::SyntaxError,
                pair,
                script,
            ));
        }
        Ok(DepthGuard)
    }
}

impl Drop for DepthGuard {
    fn drop(&mut self) {
        AST_DEPTH.fetch_sub(1, core::sync::atomic::Ordering::Relaxed);
    }
}

/// Deepest nesting any parse has reached — reported by the host harness so the
/// limit above can be set from measurements rather than a guess.
pub static AST_DEPTH_MAX: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

/// Reset the depth counter — a parse that bailed out mid-way (any `Err`) unwinds
/// its guards, but a *panic*-free abort elsewhere could leave it non-zero.
pub fn reset_ast_depth() {
    AST_DEPTH.store(0, core::sync::atomic::Ordering::Relaxed);
}

fn build_ast_from_statement(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(StatementType, Semantics), JsRuleError> {
    let _depth = DepthGuard::enter(&pair, script)?;
    let inner_pair = pair.into_inner().next().unwrap();
    let meta = get_meta(&inner_pair, script);
    Ok(match inner_pair.as_rule() {
        Rule::debugger_statement => unimplemented!(),
        Rule::continue_statement | Rule::continue_statement__yield => {
            let (c, _) = build_ast_from_continue_statement(inner_pair, script)?;
            (c, Semantics::new_empty())
        }
        Rule::break_statement | Rule::break_statement__yield => {
            // Optional `label_identifier` child = `break label;` (skip `break_kw`).
            let label = inner_pair
                .into_inner()
                .filter(|p| {
                    !matches!(
                        p.as_rule(),
                        Rule::break_kw | Rule::continue_kw | Rule::smart_semicolon
                    )
                })
                .next()
                .map(|p| p.as_str().trim().to_string());
            (BreakStatement { meta, label }, Semantics::new_empty())
        }
        Rule::labelled_statement
        | Rule::labelled_statement__yield
        | Rule::labelled_statement__return
        | Rule::labelled_statement__yield_return => {
            let mut it = inner_pair.into_inner();
            let label = it.next().unwrap().as_str().trim().to_string();
            let (body, body_s) = build_ast_from_statement(it.next().unwrap(), script)?;
            (
                StatementType::LabelledStatement {
                    meta,
                    label,
                    body: Box::new(body),
                },
                body_s,
            )
        }
        Rule::throw_statement | Rule::throw_statement__yield => {
            let (t, _) = build_ast_from_throw_statement(inner_pair, script)?;
            (t, Semantics::new_empty())
        }
        Rule::if_statement
        | Rule::if_statement__yield
        | Rule::if_statement__return
        | Rule::if_statement__yield_return => build_ast_from_if_statement(inner_pair, script)?,
        Rule::with_statement
        | Rule::with_statement__yield
        | Rule::with_statement__return
        | Rule::with_statement__yield_return => {
            return Err(get_validation_error_with_meta(
                "'with' statement is not supported".to_string(),
                AstBuilderValidationErrorType::SyntaxError,
                meta,
            ));
        }
        Rule::try_statement
        | Rule::try_statement__yield
        | Rule::try_statement__return
        | Rule::try_statement__yield_return => build_ast_from_try_statement(inner_pair, script)?,
        Rule::variable_statement | Rule::variable_statement__yield => {
            build_ast_from_variable_statement(inner_pair, script)?
        }
        Rule::breakable_statement
        | Rule::breakable_statement__yield
        | Rule::breakable_statement__return
        | Rule::breakable_statement__yield_return => {
            build_ast_from_breakable_statement(inner_pair, script)?
        }
        Rule::block_statement
        | Rule::block_statement__yield
        | Rule::block_statement__return
        | Rule::block_statement__yield_return => {
            let (bsd, bsd_s) =
                build_ast_from_block(inner_pair.into_inner().next().unwrap(), script)?;
            (StatementType::BlockStatement(bsd), bsd_s)
        }
        Rule::expression_statement | Rule::expression_statement__yield => {
            let (exp, _exp_s) =
                build_ast_from_expression(inner_pair.into_inner().next().unwrap(), script)?;
            (
                StatementType::ExpressionStatement {
                    meta,
                    expression: Box::new(exp),
                },
                Semantics::new_empty(),
            )
        }
        Rule::empty_statement => (
            StatementType::EmptyStatement { meta },
            Semantics::new_empty(),
        ),
        Rule::return_statement | Rule::return_statement__yield => {
            let (r, _) = build_ast_from_return_statement(inner_pair, script)?;
            (r, Semantics::new_empty())
        }
        _ => {
            return Err(get_unexpected_error(
                "build_ast_from_statement",
                &inner_pair,
            ))
        }
    })
}

fn build_ast_from_if_statement(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(StatementType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut inner_iter = pair.into_inner();
    let expression_pair = inner_iter.next().unwrap();
    let (expression, _) = build_ast_from_expression(expression_pair, script)?;
    let mut s = Semantics::new_empty();
    let (statement_true, statement_true_s) =
        build_ast_from_statement(inner_iter.next().unwrap(), script)?;
    s.var_declared_names = statement_true_s.var_declared_names;
    let (statement_else, _statement_else_s) = if let Some(statement_else_pair) = inner_iter.next() {
        let (st, mut st_s) = build_ast_from_statement(statement_else_pair, script)?;
        s.var_declared_names.append(&mut st_s.var_declared_names);
        (Some(Box::new(st)), st_s)
    } else {
        (None, Semantics::new_empty())
    };
    Ok((
        StatementType::IfStatement {
            meta,
            test: Box::new(expression),
            consequent: Box::new(statement_true),
            alternate: statement_else,
        },
        s,
    ))
}

fn build_ast_from_throw_statement(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(StatementType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut inners = pair.into_inner();
    let mut inner_pair = inners.next().unwrap();
    // `throw_kw` is a named atomic token (identifier-part boundary); skip it.
    if inner_pair.as_rule() == Rule::throw_kw {
        inner_pair = inners.next().unwrap();
    }
    let (e, e_s) = build_ast_from_expression(inner_pair, script)?;
    Ok((
        StatementType::ThrowStatement {
            meta,
            argument: Box::new(e),
        },
        e_s,
    ))
}

fn build_ast_from_continue_statement(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(StatementType, Semantics), JsRuleError> {
    let mut s = Semantics::new_empty();
    s.contains_unpaired_continue.make_true();
    let meta = get_meta(&pair, script);
    let label = pair
        .into_inner()
        .filter(|p| {
            !matches!(
                p.as_rule(),
                Rule::continue_kw | Rule::break_kw | Rule::smart_semicolon
            )
        })
        .next()
        .map(|p| p.as_str().trim().to_string());
    Ok((StatementType::ContinueStatement { meta, label }, s))
}

fn build_ast_from_statement_list(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(Vec<StatementType>, Semantics), JsRuleError> {
    let _meta = get_meta(&pair, script);
    let mut declarations = vec![];
    let mut s = Semantics::new_empty();
    for inner_pair in pair.into_inner() {
        declarations.push(match inner_pair.as_rule() {
            Rule::declaration | Rule::declaration__yield => {
                let (d, mut d_s) = build_ast_from_declaration(inner_pair, script)?;
                s.lexically_declared_names.append(&mut d_s.bound_names);
                StatementType::DeclarationStatement(d)
            }
            Rule::statement
            | Rule::statement__yield
            | Rule::statement__return
            | Rule::statement__yield_return => {
                let (st, st_s) = build_ast_from_statement(inner_pair, script)?;
                s.merge(st_s);
                st
            }
            _ => {
                return Err(get_unexpected_error(
                    "build_ast_from_statement_list",
                    &inner_pair,
                ))
            }
        });
    }
    Ok((declarations, s))
}

fn validate_lexically_declared_names_have_no_duplicates_and_also_not_present_in_var_declared_names(
    s: &Semantics,
) -> Result<(), JsRuleError> {
    // Look for duplicates in LexicallyDeclaredNames
    let mut found = HashMap::new();
    for n in &s.lexically_declared_names {
        if found.contains_key(&n.name) || s.var_declared_names.contains(n) {
            return Err(get_validation_error_with_meta(
                format!("Duplicate declaration found: {}", n),
                AstBuilderValidationErrorType::SyntaxError,
                n.meta.clone(),
            ));
        } else {
            found.insert(n.name.clone(), true);
        }
    }
    Ok(())
}

fn validate_bound_names_have_no_duplicates_and_also_not_present_in_var_declared_names_or_lexically_declared_names(
    bound_names: &Vec<IdentifierData>,
    var_declared_names: &Vec<IdentifierData>,
    lexically_declared_names: &Vec<IdentifierData>,
) -> Result<(), JsRuleError> {
    // Look for duplicates in LexicallyDeclaredNames
    let mut found = HashMap::new();
    for n in bound_names {
        if found.contains_key(&n.name)
            || var_declared_names.contains(n)
            || lexically_declared_names.contains(n)
        {
            return Err(get_validation_error_with_meta(
                format!("Duplicate declaration found: {}", n),
                AstBuilderValidationErrorType::SyntaxError,
                n.meta.clone(),
            ));
        } else {
            found.insert(n.name.clone(), true);
        }
    }
    Ok(())
}

fn build_ast_from_block(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(BlockStatementData, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let (declarations, s) = if let Some(inner_pair) = pair.into_inner().next() {
        build_ast_from_statement_list(inner_pair, script)?
    } else {
        (vec![], Semantics::new_empty())
    };
    validate_lexically_declared_names_have_no_duplicates_and_also_not_present_in_var_declared_names(&s)?;
    Ok((
        BlockStatementData {
            meta,
            body: declarations,
        },
        s,
    ))
}

fn build_ast_from_breakable_statement(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(StatementType, Semantics), JsRuleError> {
    let inner_pair = pair.into_inner().next().unwrap();
    Ok(
        if inner_pair.as_rule() == Rule::iteration_statement
            || inner_pair.as_rule() == Rule::iteration_statement__yield
            || inner_pair.as_rule() == Rule::iteration_statement__return
            || inner_pair.as_rule() == Rule::iteration_statement__yield_return
        {
            build_ast_from_iteration_statement(inner_pair, script)?
        } else {
            build_ast_from_switch_statement(inner_pair, script)?
        },
    )
}

fn build_ast_from_iteration_statement(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(StatementType, Semantics), JsRuleError> {
    // ChittiOS: identify the loop keyword by a word-boundary prefix, not by
    // splitting on the first space — minified code like `while(x)` / `for(;;)`
    // has no space after the keyword and used to fall through to the error arm.
    let text = pair.as_str().trim_start();
    let tag = ["while", "for", "do"]
        .iter()
        .copied()
        .find(|kw| {
            text.starts_with(kw)
                && text[kw.len()..]
                    .chars()
                    .next()
                    .map_or(true, |c| !c.is_alphanumeric() && c != '_' && c != '$')
        })
        .unwrap_or("");
    Ok(match tag {
        "do" => build_ast_for_breakable_statement_do(pair, script)?,
        "while" => build_ast_for_breakable_statement_while(pair, script)?,
        "for" => build_ast_for_breakable_statement_for(pair, script)?,
        _ => {
            return Err(get_unexpected_error(
                "build_ast_from_iteration_statement",
                &pair,
            ))
        }
    })
}

fn build_ast_for_breakable_statement_do(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(StatementType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut inner_iter = pair.into_inner();
    let statement_pair = inner_iter.next().unwrap();
    let test_expression_pair = inner_iter.next().unwrap();
    let mut s = Semantics::new_empty();
    let (e, _) = build_ast_from_expression(test_expression_pair, script)?;
    let (st, st_s) = build_ast_from_statement(statement_pair, script)?;
    s.var_declared_names = st_s.var_declared_names;
    Ok((
        StatementType::DoWhileStatement {
            meta,
            test: Box::new(e),
            body: Box::new(st),
        },
        s,
    ))
}

fn build_ast_for_breakable_statement_while(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(StatementType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut inner_iter = pair.into_inner();
    let test_expression_pair = inner_iter.next().unwrap();
    let statement_pair = inner_iter.next().unwrap();
    let mut s = Semantics::new_empty();
    let (e, _) = build_ast_from_expression(test_expression_pair, script)?;
    let (st, st_s) = build_ast_from_statement(statement_pair, script)?;
    s.var_declared_names = st_s.var_declared_names;
    Ok((
        StatementType::WhileStatement {
            meta,
            test: Box::new(e),
            body: Box::new(st),
        },
        s,
    ))
}

fn build_ast_for_breakable_statement_for(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(StatementType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut inner_iter = pair.into_inner();
    let first_pair = inner_iter.next().unwrap();
    let mut s = Semantics::new_empty();
    Ok(match first_pair.as_rule() {
        Rule::left_hand_side_expression
        | Rule::left_hand_side_expression__yield
        | Rule::for_binding
        | Rule::for_binding__yield
        | Rule::for_declaration
        | Rule::for_declaration__yield => {
            /* for-in or for-of */
            let in_of_left = match first_pair.as_rule() {
                Rule::left_hand_side_expression | Rule::left_hand_side_expression__yield => {
                    let (exp, exp_s) =
                        build_ast_from_left_hand_side_expression(first_pair, script)?;
                    let m = Box::new(convert_lhs_expression_to_pattern_for_assignment_operation(
                        exp,
                        Some(&exp_s),
                    )?);
                    VariableDeclarationOrPattern::Pattern(m)
                }
                Rule::for_binding | Rule::for_binding__yield => {
                    let meta = get_meta(&first_pair, script);
                    let meta2 = meta.clone();
                    let (b, mut b_s) = build_ast_from_for_binding(first_pair, script)?;
                    s.var_declared_names.append(&mut b_s.bound_names);

                    VariableDeclarationOrPattern::VariableDeclaration(VariableDeclarationData {
                        meta,
                        declarations: vec![VariableDeclaratorData {
                            meta: meta2,
                            id: Box::new(b),
                            init: None,
                        }],
                        kind: VariableDeclarationKind::Var,
                    })
                }
                Rule::for_declaration | Rule::for_declaration__yield => {
                    let (d, d_s) = build_ast_from_for_declaration(first_pair, script)?;
                    let mut found = HashMap::new();
                    for bn in &d_s.bound_names {
                        if bn.name == "let" {
                            return Err(get_validation_error_with_meta(
                                "Illegal name: let".to_string(),
                                AstBuilderValidationErrorType::SyntaxError,
                                bn.meta.clone(),
                            ));
                        } else if found.contains_key(&bn.name) {
                            return Err(get_validation_error_with_meta(
                                format!("Duplicate declaration present: {}", bn),
                                AstBuilderValidationErrorType::SyntaxError,
                                bn.meta.clone(),
                            ));
                        } else if d_s.var_declared_names.contains(bn) {
                            return Err(get_validation_error_with_meta(
                                format!("Duplicate declaration present: {}", bn),
                                AstBuilderValidationErrorType::SyntaxError,
                                bn.meta.clone(),
                            ));
                        }
                        found.insert(bn.name.clone(), true);
                    }
                    VariableDeclarationOrPattern::VariableDeclaration(d)
                }
                _ => {
                    return Err(get_unexpected_error(
                        "build_ast_for_breakable_statement_for:2",
                        &first_pair,
                    ))
                }
            };
            let second_pair = inner_iter.next().unwrap();
            let ((in_of_right, _in_of_right_s), is_for_of) = match second_pair.as_rule() {
                Rule::assignment_expression__in | Rule::assignment_expression__in_yield => (
                    build_ast_from_assignment_expression(second_pair, script)?,
                    true,
                ),
                _ => (build_ast_from_expression(second_pair, script)?, false),
            };
            let (statement, mut statement_s) =
                build_ast_from_statement(inner_iter.next().unwrap(), script)?;
            let node = ForIteratorData {
                meta,
                left: in_of_left,
                right: Box::new(in_of_right),
                body: Box::new(statement),
            };
            s.var_declared_names
                .append(&mut statement_s.var_declared_names);
            (
                if is_for_of {
                    StatementType::ForOfStatement(node)
                } else {
                    StatementType::ForInStatement(node)
                },
                s,
            )
        }
        _ => {
            /* for(;;) variation */
            let mut lexical_bound_names = HashMap::new();
            let init = match first_pair.as_rule() {
                Rule::lexical_declaration | Rule::lexical_declaration__yield => {
                    //Lexical Declaration rule ends with smart semicolon which is too flexible. We need to ensure it is semi-colon and nothing else.
                    let last_char = first_pair.as_str().trim_end().chars().last().unwrap();
                    if last_char != ';' {
                        return Err(get_validation_error(
                            format!(
                                "Was expecting semi-colon at the end, but got '{}'.",
                                last_char
                            ),
                            AstBuilderValidationErrorType::SyntaxError,
                            &first_pair,
                            script,
                        ));
                    } else {
                        let (d, d_s) = build_ast_from_lexical_declaration(first_pair, script)?;
                        for n in &d_s.bound_names {
                            lexical_bound_names.insert(n.name.clone(), true);
                        }
                        Some(VariableDeclarationOrExpression::VariableDeclaration(d))
                    }
                }
                Rule::variable_declaration_list | Rule::variable_declaration_list__yield => {
                    let meta = get_meta(&first_pair, script);
                    let (declarations, mut declarations_s) =
                        build_ast_from_variable_declaration_list(first_pair, script)?;
                    s.var_declared_names.append(&mut declarations_s.bound_names);

                    if declarations.is_empty() {
                        None
                    } else {
                        Some(VariableDeclarationOrExpression::VariableDeclaration(
                            VariableDeclarationData {
                                meta,
                                declarations,
                                kind: VariableDeclarationKind::Var,
                            },
                        ))
                    }
                }
                Rule::init_expression | Rule::init_expression__yield => {
                    if let Some(inner_pair) = first_pair.into_inner().next() {
                        let (e, _e_s) = build_ast_from_expression(inner_pair, script)?;
                        Some(VariableDeclarationOrExpression::Expression(Box::new(e)))
                    } else {
                        None
                    }
                }
                _ => {
                    return Err(get_unexpected_error(
                        "build_ast_for_breakable_statement_for:2",
                        &first_pair,
                    ))
                }
            };
            let test_pair = inner_iter.next().unwrap();
            let (test, _test_s) = if let Some(test_expression_pair) = test_pair.into_inner().next() {
                let (e, e_s) = build_ast_from_expression(test_expression_pair, script)?;
                (Some(Box::new(e)), e_s)
            } else {
                (None, Semantics::new_empty())
            };
            let update_pair = inner_iter.next().unwrap();
            let (update, _update_s) =
                if let Some(update_expression_pair) = update_pair.into_inner().next() {
                    let (e, e_s) = build_ast_from_expression(update_expression_pair, script)?;
                    (Some(Box::new(e)), e_s)
                } else {
                    (None, Semantics::new_empty())
                };
            let st_pair = inner_iter.next().unwrap();
            let (st, mut st_s) = build_ast_from_statement(st_pair, script)?;
            for n in &st_s.bound_names {
                if lexical_bound_names.contains_key(&n.name) {
                    return Err(get_validation_error_with_meta(
                        format!("Duplicate declaration found: {}", n),
                        AstBuilderValidationErrorType::SyntaxError,
                        n.meta.clone(),
                    ));
                }
            }
            s.var_declared_names.append(&mut st_s.var_declared_names);
            (
                StatementType::ForStatement {
                    meta,
                    init,
                    test,
                    update,
                    body: Box::new(st),
                },
                s,
            )
        }
    })
}

/// ChittiOS: build the argument pattern of a `binding_rest_element`
/// (`...id`, `...[a,b]`, `...{x}`). The inner is either a `binding_identifier`
/// or a `binding_pattern`.
fn build_ast_from_binding_rest_argument(
    binding_rest_element: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(PatternType, Semantics), JsRuleError> {
    let inner = binding_rest_element.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::binding_pattern | Rule::binding_pattern__yield => {
            build_ast_from_binding_pattern(inner, script)
        }
        _ => {
            let (id, id_s) = get_binding_identifier_data(inner, script)?;
            Ok((
                ExpressionPatternType::Identifier(id).convert_to_pattern(),
                id_s,
            ))
        }
    }
}

fn build_ast_from_binding_pattern(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(PatternType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut binding_properties = vec![];
    let mut s = Semantics::new_empty();
    let inner_pair = pair.into_inner().next().unwrap();
    Ok(match inner_pair.as_rule() {
        Rule::object_binding_pattern | Rule::object_binding_pattern__yield => {
            let binding_pattern_inner_iter = inner_pair.into_inner();
            let mut rest: Option<Box<PatternType>> = None;
            for binding_property_pair in binding_pattern_inner_iter {
                match binding_property_pair.as_rule() {
                    Rule::binding_rest_property | Rule::binding_rest_property__yield => {
                        let id_pair = binding_property_pair.into_inner().next().unwrap();
                        let (id, id_s) = get_binding_identifier_data(id_pair, script)?;
                        s.merge(id_s);
                        rest = Some(Box::new(
                            ExpressionPatternType::Identifier(id).convert_to_pattern(),
                        ));
                    }
                    _ => {
                        let (b, b_s) =
                            build_ast_from_binding_property(binding_property_pair, script)?;
                        s.merge(b_s);
                        binding_properties.push(b);
                    }
                }
            }
            (
                PatternType::ObjectPattern {
                    meta,
                    properties: binding_properties,
                    rest,
                },
                s,
            )
        }
        Rule::array_binding_pattern | Rule::array_binding_pattern__yield => {
            let mut elements: Vec<Option<Box<PatternType>>> = vec![];
            for item_pair in inner_pair.into_inner() {
                match item_pair.as_rule() {
                    Rule::elision => {
                        // Each comma in elision represents a hole
                        for _ in 0..(item_pair.as_str().matches(',').count()) {
                            elements.push(None);
                        }
                    }
                    Rule::binding_element | Rule::binding_element__yield => {
                        let (element, element_s) =
                            build_ast_from_binding_element(item_pair, script)?;
                        s.merge(element_s);
                        elements.push(Some(Box::new(element)));
                    }
                    Rule::binding_rest_element | Rule::binding_rest_element__yield => {
                        let rest_meta = get_meta(&item_pair, script);
                        let (argument, arg_s) =
                            build_ast_from_binding_rest_argument(item_pair, script)?;
                        s.merge(arg_s);
                        elements.push(Some(Box::new(PatternType::RestElement {
                            meta: rest_meta,
                            argument: Box::new(argument),
                        })));
                    }
                    _ => {
                        return Err(get_unexpected_error(
                            "build_ast_from_binding_pattern:array",
                            &item_pair,
                        ))
                    }
                }
            }
            (PatternType::ArrayPattern { meta, elements }, s)
        }
        _ => {
            return Err(get_unexpected_error(
                "build_ast_from_binding_pattern",
                &inner_pair,
            ))
        }
    })
}

/// Handles binding_element rule: single_name_binding | binding_pattern ~ initializer?
fn build_ast_from_binding_element(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(PatternType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut inner_iter = pair.into_inner();
    let inner_pair = inner_iter.next().ok_or_else(|| {
        get_validation_error_with_meta(
            "Expected binding element content".to_string(),
            AstBuilderValidationErrorType::SyntaxError,
            meta.clone(),
        )
    })?;

    match inner_pair.as_rule() {
        Rule::single_name_binding | Rule::single_name_binding__yield => {
            let (var_decl, s) = build_ast_from_single_name_binding(inner_pair, script)?;
            let VariableDeclaratorData { meta, id, init } = var_decl;
            if let Some(init_expr) = init {
                // Has default value: create AssignmentPattern
                Ok((
                    PatternType::AssignmentPattern {
                        meta,
                        left: id,
                        right: init_expr,
                    },
                    s,
                ))
            } else {
                // No default value: just return the pattern
                Ok((*id, s))
            }
        }
        Rule::binding_pattern | Rule::binding_pattern__yield => {
            let (pattern, s) = build_ast_from_binding_pattern(inner_pair, script)?;
            // Check for optional initializer
            if let Some(init_pair) = inner_iter.next() {
                let init_inner = init_pair.into_inner().next().ok_or_else(|| {
                    get_validation_error_with_meta(
                        "Expected initializer expression".to_string(),
                        AstBuilderValidationErrorType::SyntaxError,
                        meta.clone(),
                    )
                })?;
                let (init_expr, _init_s) =
                    build_ast_from_assignment_expression(init_inner, script)?;
                Ok((
                    PatternType::AssignmentPattern {
                        meta,
                        left: Box::new(pattern),
                        right: Box::new(init_expr),
                    },
                    s,
                ))
            } else {
                Ok((pattern, s))
            }
        }
        _ => Err(get_unexpected_error(
            "build_ast_from_binding_element",
            &inner_pair,
        )),
    }
}

fn build_ast_from_binding_property(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(AssignmentPropertyData, Semantics), JsRuleError> {
    let mut inner_iter = pair.into_inner();
    let inner_pair = inner_iter.next().unwrap();
    let meta = get_meta(&inner_pair, script);
    if inner_pair.as_rule() == Rule::single_name_binding
        || inner_pair.as_rule() == Rule::single_name_binding__yield
    {
        let inner_pair_rule = &inner_pair.as_rule();
        let (VariableDeclaratorData { meta, id, init }, s) =
            build_ast_from_single_name_binding(inner_pair, script)?;
        let id = if let PatternType::PatternWhichCanBeExpression(
            ExpressionPatternType::Identifier(id),
        ) = *id
        {
            id
        } else {
            return Err(get_unexpected_error_with_rule(
                "build_ast_from_binding_property:1",
                inner_pair_rule,
            ));
        };
        let meta2 = meta.clone();
        let id2 = id.clone();
        let value = if let Some(init) = init {
            PatternType::AssignmentPattern {
                meta,
                left: Box::new(ExpressionPatternType::Identifier(id).convert_to_pattern()),
                right: init,
            }
        } else {
            ExpressionPatternType::Identifier(id).convert_to_pattern()
        };
        Ok((
            AssignmentPropertyData::new_with_identifier_key(meta2, id2, value, true),
            s,
        ))
    } else if inner_pair.as_rule() == Rule::property_name
        || inner_pair.as_rule() == Rule::property_name__yield
    {
        let (key, _key_s, key_computed) = build_ast_from_property_name(inner_pair, script)?;
        let (value, value_s) =
            build_ast_from_lexical_binding_or_variable_declaration_or_binding_element(
                inner_iter.next().unwrap(),
                script,
            )?;
        let value_exp = if let Some(init) = value.init {
            PatternType::AssignmentPattern {
                meta: value.meta,
                left: value.id,
                right: init,
            }
        } else {
            *value.id
        };
        // s.merge(value_s); bound_names is read only from value_s (binding_element)
        Ok((
            AssignmentPropertyData::new_with_any_expression_key(meta, key, key_computed, value_exp, false),
            value_s,
        ))
    } else {
        Err(get_unexpected_error(
            "build_ast_from_binding_property:2",
            &inner_pair,
        ))
    }
}

/// Build an object/class property *name*.
///
/// The third element of the result is whether the source wrote a **computed**
/// key (`[expr]`). It has to be reported: deciding "computed" from the shape of
/// the key expression — as `new_with_any_expression_key` used to — reads
/// `{ [node]: v }` as the *literal* key `"node"`, because a computed key whose
/// expression happens to be a bare identifier is indistinguishable from a
/// static one after parsing. Radix builds every one of its primitives that way
/// (`NODES.reduce((p, node) => ({ ...p, [node]: … }), {})`), so `Primitive.div`
/// came out `undefined` and React refused the element with `got: undefined`.
fn build_ast_from_property_name(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics, bool), JsRuleError> {
    let inner_pair = pair.into_inner().next().unwrap();
    Ok(if inner_pair.as_rule() == Rule::literal_property_name {
        let pn_pair = inner_pair.into_inner().next().unwrap();
        let s = Semantics::new_empty();
        (
            if pn_pair.as_rule() == Rule::identifier_name || pn_pair.as_rule() == Rule::private_name {
                // A private name (`#x`) is carried as an identifier whose name
                // includes the leading `#`; it resolves to the string key "#x".
                let id = get_identifier_data(pn_pair, script);
                ExpressionPatternType::Identifier(id).convert_to_expression()
            } else if pn_pair.as_rule() == Rule::string_literal {
                let d = build_ast_from_string_literal(pn_pair, script)?;
                ExpressionType::Literal(d)
            } else if pn_pair.as_rule() == Rule::numeric_literal {
                let meta = get_meta(&pn_pair, script);
                let n_pair = pn_pair.into_inner().next().unwrap();
                let value = if n_pair.as_rule() == Rule::bigint_literal {
                    build_ast_from_bigint_literal(n_pair)?
                } else {
                    LiteralType::NumberLiteral(build_ast_from_numeric_literal_inner(n_pair)?)
                };
                ExpressionType::Literal(LiteralData { meta, value })
            } else {
                return Err(get_unexpected_error(
                    "build_ast_from_property_name:1",
                    &pn_pair,
                ));
            },
            s,
            false,
        )
    } else if inner_pair.as_rule() == Rule::computed_property_name
        || inner_pair.as_rule() == Rule::computed_property_name__yield
    {
        let (e, s) =
            build_ast_from_assignment_expression(inner_pair.into_inner().next().unwrap(), script)?;
        return Ok((e, s, true));
    } else {
        return Err(get_unexpected_error(
            "build_ast_from_property_name:2",
            &inner_pair,
        ));
    })
}

fn build_ast_from_for_binding(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(PatternType, Semantics), JsRuleError> {
    let inner_pair = pair.into_inner().next().unwrap();
    Ok(match inner_pair.as_rule() {
        Rule::binding_identifier | Rule::binding_identifier__yield => {
            let (bi, bi_s) = get_binding_identifier_data(inner_pair, script)?;
            (
                ExpressionPatternType::Identifier(bi).convert_to_pattern(),
                bi_s,
            )
        }
        Rule::binding_pattern | Rule::binding_pattern__yield => {
            build_ast_from_binding_pattern(inner_pair, script)?
        }
        _ => {
            return Err(get_unexpected_error(
                "build_ast_from_for_binding",
                &inner_pair,
            ))
        }
    })
}

fn build_ast_from_for_declaration(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(VariableDeclarationData, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut inner_iter = pair.into_inner();
    let let_or_const_pair = inner_iter.next().unwrap();
    let for_binding_pair = inner_iter.next().unwrap();
    let meta2 = meta.clone();
    let (b, b_s) = build_ast_from_for_binding(for_binding_pair, script)?;
    Ok((
        VariableDeclarationData {
            meta,
            declarations: vec![VariableDeclaratorData {
                meta: meta2,
                id: Box::new(b),
                init: None,
            }],
            kind: get_let_or_const(let_or_const_pair)?,
        },
        b_s,
    ))
}

fn get_let_or_const(let_or_const_pair: Pair<Rule>) -> Result<VariableDeclarationKind, JsRuleError> {
    Ok(match let_or_const_pair.as_str() {
        "let" => VariableDeclarationKind::Let,
        "const" => VariableDeclarationKind::Const,
        _ => return Err(get_unexpected_error("get_let_or_const", &let_or_const_pair)),
    })
}

fn build_ast_from_switch_statement(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(StatementType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut inner_iter = pair.into_inner();
    let condition_pair = inner_iter.next().unwrap();
    let (condition, _condition_s) = build_ast_from_expression(condition_pair, script)?;
    let case_block_pair = inner_iter.next().unwrap();
    let mut cases = vec![];
    let mut s = Semantics::new_empty();
    for case_clause_pair in case_block_pair.into_inner() {
        match case_clause_pair.as_rule() {
            Rule::case_clause
            | Rule::case_clause__yield
            | Rule::case_clause__return
            | Rule::case_clause__yield_return => {
                let (c, c_s) = build_ast_from_case_clause(case_clause_pair, script)?;
                cases.push(c);
                s.merge(c_s);
            }
            Rule::default_clause
            | Rule::default_clause__yield
            | Rule::default_clause__return
            | Rule::default_clause__yield_return => {
                let (c, c_s) = build_ast_from_default_clause(case_clause_pair, script)?;
                cases.push(c);
                s.merge(c_s);
            }
            _ => {
                return Err(get_unexpected_error(
                    "build_ast_from_switch_statement",
                    &case_clause_pair,
                ))
            }
        }
    }

    Ok((
        StatementType::SwitchStatement {
            meta,
            discriminant: Box::new(condition),
            cases,
        },
        s,
    ))
}

fn build_ast_from_case_clause(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(SwitchCaseData, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut inner_iter = pair.into_inner();
    let test_pair = inner_iter.next().unwrap();
    let (test_exp, _test_exp_s) = build_ast_from_expression(test_pair, script)?;
    let (statements, statements_s) = if let Some(statement_pair) = inner_iter.next() {
        build_ast_from_statement_list(statement_pair, script)?
    } else {
        (vec![], Semantics::new_empty())
    };
    // Resetting this flag, since if there is a 'break' here then it applies to this 'case' clause.
    // statements_s.contains_unpaired_break.make_false();
    validate_lexically_declared_names_have_no_duplicates_and_also_not_present_in_var_declared_names(&statements_s)?;
    let mut s = Semantics::new_empty();
    s.var_declared_names = statements_s.var_declared_names;
    Ok((
        SwitchCaseData {
            meta,
            test: Some(Box::new(test_exp)),
            consequent: statements,
        },
        s,
    ))
}

fn build_ast_from_default_clause(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(SwitchCaseData, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut inner_iter = pair.into_inner();
    let (statements, statements_s) = if let Some(statement_pair) = inner_iter.next() {
        build_ast_from_statement_list(statement_pair, script)?
    } else {
        (vec![], Semantics::new_empty())
    };
    // Resetting this flag, since if there is a 'break' here then it applies to this 'case' clause.
    // statements_s.contains_unpaired_break.make_false();
    validate_lexically_declared_names_have_no_duplicates_and_also_not_present_in_var_declared_names(&statements_s)?;
    let mut s = Semantics::new_empty();
    s.var_declared_names = statements_s.var_declared_names;
    Ok((
        SwitchCaseData {
            meta,
            test: None,
            consequent: statements,
        },
        s,
    ))
}

fn build_ast_from_return_statement(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(StatementType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut inners = pair.into_inner();
    let mut inner_pair = inners.next().unwrap();
    if inner_pair.as_rule() == Rule::return_kw {
        inner_pair = match inners.next() {
            Some(p) => p,
            None => {
                return Ok((
                    StatementType::ReturnStatement {
                        meta,
                        argument: None,
                    },
                    Semantics::new_empty(),
                ));
            }
        };
    }
    let (argument, s) = if inner_pair.as_rule() == Rule::expression__in {
        let (e, e_s) = build_ast_from_expression(inner_pair, script)?;
        (Some(Box::new(e)), e_s)
    } else if inner_pair.as_rule() == Rule::smart_semicolon {
        (None, Semantics::new_empty())
    } else {
        return Err(get_unexpected_error(
            "build_ast_from_return_statement",
            &inner_pair,
        ));
    };
    Ok((StatementType::ReturnStatement { meta, argument }, s))
}

fn build_ast_from_try_statement(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(StatementType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut inner_iter = pair.into_inner();
    let block_pair = inner_iter.next().unwrap();
    let mut s = Semantics::new_empty();
    let (block, mut block_s) = build_ast_from_block(block_pair, script)?;
    let next_pair = inner_iter.next().unwrap();
    let (handler, mut handler_s, next_pair_option) = match next_pair.as_rule() {
        Rule::catch | Rule::catch__yield | Rule::catch__return | Rule::catch__yield_return => {
            let meta = get_meta(&next_pair, script);
            let mut catch_inner_iter = next_pair.into_inner();
            // Optional catch binding: the first child is either a
            // `catch_parameter` (then the block) or the block itself.
            let first = catch_inner_iter.next().unwrap();
            let (catch_param, catch_param_s, block_pair) = if matches!(
                first.as_rule(),
                Rule::catch_parameter | Rule::catch_parameter__yield
            ) {
                let catch_parameter_inner_pair = first.into_inner().next().unwrap();
                let (p, ps) = match catch_parameter_inner_pair.as_rule() {
                    Rule::binding_identifier | Rule::binding_identifier__yield => {
                        let (bi, bi_s) =
                            get_binding_identifier_data(catch_parameter_inner_pair, script)?;
                        (ExpressionPatternType::Identifier(bi).convert_to_pattern(), bi_s)
                    }
                    Rule::binding_pattern | Rule::binding_pattern__yield => {
                        build_ast_from_binding_pattern(catch_parameter_inner_pair, script)?
                    }
                    _ => {
                        return Err(get_unexpected_error(
                            "build_ast_from_try_statement",
                            &catch_parameter_inner_pair,
                        ))
                    }
                };
                (Some(p), ps, catch_inner_iter.next().unwrap())
            } else {
                // `catch { … }` — no binding; `first` is the block.
                (None, Semantics::new_empty(), first)
            };
            let (block, block_s) = build_ast_from_block(block_pair, script)?;
            validate_bound_names_have_no_duplicates_and_also_not_present_in_var_declared_names_or_lexically_declared_names(&catch_param_s.bound_names, &block_s.var_declared_names,&block_s.lexically_declared_names)?;
            let mut s = Semantics::new_empty();
            s.var_declared_names = block_s.var_declared_names;
            (
                Some(CatchClauseData {
                    meta,
                    param: catch_param.map(Box::new),
                    body: block,
                }),
                s,
                inner_iter.next(),
            )
        }
        _ => (None, Semantics::new_empty(), Some(next_pair)),
    };
    let (finalizer, mut finalizer_s) = if let Some(finally_pair) = next_pair_option {
        let finally_block_pair = finally_pair.into_inner().next().unwrap();
        let (b, b_s) = build_ast_from_block(finally_block_pair, script)?;
        (Some(b), b_s)
    } else {
        (None, Semantics::new_empty())
    };
    s.var_declared_names.append(&mut block_s.var_declared_names);
    s.var_declared_names
        .append(&mut handler_s.var_declared_names);
    s.var_declared_names
        .append(&mut finalizer_s.var_declared_names);
    Ok((
        StatementType::TryStatement {
            meta,
            block,
            handler,
            finalizer,
        },
        s,
    ))
}

fn build_ast_from_variable_statement(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(StatementType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let (dl, dl_s) =
        build_ast_from_variable_declaration_list(pair.into_inner().next().unwrap(), script)?;
    let mut s = Semantics::new_empty();
    s.var_declared_names = dl_s.bound_names;
    Ok((
        StatementType::DeclarationStatement(DeclarationType::VariableDeclaration(
            VariableDeclarationData {
                meta,
                declarations: dl,
                kind: VariableDeclarationKind::Var,
            },
        )),
        s,
    ))
}

fn build_ast_from_variable_declaration_list(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(Vec<VariableDeclaratorData>, Semantics), JsRuleError> {
    let mut declarations = vec![];
    let mut s = Semantics::new_empty();
    for var_pair in pair.into_inner() {
        if var_pair.as_rule() == Rule::variable_declaration
            || var_pair.as_rule() == Rule::variable_declaration__in
        {
            let (d, d_s) =
                build_ast_from_lexical_binding_or_variable_declaration_or_binding_element(
                    var_pair, script,
                )?;
            s.merge(d_s);
            declarations.push(d)
        } else {
            return Err(get_unexpected_error(
                "build_ast_from_variable_declaration_list",
                &var_pair,
            ));
        }
    }
    Ok((declarations, s))
}

fn build_ast_from_lexical_binding_or_variable_declaration_or_binding_element(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(VariableDeclaratorData, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut inner_iter = pair.into_inner();
    let inner_pair = inner_iter.next().unwrap();
    Ok(
        if inner_pair.as_rule() == Rule::binding_identifier
            || inner_pair.as_rule() == Rule::binding_identifier__yield
        {
            let (bi, bi_s) = get_binding_identifier_data(inner_pair, script)?;
            let id = Box::new(ExpressionPatternType::Identifier(bi).convert_to_pattern());
            (
                if let Some(initializer) = inner_iter.next() {
                    let (a, _a_s) = build_ast_from_assignment_expression(
                        initializer.into_inner().next().unwrap(),
                        script,
                    )?;
                    // bi_s.merge(a_s); bound_names is required only from bi_s
                    VariableDeclaratorData {
                        meta,
                        id,
                        init: Some(Box::new(a)),
                    }
                } else {
                    VariableDeclaratorData {
                        meta,
                        id,
                        init: None,
                    }
                },
                bi_s,
            )
        } else if inner_pair.as_rule() == Rule::binding_pattern
            || inner_pair.as_rule() == Rule::binding_pattern__yield
        {
            let (b, b_s) = build_ast_from_binding_pattern(inner_pair, script)?;
            let id = Box::new(b);
            let init = if let Some(initializer) = inner_iter.next() {
                let (a, _a_s) = build_ast_from_assignment_expression(
                    initializer.into_inner().next().unwrap(),
                    script,
                )?;
                // b_s.merge(a_s);  bound_names is required only from b_s
                Some(Box::new(a))
            } else {
                None
            };
            (VariableDeclaratorData { meta, id, init }, b_s)
        } else if inner_pair.as_rule() == Rule::single_name_binding
            || inner_pair.as_rule() == Rule::single_name_binding__yield
        {
            build_ast_from_single_name_binding(inner_pair, script)?
        } else {
            return Err(get_unexpected_error(
                "build_ast_from_lexical_binding_or_variable_declaration",
                &inner_pair,
            ));
        },
    )
}

fn build_ast_from_single_name_binding(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(VariableDeclaratorData, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut inner_iter = pair.into_inner();
    let (b, s) = get_binding_identifier_data(inner_iter.next().unwrap(), script)?;
    let id = Box::new(ExpressionPatternType::Identifier(b).convert_to_pattern());
    Ok(if let Some(initializer) = inner_iter.next() {
        // `initializer` is the `= AssignmentExpression` rule; descend to the
        // inner assignment_expression (as the sibling binding builders do).
        let (a, _a_s) = build_ast_from_assignment_expression(
            initializer.into_inner().next().unwrap(),
            script,
        )?;
        // s.merge(a_s); bound_names from s is only used
        (
            VariableDeclaratorData {
                meta,
                id,
                init: Some(Box::new(a)),
            },
            s,
        )
    } else {
        (
            VariableDeclaratorData {
                meta,
                id,
                init: None,
            },
            s,
        )
    })
}

fn build_ast_from_assignment_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let _depth = DepthGuard::enter(&pair, script)?;
    let meta = get_meta(&pair, script);
    let mut inner_pair_iter = pair.into_inner().peekable();
    let inner_pair = inner_pair_iter.next().unwrap();
    Ok(match inner_pair.as_rule() {
        Rule::arrow_function
        | Rule::arrow_function__in
        | Rule::arrow_function__yield
        | Rule::arrow_function__in_yield => build_ast_from_arrow_function(inner_pair, script)?,
        // ChittiOS: the grammar now parses `cond (op rhs)?` instead of
        // re-parsing the LHS separately (exponential backtracking on minified
        // code); when an operator follows, the conditional result must be a
        // valid assignment target, validated structurally here.
        Rule::conditional_expression
        | Rule::conditional_expression__in
        | Rule::conditional_expression__yield
        | Rule::conditional_expression__in_yield
            if inner_pair_iter.peek().is_some() =>
        {
            let lhs_pair = inner_pair;
            let lhs_meta = get_meta(&lhs_pair, script);
            let (lhs, mut s) = build_ast_from_conditional_expression(lhs_pair, script)?;
            let mut next_pair = inner_pair_iter.next().unwrap();
            let (left, operator) = if next_pair.as_rule() == Rule::assignment_operator {
                // A compound assignment target must be a simple reference —
                // an identifier or a member expression.
                if !matches!(
                    &lhs,
                    ExpressionType::MemberExpression(_)
                        | ExpressionType::ExpressionWhichCanBePattern(
                            ExpressionPatternType::Identifier(_)
                        )
                ) {
                    return Err(get_validation_error_with_meta(
                        "L.H.S. needs to be a simple expression".to_string(),
                        AstBuilderValidationErrorType::ReferenceError,
                        lhs_meta,
                    ));
                }
                if s.is_valid_simple_assignment_target.is_false() {
                    return Err(get_validation_error_with_meta(
                        "L.H.S. needs to be a simple expression".to_string(),
                        AstBuilderValidationErrorType::ReferenceError,
                        lhs_meta,
                    ));
                }
                let op_str = next_pair.as_str();
                next_pair = inner_pair_iter.next().unwrap();
                (
                    PatternOrExpression::Expression(Box::new(lhs)),
                    match op_str {
                        "*=" => AssignmentOperator::MultiplyEquals,
                        "/=" => AssignmentOperator::DivideEquals,
                        "%=" => AssignmentOperator::ModuloEquals,
                        "+=" => AssignmentOperator::AddEquals,
                        "-=" => AssignmentOperator::SubtractEquals,
                        "<<=" => AssignmentOperator::BitwiseLeftShiftEquals,
                        ">>=" => AssignmentOperator::BitwiseRightShiftEquals,
                        ">>>=" => AssignmentOperator::BitwiseUnsignedRightShiftEquals,
                        "&=" => AssignmentOperator::BitwiseAndEquals,
                        "^=" => AssignmentOperator::BitwiseXorEquals,
                        "|=" => AssignmentOperator::BitwiseOrEquals,
                        "**=" => AssignmentOperator::ExponentEquals,
                        "&&=" => AssignmentOperator::LogicalAndEquals,
                        "||=" => AssignmentOperator::LogicalOrEquals,
                        "??=" => AssignmentOperator::NullishEquals,
                        _ => {
                            return Err(get_unexpected_error(
                                "build_ast_from_assignment_expression:1",
                                &next_pair,
                            ))
                        }
                    },
                )
            } else {
                // For simple `=` assignment, check if LHS needs pattern conversion
                // Member expressions and identifiers stay as expressions
                // Array/object literals become destructuring patterns
                let left = match &lhs {
                    ExpressionType::MemberExpression(_) => {
                        // Member expression stays as expression (e.g., obj.prop = value)
                        PatternOrExpression::Expression(Box::new(lhs))
                    }
                    ExpressionType::ExpressionWhichCanBePattern(ExpressionPatternType::Identifier(_)) => {
                        // Simple identifier stays as expression (e.g., x = value)
                        PatternOrExpression::Expression(Box::new(lhs))
                    }
                    _ => {
                        // Array/object literals need pattern conversion for destructuring
                        PatternOrExpression::Pattern(Box::new(
                            convert_lhs_expression_to_pattern_for_assignment_operation(lhs, Some(&s))?,
                        ))
                    }
                };
                (left, AssignmentOperator::Equals)
            };
            // next_pair is now assignment_expression
            let (assignment_exp, assignment_exp_s) =
                build_ast_from_assignment_expression(next_pair, script)?;
            s.merge(assignment_exp_s);
            (
                ExpressionType::AssignmentExpression {
                    meta,
                    left,
                    operator,
                    right: Box::new(assignment_exp),
                },
                s,
            )
        }
        Rule::yield_expression | Rule::yield_expression__in => {
            match inner_pair.into_inner().next() {
                Some(inner_pair) => {
                    let (assign_rule_pair, delegate) = if inner_pair.as_rule()
                        == Rule::star_assignment_expression__yield
                        || inner_pair.as_rule() == Rule::star_assignment_expression__in_yield
                    {
                        (inner_pair.into_inner().next().unwrap(), true)
                    } else {
                        (inner_pair, false)
                    };
                    let (assignment_exp, assignment_exp_s) =
                        build_ast_from_assignment_expression(assign_rule_pair, script)?;
                    (
                        ExpressionType::YieldExpression {
                            meta,
                            delegate,
                            argument: Some(Box::new(assignment_exp)),
                        },
                        assignment_exp_s,
                    )
                }
                None => (
                    ExpressionType::YieldExpression {
                        meta,
                        delegate: false,
                        argument: None,
                    },
                    Semantics::new_empty(),
                ),
            }
        }
        Rule::conditional_expression
        | Rule::conditional_expression__in
        | Rule::conditional_expression__yield
        | Rule::conditional_expression__in_yield => {
            build_ast_from_conditional_expression(inner_pair, script)?
        }
        _ => {
            return Err(get_unexpected_error(
                "build_ast_from_assignment_expression:2",
                &inner_pair,
            ))
        }
    })
}

fn convert_lhs_expression_to_pattern_for_assignment_operation(
    lhs_exp: ExpressionType,
    s: Option<&Semantics>,
) -> Result<PatternType, JsRuleError> {
    match lhs_exp {
        ExpressionType::ExpressionWhichCanBePattern(p) => Ok(p.convert_to_pattern()),
        ExpressionType::MemberExpression(member_expr) => {
            // MemberExpression is a valid assignment target (e.g., obj.prop = value)
            // but it doesn't convert directly to a pattern in the destructuring sense.
            // Return an error as member expressions can't be destructured.
            Err(get_validation_error_with_meta(
                "Member expression cannot be used as destructuring pattern".to_string(),
                AstBuilderValidationErrorType::ReferenceError,
                member_expr.get_meta().clone(),
            ))
        }
        ExpressionType::ArrayExpression {
            meta,
            elements: o_elements,
        } => {
            // We need to convert this to ObjectPattern
            let mut elements = vec![];
            for p in o_elements {
                elements.push(if let Some(es) = p {
                    Some(Box::new(match es {
                        ExpressionOrSpreadElement::Expression(e) => {
                            convert_lhs_expression_to_pattern_for_assignment_operation(
                                *e,
                                None,
                            )?
                        }
                        ExpressionOrSpreadElement::SpreadElement(s) => {
                            let s = convert_lhs_expression_to_pattern_for_assignment_operation(
                                *s,
                                None,
                            )?;
                            PatternType::RestElement {
                                meta: s.get_meta().clone(),
                                argument: Box::new(s),
                            }
                        }
                    }))
                } else {
                    None
                });
            }
            Ok(PatternType::ArrayPattern { meta, elements })
        }
        ExpressionType::ObjectExpression {
            meta,
            properties: o_props,
        } => {
            // We need to convert this to ObjectPattern
            let mut properties = vec![];
            for p in o_props {
                if p.method {
                    return Err(get_validation_error_with_meta(
                        "Invalid object pattern. Cannot have methods.".to_string(),
                        AstBuilderValidationErrorType::SyntaxError,
                        p.meta,
                    ));
                } else {
                    properties.push(AssignmentPropertyData::new_with_any_expression_key(
                        p.meta,
                        *p.key,
                        p.computed,
                        convert_lhs_expression_to_pattern_for_assignment_operation(
                            *p.value,
                            None,
                        )?,
                        p.shorthand,
                    ));
                }
            }
            Ok(PatternType::ObjectPattern { meta, properties, rest: None })
        }
        ExpressionType::Literal(LiteralData { meta, .. })
        | ExpressionType::ThisExpression { meta }
        | ExpressionType::FunctionOrGeneratorExpression(FunctionData { meta, .. })
        | ExpressionType::UnaryExpression { meta, .. }
        | ExpressionType::UpdateExpression { meta, .. }
        | ExpressionType::BinaryExpression { meta, .. }
        | ExpressionType::AssignmentExpression { meta, .. }
        | ExpressionType::LogicalExpression { meta, .. }
        | ExpressionType::ConditionalExpression { meta, .. }
        | ExpressionType::CallExpression { meta, .. }
        | ExpressionType::OptionalChain { meta, .. }
        | ExpressionType::NewExpression { meta, .. }
        | ExpressionType::SequenceExpression { meta, .. }
        | ExpressionType::ArrowFunctionExpression { meta, .. }
        | ExpressionType::YieldExpression { meta, .. }
        | ExpressionType::TemplateLiteral(TemplateLiteralData { meta, .. })
        | ExpressionType::TaggedTemplateExpression { meta, .. }
        | ExpressionType::ClassExpression(ClassData { meta, .. })
        | ExpressionType::ImportCall { meta, .. }
        | ExpressionType::ImportMeta { meta }
        | ExpressionType::MetaProperty { meta, .. } => {
            if s.is_some() && s.unwrap().is_valid_simple_assignment_target.is_true() {
                Err(JsRuleError{ kind: JsErrorType::Unexpected("Unexpected error reached in convert_lhs_expression_to_pattern"), message: "Did not expect a simple assignment target here. It then needs to be converted to pattern".to_string() })
            } else {
                Err( get_validation_error_with_meta("Parsing error encountered: L.H.S. needs to be a simple expression or object/array literal".to_string(),AstBuilderValidationErrorType::ReferenceError, meta ) )
            }
        }
    }
}

fn build_ast_from_arrow_function(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    // ChittiOS: optional `async` prefix (grammar allows `async () => …`).
    let is_async = pair.as_str().trim_start().starts_with("async");
    let mut inner_iter = pair.into_inner();
    let arrow_parameters_pair = inner_iter.next().unwrap();
    let inner_arrow_parameters_pair = arrow_parameters_pair.into_inner().next().unwrap();
    let (params, params_s) = match inner_arrow_parameters_pair.as_rule() {
        Rule::binding_identifier | Rule::binding_identifier__yield => {
            let (b, b_s) = get_binding_identifier_data(inner_arrow_parameters_pair, script)?;
            (
                vec![
                    ExpressionPatternType::Identifier(b).convert_to_pattern()
                ],
                b_s,
            )
        }
        Rule::formal_parameters | Rule::formal_parameters__yield => {
            let (f, f_s) = build_ast_from_formal_parameters(inner_arrow_parameters_pair, script)?;
            (f, f_s)
        }
        _ => {
            return Err(get_unexpected_error(
                "build_ast_from_arrow_function:1",
                &inner_arrow_parameters_pair,
            ));
        }
    };
    let concise_body_pair = inner_iter.next().unwrap();
    let inner_concise_body_pair = concise_body_pair.into_inner().next().unwrap();
    let (body, body_s) = match inner_concise_body_pair.as_rule() {
        Rule::function_body => {
            let (f, f_s) = build_ast_from_function_body(inner_concise_body_pair, script)?;
            (Box::new(FunctionBodyOrExpression::FunctionBody(f)), f_s)
        }
        Rule::assignment_expression | Rule::assignment_expression__in => {
            let (a, _a_s) = build_ast_from_assignment_expression(inner_concise_body_pair, script)?;
            (
                Box::new(FunctionBodyOrExpression::Expression(a)),
                Semantics::new_empty(),
            )
        }
        _ => {
            return Err(get_unexpected_error(
                "build_ast_from_arrow_function:2",
                &inner_concise_body_pair,
            ));
        }
    };
    if body_s.contains_yield_expression.is_true() || params_s.contains_yield_expression.is_true() {
        Err(get_validation_error_with_meta(
            "'yield' is not allowed in arrow function".to_string(),
            AstBuilderValidationErrorType::SyntaxError,
            body.get_meta().clone(),
        ))
    } else {
        validate_bound_names_have_no_duplicates_and_also_not_present_in_var_declared_names_or_lexically_declared_names(&params_s.bound_names,&vec![], &body_s.lexically_declared_names)?;

        // let mut s = Semantics::new_empty();
        // s.merge(params_s).merge(body_s);
        // params_s produces bound_names and body_s produces var_declared_names & lexically_declared_names
        Ok((
            ExpressionType::ArrowFunctionExpression { meta, params, body, is_async },
            Semantics::new_empty(),
        ))
    }
}

/// ChittiOS: unified LHS builder for the restructured grammar
/// `new_op* ~ (super_call | member_expression) ~ suffix*`. The member base is
/// parsed once; each `arguments` suffix is consumed by a pending bare `new`
/// (innermost first — spec `new MemberExpression Arguments` semantics) or
/// becomes a call; leftover `new`s wrap the result with empty arguments.
fn build_ast_from_left_hand_side_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let mut pair_iter = pair.into_inner().peekable();
    let mut new_count = 0usize;
    while pair_iter
        .peek()
        .map(|p| p.as_rule() == Rule::new_op)
        .unwrap_or(false)
    {
        pair_iter.next();
        new_count += 1;
    }
    let base_pair = pair_iter.next().ok_or_else(|| {
        get_unexpected_error_with_rule(
            "build_ast_from_left_hand_side_expression:empty",
            &Rule::left_hand_side_expression,
        )
    })?;
    let base_meta = get_meta(&base_pair, script);
    let mut s = Semantics::new_empty();
    let mut obj = match base_pair.as_rule() {
        Rule::super_call | Rule::super_call__yield => {
            let arguments_pair = base_pair.into_inner().next().ok_or_else(|| {
                get_unexpected_error_with_rule(
                    "build_ast_from_left_hand_side_expression:super",
                    &Rule::super_call,
                )
            })?;
            let (a, a_s) = build_ast_from_arguments(arguments_pair, script)?;
            s.merge(a_s);
            ExpressionType::CallExpression {
                meta: base_meta,
                callee: ExpressionOrSuper::Super,
                arguments: a,
            }
        }
        Rule::member_expression | Rule::member_expression__yield => {
            let (m, m_s) = build_ast_from_member_expression(base_pair, script)?;
            s.merge(m_s);
            m
        }
        _ => {
            return Err(get_unexpected_error(
                "build_ast_from_left_hand_side_expression",
                &base_pair,
            ))
        }
    };
    for suffix_pair in pair_iter {
        let second_meta = get_meta(&suffix_pair, script);
        let meta = Meta {
            start_index: obj.get_meta().start_index,
            end_index: second_meta.end_index,
            script: script.clone(),
        };
        obj = match suffix_pair.as_rule() {
            Rule::arguments | Rule::arguments__yield => {
                let (a, a_s) = build_ast_from_arguments(suffix_pair, script)?;
                s.merge(a_s);
                if new_count > 0 {
                    new_count -= 1;
                    ExpressionType::NewExpression {
                        meta,
                        callee: Box::new(obj),
                        arguments: a,
                    }
                } else {
                    ExpressionType::CallExpression {
                        meta,
                        callee: ExpressionOrSuper::Expression(Box::new(obj)),
                        arguments: a,
                    }
                }
            }
            Rule::expression__in | Rule::expression__in_yield => {
                let (e, e_s) = build_ast_from_expression(suffix_pair, script)?;
                s.merge(e_s);
                ExpressionType::MemberExpression(
                    MemberExpressionType::ComputedMemberExpression {
                        meta,
                        object: ExpressionOrSuper::Expression(Box::new(obj)),
                        property: Box::new(e),
                    },
                )
            }
            Rule::identifier_name | Rule::private_name => ExpressionType::MemberExpression(
                MemberExpressionType::SimpleMemberExpression {
                    meta,
                    object: ExpressionOrSuper::Expression(Box::new(obj)),
                    property: get_identifier_data(suffix_pair, script),
                },
            ),
            Rule::optional_member => ExpressionType::OptionalChain {
                meta,
                object: Box::new(obj),
                access: crate::parser::ast::OptionalAccess::Member(
                    suffix_pair.into_inner().next().unwrap().as_str().trim().to_string(),
                ),
            },
            Rule::optional_index | Rule::optional_index__yield => {
                let (e, e_s) =
                    build_ast_from_expression(suffix_pair.into_inner().next().unwrap(), script)?;
                s.merge(e_s);
                ExpressionType::OptionalChain {
                    meta,
                    object: Box::new(obj),
                    access: crate::parser::ast::OptionalAccess::Computed(Box::new(e)),
                }
            }
            Rule::optional_call | Rule::optional_call__yield => {
                let (a, a_s) =
                    build_ast_from_arguments(suffix_pair.into_inner().next().unwrap(), script)?;
                s.merge(a_s);
                ExpressionType::OptionalChain {
                    meta,
                    object: Box::new(obj),
                    access: crate::parser::ast::OptionalAccess::Call(a),
                }
            }
            Rule::template_literal | Rule::template_literal__yield => {
                let (template, t_s) = build_ast_from_template_literal(suffix_pair, script)?;
                s.merge(t_s);
                let quasi = if let ExpressionType::TemplateLiteral(data) = template {
                    data
                } else {
                    return Err(get_validation_error_with_meta(
                        "Expected template literal".to_string(),
                        AstBuilderValidationErrorType::SyntaxError,
                        meta.clone(),
                    ));
                };
                ExpressionType::TaggedTemplateExpression {
                    meta,
                    tag: Box::new(obj),
                    quasi,
                }
            }
            _ => {
                return Err(get_unexpected_error(
                    "build_ast_from_left_hand_side_expression:suffix",
                    &suffix_pair,
                ))
            }
        };
    }
    // Bare `new`s with no argument list (`new X`, `new new X`).
    while new_count > 0 {
        new_count -= 1;
        obj = ExpressionType::NewExpression {
            meta: obj.get_meta().clone(),
            callee: Box::new(obj),
            arguments: vec![],
        };
    }
    Ok((obj, s))
}

fn build_ast_from_conditional_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut pair_iter = pair.into_inner();
    let logical_or_pair = pair_iter.next().unwrap();
    let (or_node, mut s) = build_ast_from_coalesce_expression(logical_or_pair, script)?;
    if let Some(inner_pair) = pair_iter.next() {
        let (truthy, truthy_s) = build_ast_from_assignment_expression(inner_pair, script)?;
        let (falsy, falsy_s) =
            build_ast_from_assignment_expression(pair_iter.next().unwrap(), script)?;
        s.merge(truthy_s).merge(falsy_s);
        Ok((
            ExpressionType::ConditionalExpression {
                meta,
                test: Box::new(or_node),
                consequent: Box::new(truthy),
                alternate: Box::new(falsy),
            },
            s,
        ))
    } else {
        Ok((or_node, s))
    }
}

fn get_ast_for_logical_expression(
    left: Option<ExpressionType>,
    right: ExpressionType,
    operator: LogicalOperator,
    script: &Rc<String>,
) -> ExpressionType {
    if let Some(actual_left) = left {
        ExpressionType::LogicalExpression {
            meta: Meta {
                start_index: actual_left.get_meta().start_index,
                end_index: right.get_meta().end_index,
                script: script.clone(),
            },
            operator,
            left: Box::new(actual_left),
            right: Box::new(right),
        }
    } else {
        right
    }
}

fn get_ast_for_binary_expression(
    left: Option<ExpressionType>,
    right: ExpressionType,
    operator: Option<BinaryOperator>,
    script: &Rc<String>,
) -> ExpressionType {
    if let Some(actual_left) = left {
        ExpressionType::BinaryExpression {
            meta: Meta {
                start_index: actual_left.get_meta().start_index,
                end_index: right.get_meta().end_index,
                script: script.clone(),
            },
            operator: operator.unwrap(),
            left: Box::new(actual_left),
            right: Box::new(right),
        }
    } else {
        right
    }
}

fn build_ast_from_coalesce_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let mut left = None;
    let mut s = Semantics::new_empty();
    for inner_pair in pair.into_inner() {
        let (right, right_s) = build_ast_from_logical_or_expression(inner_pair, script)?;
        s.merge(right_s);
        left = Some(get_ast_for_logical_expression(
            left,
            right,
            LogicalOperator::Coalesce,
            script,
        ));
    }
    Ok((left.unwrap(), s))
}

fn build_ast_from_logical_or_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let mut left = None;
    let mut s = Semantics::new_empty();
    for inner_pair in pair.into_inner() {
        let (right, right_s) = build_ast_from_logical_and_expression(inner_pair, script)?;
        s.merge(right_s);
        left = Some(get_ast_for_logical_expression(
            left,
            right,
            LogicalOperator::Or,
            script,
        ));
    }
    Ok((left.unwrap(), s))
}

fn build_ast_from_logical_and_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let mut left = None;
    let mut s = Semantics::new_empty();
    for inner_pair in pair.into_inner() {
        let (right, right_s) = build_ast_from_bitwise_or_expression(inner_pair, script)?;
        s.merge(right_s);
        left = Some(get_ast_for_logical_expression(
            left,
            right,
            LogicalOperator::And,
            script,
        ));
    }
    Ok((left.unwrap(), s))
}

fn build_ast_from_bitwise_or_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let mut left = None;
    let mut s = Semantics::new_empty();
    for inner_pair in pair.into_inner() {
        let (right, right_s) = build_ast_from_bitwise_xor_expression(inner_pair, script)?;
        s.merge(right_s);
        left = Some(get_ast_for_binary_expression(
            left,
            right,
            Some(BinaryOperator::BitwiseOr),
            script,
        ))
    }
    Ok((left.unwrap(), s))
}

fn build_ast_from_bitwise_xor_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let mut left = None;
    let mut s = Semantics::new_empty();
    for inner_pair in pair.into_inner() {
        let (right, right_s) = build_ast_from_bitwise_and_expression(inner_pair, script)?;
        s.merge(right_s);
        left = Some(get_ast_for_binary_expression(
            left,
            right,
            Some(BinaryOperator::BitwiseXor),
            script,
        ))
    }
    Ok((left.unwrap(), s))
}

fn build_ast_from_bitwise_and_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let mut left = None;
    let mut s = Semantics::new_empty();
    for inner_pair in pair.into_inner() {
        let (right, right_s) = build_ast_from_equality_expression(inner_pair, script)?;
        s.merge(right_s);
        left = Some(get_ast_for_binary_expression(
            left,
            right,
            Some(BinaryOperator::BitwiseAnd),
            script,
        ))
    }
    Ok((left.unwrap(), s))
}

fn build_ast_from_equality_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    if cfg!(debug_assertions) {
        if pair.as_rule() != Rule::equality_expression
            && pair.as_rule() != Rule::equality_expression__in
            && pair.as_rule() != Rule::equality_expression__yield
            && pair.as_rule() != Rule::equality_expression__in_yield
        {
            return Err(get_unexpected_error(
                "build_ast_from_equality_expression:0",
                &pair,
            ));
        }
    }
    let mut left = None;
    let mut s = Semantics::new_empty();
    let mut pair_iter = pair.into_inner();
    loop {
        if let Some(mut inner_pair) = pair_iter.next() {
            let mut operator = None;
            if inner_pair.as_rule() == Rule::equality_operator {
                operator = Some(match inner_pair.as_str() {
                    "===" => BinaryOperator::StrictlyEqual,
                    "!==" => BinaryOperator::StrictlyUnequal,
                    "==" => BinaryOperator::LooselyEqual,
                    "!=" => BinaryOperator::LooselyUnequal,
                    _ => {
                        return Err(get_unexpected_error(
                            "build_ast_from_equality_expression",
                            &inner_pair,
                        ))
                    }
                });
                inner_pair = pair_iter.next().unwrap();
            }
            let (right, right_s) = build_ast_from_relational_expression(inner_pair, script)?;
            s.merge(right_s);
            left = Some(get_ast_for_binary_expression(left, right, operator, script));
        } else {
            break;
        }
    }
    Ok((left.unwrap(), s))
}

fn build_ast_from_relational_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    if cfg!(debug_assertions) {
        if pair.as_rule() != Rule::relational_expression
            && pair.as_rule() != Rule::relational_expression__in
            && pair.as_rule() != Rule::relational_expression__yield
            && pair.as_rule() != Rule::relational_expression__in_yield
        {
            return Err(get_unexpected_error(
                "build_ast_from_relational_expression:0",
                &pair,
            ));
        }
    }
    let mut left = None;
    let mut s = Semantics::new_empty();
    let mut pair_iter = pair.into_inner();
    loop {
        if let Some(mut inner_pair) = pair_iter.next() {
            let mut operator = None;
            if inner_pair.as_rule() == Rule::relational_operator
                || inner_pair.as_rule() == Rule::relational_operator__in
            {
                operator = Some(match inner_pair.as_str() {
                    "<=" => BinaryOperator::LessThanEqual,
                    ">=" => BinaryOperator::GreaterThanEqual,
                    "<" => BinaryOperator::LessThan,
                    ">" => BinaryOperator::GreaterThan,
                    "instanceof" => BinaryOperator::InstanceOf,
                    "in" => BinaryOperator::In,
                    _ => {
                        return Err(get_unexpected_error(
                            "build_ast_from_relational_expression",
                            &inner_pair,
                        ))
                    }
                });
                inner_pair = pair_iter.next().unwrap();
            }
            let (right, right_s) = build_ast_from_shift_expression(inner_pair, script)?;
            s.merge(right_s);
            left = Some(get_ast_for_binary_expression(left, right, operator, script));
        } else {
            break;
        }
    }
    Ok((left.unwrap(), s))
}

fn build_ast_from_shift_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    if cfg!(debug_assertions) {
        if pair.as_rule() != Rule::shift_expression
            && pair.as_rule() != Rule::shift_expression__yield
        {
            return Err(get_unexpected_error(
                "build_ast_from_shift_expression:0",
                &pair,
            ));
        }
    }
    let mut left = None;
    let mut s = Semantics::new_empty();
    let mut pair_iter = pair.into_inner();
    loop {
        if let Some(mut inner_pair) = pair_iter.next() {
            let mut operator = None;
            if inner_pair.as_rule() == Rule::shift_operator {
                operator = Some(match inner_pair.as_str() {
                    "<<" => BinaryOperator::BitwiseLeftShift,
                    ">>>" => BinaryOperator::BitwiseUnsignedRightShift,
                    ">>" => BinaryOperator::BitwiseRightShift,
                    _ => {
                        return Err(get_unexpected_error(
                            "build_ast_from_shift_expression",
                            &inner_pair,
                        ))
                    }
                });
                inner_pair = pair_iter.next().unwrap();
            }
            let (right, right_s) = build_ast_from_additive_expression(inner_pair, script)?;
            s.merge(right_s);
            left = Some(get_ast_for_binary_expression(left, right, operator, script));
        } else {
            break;
        }
    }
    Ok((left.unwrap(), s))
}

fn build_ast_from_additive_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let mut left = None;
    let mut s = Semantics::new_empty();
    let mut pair_iter = pair.into_inner();
    loop {
        if let Some(mut inner_pair) = pair_iter.next() {
            let mut operator = None;
            if inner_pair.as_rule() == Rule::additive_operator {
                operator = Some(match inner_pair.as_str() {
                    "+" => BinaryOperator::Add,
                    "-" => BinaryOperator::Subtract,
                    _ => {
                        return Err(get_unexpected_error(
                            "build_ast_from_additive_expression",
                            &inner_pair,
                        ))
                    }
                });
                inner_pair = pair_iter.next().unwrap();
            }
            let (right, right_s) = build_ast_from_multiplicative_expression(inner_pair, script)?;
            s.merge(right_s);
            left = Some(get_ast_for_binary_expression(left, right, operator, script));
        } else {
            break;
        }
    }
    Ok((left.unwrap(), s))
}

fn build_ast_from_multiplicative_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let mut left = None;
    let mut s = Semantics::new_empty();
    let mut pair_iter = pair.into_inner();
    loop {
        if let Some(mut inner_pair) = pair_iter.next() {
            let mut operator = None;
            if inner_pair.as_rule() == Rule::multiplicative_operator {
                operator = Some(match inner_pair.as_str() {
                    "*" => BinaryOperator::Multiply,
                    "/" => BinaryOperator::Divide,
                    "%" => BinaryOperator::Modulo,
                    _ => {
                        return Err(get_unexpected_error(
                            "build_ast_from_multiplicative_expression",
                            &inner_pair,
                        ))
                    }
                });
                inner_pair = pair_iter.next().unwrap();
            }
            let (right, right_s) = build_ast_from_exponentiation_expression(inner_pair, script)?;
            s.merge(right_s);
            left = Some(get_ast_for_binary_expression(left, right, operator, script));
        } else {
            break;
        }
    }
    Ok((left.unwrap(), s))
}

/// `a ** b ** c` — right-associative exponentiation (binds tighter than `*`).
fn build_ast_from_exponentiation_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let mut pair_iter = pair.into_inner();
    let base_pair = pair_iter.next().unwrap();
    let (base, mut s) = build_ast_from_unary_expression(base_pair, script)?;
    if let Some(exp_pair) = pair_iter.next() {
        // Right operand is itself an exponentiation_expression (right-assoc).
        let (rhs, rhs_s) = build_ast_from_exponentiation_expression(exp_pair, script)?;
        s.merge(rhs_s);
        Ok((
            get_ast_for_binary_expression(Some(base), rhs, Some(BinaryOperator::Exponent), script),
            s,
        ))
    } else {
        Ok((base, s))
    }
}

fn build_ast_from_unary_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut pair_iter = pair.into_inner();
    let first_pair = pair_iter.next().unwrap();

    Ok(
        if first_pair.as_rule() == Rule::postfix_expression
            || first_pair.as_rule() == Rule::postfix_expression__yield
        {
            build_ast_from_postfix_expression(first_pair, script)?
        } else {
            // first_pair is unary_operator - get the operator string directly
            let operator_str = first_pair.as_str();
            match operator_str {
                "++" | "--" => {
                    let u_pair = pair_iter.next().unwrap();
                    let u_pair_meta = get_meta(&u_pair, script);
                    let (u, u_s) = build_ast_from_unary_expression(u_pair, script)?;
                    if u_s.is_valid_simple_assignment_target.is_false() {
                        return Err(get_validation_error_with_meta(
                            "Invalid expression for prefix operator".to_string(),
                            AstBuilderValidationErrorType::ReferenceError,
                            u_pair_meta,
                        ));
                    } else {
                        (
                            ExpressionType::UpdateExpression {
                                meta,
                                operator: match operator_str {
                                    "++" => UpdateOperator::PlusPlus,
                                    "--" => UpdateOperator::MinusMinus,
                                    _ => unreachable!(),
                                },
                                argument: Box::new(u),
                                prefix: true,
                            },
                            u_s,
                        )
                    }
                }
                _ => {
                    let (u, u_s) =
                        build_ast_from_unary_expression(pair_iter.next().unwrap(), script)?;
                    (
                        ExpressionType::UnaryExpression {
                            meta,
                            operator: match operator_str {
                                "delete" => {
                                    if let ExpressionType::ExpressionWhichCanBePattern(
                                        ExpressionPatternType::Identifier(id),
                                    ) = &u
                                    {
                                        return Err(get_validation_error_with_meta(
                                            format!(
                                                "Cannot delete identifier reference: {}",
                                                id.name
                                            ),
                                            AstBuilderValidationErrorType::SyntaxError,
                                            id.meta.clone(),
                                        ));
                                    }
                                    UnaryOperator::Delete
                                }
                                "void" => UnaryOperator::Void,
                                "typeof" => UnaryOperator::TypeOf,
                                "+" => UnaryOperator::Plus,
                                "-" => UnaryOperator::Minus,
                                "~" => UnaryOperator::BitwiseNot,
                                "!" => UnaryOperator::LogicalNot,
                                "await" => UnaryOperator::Await,
                                _ => {
                                    return Err(get_unexpected_error(
                                        "build_ast_from_unary_expression:2",
                                        &first_pair,
                                    ))
                                }
                            },
                            argument: Box::new(u),
                        },
                        u_s,
                    )
                }
            }
        },
    )
}

fn build_ast_from_postfix_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut pair_iter = pair.into_inner();
    let lhs_pair = pair_iter.next().unwrap();
    let (lhs, lhs_s) = build_ast_from_left_hand_side_expression(lhs_pair, script)?;
    if lhs_s.is_valid_simple_assignment_target.is_false() {
        Err(get_validation_error_with_meta(
            "Invalid expression for postfix operator".to_string(),
            AstBuilderValidationErrorType::ReferenceError,
            lhs.get_meta().clone(),
        ))
    } else {
        Ok((
            if let Some(op_pair) = pair_iter.next() {
                ExpressionType::UpdateExpression {
                    meta,
                    operator: match op_pair.as_str() {
                        "++" => UpdateOperator::PlusPlus,
                        "--" => UpdateOperator::MinusMinus,
                        _ => {
                            return Err(get_unexpected_error(
                                "build_ast_from_postfix_expression",
                                &op_pair,
                            ))
                        }
                    },
                    argument: Box::new(lhs),
                    prefix: false,
                }
            } else {
                lhs
            },
            lhs_s,
        ))
    }
}

fn get_binding_identifier_data(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(IdentifierData, Semantics), JsRuleError> {
    let mut id = get_identifier_data(pair, script);
    if id.name == "arguments" || id.name == "eval" || id.name == "yield" {
        Err(get_validation_error_with_meta(
            format!("Invalid binding identifier: {}", id.name),
            AstBuilderValidationErrorType::SyntaxError,
            id.meta,
        ))
    } else {
        id.is_binding_identifier = true;
        let mut s = Semantics::new_empty();
        s.bound_names.push(id.clone());
        Ok((id, s))
    }
}

fn get_identifier_data(pair: Pair<Rule>, script: &Rc<String>) -> IdentifierData {
    IdentifierData {
        meta: get_meta(&pair, script),
        name: pair.as_str().trim().to_string(),
        is_binding_identifier: false,
    }
}

fn build_ast_from_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut node_children: Vec<Box<ExpressionType>> = vec![];
    let mut s = Semantics::new_empty();
    for inner_pair in pair.into_inner() {
        let (a, a_s) = build_ast_from_assignment_expression(inner_pair, script)?;
        s.merge(a_s);
        node_children.push(Box::new(a));
    }
    Ok((
        ExpressionType::SequenceExpression {
            meta,
            expressions: node_children,
        },
        s,
    ))
}

fn build_ast_from_arguments(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(Vec<ExpressionOrSpreadElement>, Semantics), JsRuleError> {
    let mut arguments = vec![];
    let mut s = Semantics::new_empty();
    if let Some(argument_list_pair) = pair.into_inner().next() {
        for inner_pair in argument_list_pair.into_inner() {
            arguments.push(
                if inner_pair.as_rule() == Rule::rest_assignment_expression__in
                    || inner_pair.as_rule() == Rule::rest_assignment_expression__in_yield
                {
                    let (a, a_s) = build_ast_from_assignment_expression(
                        inner_pair.into_inner().next().unwrap(),
                        script,
                    )?;
                    s.merge(a_s);
                    ExpressionOrSpreadElement::SpreadElement(Box::new(a))
                } else {
                    let (a, a_s) = build_ast_from_assignment_expression(inner_pair, script)?;
                    s.merge(a_s);
                    ExpressionOrSpreadElement::Expression(Box::new(a))
                },
            );
        }
    }
    Ok((arguments, s))
}

fn build_ast_from_member_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut pair_iter = pair.into_inner();
    let pair_1 = pair_iter.next().unwrap();
    Ok(
        {
            let mut s = Semantics::new_empty();
            // ChittiOS: `new X(args)` is now a base that the trailing
            // `.member`/`[expr]`/tagged-template chain can extend (so
            // `new A(5).g()` parses); build the NewExpression base first, then
            // fall through to the shared chain loop below.
            let mut obj: ExpressionType = match pair_1.as_rule() {
                Rule::new_member_expression | Rule::new_member_expression__yield => {
                    let member_expression_pair = pair_1.into_inner().next().unwrap();
                    let arguments_pair = pair_iter.next().unwrap();
                    let (m, m_s) = build_ast_from_member_expression(member_expression_pair, script)?;
                    let (a, a_s) = build_ast_from_arguments(arguments_pair, script)?;
                    s.merge(m_s);
                    s.merge(a_s);
                    ExpressionType::NewExpression {
                        meta: meta.clone(),
                        callee: Box::new(m),
                        arguments: a,
                    }
                }
                Rule::super_property | Rule::super_property__yield => {
                    let super_pair = pair_1.into_inner().next().unwrap();
                    if super_pair.as_rule() == Rule::identifier_name {
                        ExpressionType::MemberExpression(
                            MemberExpressionType::SimpleMemberExpression {
                                meta,
                                object: ExpressionOrSuper::Super,
                                property: get_identifier_data(super_pair, script),
                            },
                        )
                    } else {
                        let (e, e_s) = build_ast_from_expression(super_pair, script)?;
                        s.merge(e_s);
                        ExpressionType::MemberExpression(
                            MemberExpressionType::ComputedMemberExpression {
                                meta,
                                object: ExpressionOrSuper::Super,
                                property: Box::new(e),
                            },
                        )
                    }
                }
                Rule::meta_property => {
                    let start = meta.start_index;
                    let end = meta.end_index;
                    ExpressionType::MetaProperty {
                        meta,
                        meta_object: IdentifierData {
                            meta: Meta {
                                start_index: start,
                                end_index: start + 3,
                                script: script.clone(),
                            },
                            name: "new".to_string(),
                            is_binding_identifier: false,
                        },
                        property: IdentifierData {
                            meta: Meta {
                                start_index: start + 4,
                                end_index: end,
                                script: script.clone(),
                            },
                            name: "target".to_string(),
                            is_binding_identifier: false,
                        },
                    }
                }
                Rule::primary_expression | Rule::primary_expression__yield => {
                    let (p, p_s) = build_ast_from_primary_expression(pair_1, script)?;
                    s.merge(p_s);
                    p
                }
                _ => {
                    return Err(get_unexpected_error(
                        "build_ast_from_member_expression:1",
                        &pair_1,
                    ))
                }
            };
            for pair in pair_iter {
                let second_meta = get_meta(&pair, script);
                let meta = Meta {
                    start_index: obj.get_meta().start_index,
                    end_index: second_meta.end_index,
                    script: script.clone(),
                };
                obj = match pair.as_rule() {
                    Rule::expression__in_yield | Rule::expression__in => {
                        let (st, st_s) = build_ast_from_expression(pair, script)?;
                        s.merge(st_s);
                        ExpressionType::MemberExpression(
                            MemberExpressionType::ComputedMemberExpression {
                                meta,
                                object: ExpressionOrSuper::Expression(Box::new(obj)),
                                property: Box::new(st),
                            },
                        )
                    }
                    Rule::identifier_name | Rule::private_name => ExpressionType::MemberExpression(
                        MemberExpressionType::SimpleMemberExpression {
                            meta,
                            object: ExpressionOrSuper::Expression(Box::new(obj)),
                            property: get_identifier_data(pair, script),
                        },
                    ),
                    Rule::optional_member => ExpressionType::OptionalChain {
                        meta,
                        object: Box::new(obj),
                        access: crate::parser::ast::OptionalAccess::Member(
                            pair.into_inner().next().unwrap().as_str().trim().to_string(),
                        ),
                    },
                    Rule::optional_index | Rule::optional_index__yield => {
                        let (e, e_s) =
                            build_ast_from_expression(pair.into_inner().next().unwrap(), script)?;
                        s.merge(e_s);
                        ExpressionType::OptionalChain {
                            meta,
                            object: Box::new(obj),
                            access: crate::parser::ast::OptionalAccess::Computed(Box::new(e)),
                        }
                    }
                    Rule::template_literal | Rule::template_literal__yield => {
                        let (template, t_s) = build_ast_from_template_literal(pair, script)?;
                        s.merge(t_s);
                        let quasi = if let ExpressionType::TemplateLiteral(data) = template {
                            data
                        } else {
                            return Err(get_validation_error_with_meta(
                                "Expected template literal".to_string(),
                                AstBuilderValidationErrorType::SyntaxError,
                                meta.clone(),
                            ));
                        };
                        ExpressionType::TaggedTemplateExpression {
                            meta,
                            tag: Box::new(obj),
                            quasi,
                        }
                    }
                    _ => {
                        return Err(get_unexpected_error(
                            "build_ast_from_member_expression:2",
                            &pair,
                        ))
                    }
                };
            }
            (obj, s)
        },
    )
}

fn build_ast_from_primary_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let inner_pair = pair.into_inner().next().unwrap();
    let meta = get_meta(&inner_pair, script);
    Ok(match inner_pair.as_rule() {
        Rule::identifier_reference | Rule::identifier_reference__yield => {
            let (i, i_s) = build_ast_from_identifier_reference(inner_pair, script)?;
            (i, i_s)
        }
        Rule::literal => (
            ExpressionType::Literal(build_ast_from_literal(inner_pair, script)?),
            Semantics::new_empty(),
        ),
        Rule::this_exp => (
            ExpressionType::ThisExpression { meta },
            Semantics::new_empty(),
        ),
        Rule::import_meta => (
            ExpressionType::ImportMeta { meta },
            Semantics::new_empty(),
        ),
        Rule::import_call | Rule::import_call__yield => {
            let arg_pair = inner_pair.into_inner().next().ok_or_else(|| {
                get_validation_error_with_meta(
                    "import() requires a specifier".to_string(),
                    AstBuilderValidationErrorType::SyntaxError,
                    meta.clone(),
                )
            })?;
            let (a, a_s) = build_ast_from_assignment_expression(arg_pair, script)?;
            (
                ExpressionType::ImportCall {
                    meta,
                    argument: Box::new(a),
                },
                a_s,
            )
        }
        Rule::array_literal | Rule::array_literal__yield => {
            let (a, a_s) = build_ast_from_array_literal(inner_pair, script)?;
            (a, a_s)
        }
        Rule::object_literal | Rule::object_literal__yield => {
            build_ast_from_object_literal(inner_pair, script)?
        } /* is_valid_simple_assignment_target&is_function_definition&is_identifier_ref=false */
        Rule::generator_expression => {
            let (f, f_s) =
                build_ast_from_generator_declaration_or_generator_expression(inner_pair, script)?;
            (ExpressionType::FunctionOrGeneratorExpression(f), f_s)
        } /* is_valid_simple_assignment_target&is_identifier_ref=false */
        Rule::function_expression => {
            let (f, f_s) =
                build_ast_from_function_declaration_or_function_expression(inner_pair, script)?;
            (ExpressionType::FunctionOrGeneratorExpression(f), f_s)
        }
        Rule::class_expression | Rule::class_expression__yield => {
            let (c, c_s) = build_ast_from_class_expression(inner_pair, script)?;
            (ExpressionType::ClassExpression(c), c_s)
        } /* is_valid_simple_assignment_target&is_identifier_ref=false */
        Rule::regular_expression_literal => {
            // ChittiOS: `/pattern/flags` → a RegExp literal. The pest text is the
            // whole literal incl. the slashes; split on the last `/`.
            let meta = get_meta(&inner_pair, script);
            let text = inner_pair.as_str();
            let inner = text.strip_prefix('/').unwrap_or(text);
            let (pattern, flags) = match inner.rfind('/') {
                Some(i) => (inner[..i].to_string(), inner[i + 1..].to_string()),
                None => (inner.to_string(), String::new()),
            };
            // Parse-phase syntax validation for named group specifiers
            // `(?<name>…)` and named backreferences `\k<name>` (a `SyntaxError`
            // in the spec). Gated to patterns that actually use these
            // constructs, so ordinary regexes are never rejected.
            if let Err(msg) = validate_regexp_named_groups(&pattern, &flags) {
                return Err(get_validation_error_with_meta(
                    msg,
                    AstBuilderValidationErrorType::SyntaxError,
                    meta,
                ));
            }
            // Structural early-error validation (bad/duplicate flags, quantifier
            // with no atom, invalid class ranges, `u`-mode escape strictness,
            // quantified lookbehind/lookahead, …) → a parse-time `SyntaxError`.
            if let Err(msg) = crate::runner::std_lib::regexp::validate(&pattern, &flags) {
                return Err(get_validation_error_with_meta(
                    msg,
                    AstBuilderValidationErrorType::SyntaxError,
                    meta,
                ));
            }
            (
                ExpressionType::Literal(LiteralData {
                    meta,
                    value: LiteralType::RegExpLiteral(crate::parser::ast::RegExpLiteralData {
                        pattern,
                        flags,
                    }),
                }),
                Semantics::new_empty(),
            )
        } /* is_valid_simple_assignment_target&is_function_definition&is_identifier_ref=false */
        Rule::template_literal | Rule::template_literal__yield => {
            build_ast_from_template_literal(inner_pair, script)?
        } /* is_valid_simple_assignment_target&is_function_definition&is_identifier_ref=false */
        Rule::cover_parenthesized_expression_and_arrow_parameter_list
        | Rule::cover_parenthesized_expression_and_arrow_parameter_list__yield => {
            build_ast_from_cover_parenthesized_expression_and_arrow_parameter_list(
                inner_pair, script,
            )?
        }
        _ => {
            return Err(get_unexpected_error(
                "build_ast_from_primary_expression",
                &inner_pair,
            ))
        }
    })
}

/// Validate the named-group syntax of a regular-expression pattern at parse
/// time. Returns `Err` for the `SyntaxError` cases the spec mandates for
/// `GroupSpecifier` `(?<name>…)` and named backreferences `\k<name>`:
/// empty/duplicate/ill-formed group names, unterminated `(?<…`, and — when the
/// pattern uses named groups or the `u` flag — a `\k` that is not a well-formed
/// reference to a declared name. Patterns without these constructs always pass,
/// so ordinary regexes are unaffected.
fn validate_regexp_named_groups(pattern: &str, flags: &str) -> Result<(), String> {
    let chars: Vec<char> = pattern.chars().collect();
    let unicode = flags.contains('u') || flags.contains('v');

    // A valid group-name start / continue code point. `\u`-escapes are accepted
    // optimistically (a stricter check would decode them).
    fn is_name_start(c: char) -> bool {
        c == '$' || c == '_' || c.is_alphabetic()
    }
    fn is_name_continue(c: char) -> bool {
        c == '$' || c == '_' || c == '\u{200C}' || c == '\u{200D}' || c.is_alphanumeric()
    }

    // Decode a unicode escape whose leading `\u` has already been consumed;
    // `*i` points at `{` or the first of four hex digits. Handles `\u{HHHH}`
    // (with range check) and `\uHHHH`, combining a high+low surrogate pair into
    // its astral code point. Advances `*i` past the escape.
    fn read_unicode_escape(chars: &[char], i: &mut usize) -> Result<u32, String> {
        fn hex4(chars: &[char], i: &mut usize) -> Result<u32, String> {
            let mut v = 0u32;
            for _ in 0..4 {
                let d = chars
                    .get(*i)
                    .and_then(|c| c.to_digit(16))
                    .ok_or_else(|| "Invalid unicode escape".to_string())?;
                v = v * 16 + d;
                *i += 1;
            }
            Ok(v)
        }
        if chars.get(*i) == Some(&'{') {
            *i += 1;
            let mut v = 0u32;
            let mut any = false;
            while let Some(&c) = chars.get(*i) {
                if c == '}' {
                    break;
                }
                let d = c.to_digit(16).ok_or_else(|| "Invalid unicode escape".to_string())?;
                v = v * 16 + d;
                any = true;
                if v > 0x10_FFFF {
                    return Err("Code point out of range".to_string());
                }
                *i += 1;
            }
            if !any || chars.get(*i) != Some(&'}') {
                return Err("Invalid unicode escape".to_string());
            }
            *i += 1;
            return Ok(v);
        }
        let hi = hex4(chars, i)?;
        // Combine a surrogate pair `\uD800-DBFF \uDC00-DFFF`.
        if (0xD800..=0xDBFF).contains(&hi)
            && chars.get(*i) == Some(&'\\')
            && chars.get(*i + 1) == Some(&'u')
        {
            let save = *i;
            *i += 2;
            let lo = hex4(chars, i)?;
            if (0xDC00..=0xDFFF).contains(&lo) {
                return Ok(0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00));
            }
            *i = save; // not a valid low surrogate; leave the lone (invalid) hi
        }
        Ok(hi)
    }

    // Read a `(?<name>` specifier starting just after `(?<`. Returns the name.
    // `i` points at the first char of the name; on success it is advanced past
    // the closing `>`.
    fn read_group_name(chars: &[char], i: &mut usize) -> Result<String, String> {
        let mut name = String::new();
        let mut first = true;
        loop {
            match chars.get(*i) {
                None => return Err("Unterminated named group".to_string()),
                Some('>') => {
                    *i += 1;
                    break;
                }
                Some('\\') => {
                    // Only `\u` escapes are legal within a group name, and the
                    // decoded code point must itself be a valid identifier
                    // start/continue char.
                    if chars.get(*i + 1) != Some(&'u') {
                        return Err("Invalid escape in group name".to_string());
                    }
                    *i += 2;
                    let cp = read_unicode_escape(chars, i)?;
                    let c = char::from_u32(cp).ok_or_else(|| "Invalid code point".to_string())?;
                    if first {
                        if !is_name_start(c) {
                            return Err("Invalid group name start".to_string());
                        }
                        first = false;
                    } else if !is_name_continue(c) {
                        return Err("Invalid group name character".to_string());
                    }
                    name.push(c);
                    continue;
                }
                Some(&c) => {
                    if first {
                        if !is_name_start(c) {
                            return Err("Invalid group name start".to_string());
                        }
                        first = false;
                    } else if !is_name_continue(c) {
                        return Err("Invalid group name character".to_string());
                    }
                    name.push(c);
                    *i += 1;
                }
            }
        }
        if name.is_empty() {
            return Err("Empty group name".to_string());
        }
        Ok(name)
    }

    // Pass 1: collect + validate declared group names.
    let mut declared: Vec<String> = Vec::new();
    let mut in_class = false;
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c == '\\' {
            i += 2; // skip escaped char
            continue;
        }
        if in_class {
            if c == ']' {
                in_class = false;
            }
            i += 1;
            continue;
        }
        if c == '[' {
            in_class = true;
            i += 1;
            continue;
        }
        // `(?<` group specifier — but not lookbehind `(?<=` / `(?<!`.
        if c == '('
            && chars.get(i + 1) == Some(&'?')
            && chars.get(i + 2) == Some(&'<')
            && !matches!(chars.get(i + 3), Some('=') | Some('!'))
        {
            i += 3;
            let name = read_group_name(&chars, &mut i)?;
            if declared.contains(&name) {
                return Err(format!("Duplicate regexp group name '{}'", name));
            }
            declared.push(name);
            continue;
        }
        // `(?…` that is not `(?:` / `(?=` / `(?!` / `(?<…` must be a
        // regexp-modifiers group `(?ims-ims:…)` (ES2025). Validate its
        // early-error rules; anything else after `(?` is a SyntaxError.
        // Ordinary group forms above are skipped, so real-world regexes
        // that never use modifiers are unaffected.
        if c == '('
            && chars.get(i + 1) == Some(&'?')
            && !matches!(chars.get(i + 2), Some(':') | Some('=') | Some('!') | Some('<') | None)
        {
            i += 2;
            validate_regexp_modifiers(&chars, &mut i)?;
            continue;
        }
        i += 1;
    }

    let has_named = !declared.is_empty();

    // Pass 2: validate `\k<name>` backreferences. In a pattern with named
    // groups (or under the `u` flag) `\k` MUST be a reference to a declared
    // name; otherwise it is a legacy identity escape and is left alone.
    if has_named || unicode {
        let mut in_class = false;
        let mut i = 0usize;
        while i < chars.len() {
            let c = chars[i];
            if c == '\\' {
                if chars.get(i + 1) == Some(&'k') {
                    let mut j = i + 2;
                    if chars.get(j) != Some(&'<') {
                        return Err("\\k not followed by <name>".to_string());
                    }
                    j += 1;
                    let name = read_group_name(&chars, &mut j)?;
                    if !declared.contains(&name) {
                        return Err(format!("Reference to undeclared group name '{}'", name));
                    }
                    i = j;
                    continue;
                }
                i += 2;
                continue;
            }
            if in_class {
                if c == ']' {
                    in_class = false;
                }
                i += 1;
                continue;
            }
            if c == '[' {
                in_class = true;
            }
            i += 1;
        }
    }

    Ok(())
}

/// Validate an ES2025 regexp-modifiers group `(?ims-ims:…)`. `*i` points just
/// past `(?`; on success it is advanced past the `:`. Early errors (spec
/// `Atom :: ( ? RegularExpressionFlags [- RegularExpressionFlags] : … )`):
/// a flag other than `i`/`m`/`s`, a duplicate flag within either list, a flag
/// present in both lists, and the dash form with both lists empty.
fn validate_regexp_modifiers(chars: &[char], i: &mut usize) -> Result<(), String> {
    let mut add: Vec<char> = Vec::new();
    let mut remove: Vec<char> = Vec::new();
    let mut has_dash = false;
    loop {
        match chars.get(*i) {
            None => return Err("Unterminated regexp modifiers group".to_string()),
            Some(':') => {
                *i += 1;
                break;
            }
            Some('-') if !has_dash => {
                has_dash = true;
                *i += 1;
            }
            Some(&c) => {
                if !matches!(c, 'i' | 'm' | 's') {
                    return Err(format!("Invalid regexp modifier '{}'", c));
                }
                let list = if has_dash { &mut remove } else { &mut add };
                if list.contains(&c) {
                    return Err(format!("Duplicate regexp modifier '{}'", c));
                }
                list.push(c);
                *i += 1;
            }
        }
    }
    if has_dash && add.is_empty() && remove.is_empty() {
        return Err("Regexp modifiers group with both lists empty".to_string());
    }
    if add.iter().any(|c| remove.contains(c)) {
        return Err("Regexp modifier present in both add and remove lists".to_string());
    }
    Ok(())
}

fn build_ast_from_cover_parenthesized_expression_and_arrow_parameter_list(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let inner_pair = expect_inner(pair, "build_ast_from_cover_parenthesized_expression_and_arrow_parameter_list", script)?;
    build_ast_from_expression(inner_pair, script)
}

fn build_ast_from_literal(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<LiteralData, JsRuleError> {
    let inner_pair = expect_inner(pair, "build_ast_from_literal", script)?;
    let meta = get_meta(&inner_pair, script);
    Ok(match inner_pair.as_rule() {
        Rule::null_literal => LiteralData {
            meta,
            value: LiteralType::NullLiteral,
        },
        Rule::numeric_literal => {
            // Peek the inner rule: a `bigint_literal` (…`n`) yields a BigInt
            // literal; anything else is an ordinary Number literal.
            let n_pair = inner_pair.into_inner().next().unwrap();
            if n_pair.as_rule() == Rule::bigint_literal {
                LiteralData {
                    meta,
                    value: build_ast_from_bigint_literal(n_pair)?,
                }
            } else {
                LiteralData {
                    meta,
                    value: LiteralType::NumberLiteral(build_ast_from_numeric_literal_inner(n_pair)?),
                }
            }
        }
        Rule::string_literal => build_ast_from_string_literal(inner_pair, script)?,
        Rule::boolean_literal => {
            let bool = inner_pair.as_str();
            LiteralData {
                meta,
                value: LiteralType::BooleanLiteral(bool == "true"),
            }
        }
        _ => return Err(get_unexpected_error("build_ast_from_literal", &inner_pair)),
    })
}

/// Post-parse Annex-B strict-mode gate. In strict-mode code a
/// LegacyOctalIntegerLiteral / NonOctalDecimalIntegerLiteral (`010`, `08`) and a
/// LegacyOctal / NonOctalDecimal string escape (`\1`, `\052`, `\8`) are Syntax
/// Errors, even though the (mode-free) PEG grammar accepts them. Strict mode is
/// propagated inward: the top-level program is strict when `initial_strict` (the
/// host said so) or its directive prologue holds `"use strict"`, and any
/// function whose own directive prologue holds `"use strict"` is strict for its
/// entire body (which includes that prologue — so an octal-bearing directive
/// preceding the `"use strict"` is rejected too, per B.1.2).
fn check_strict_legacy_octal(
    pairs: Pairs<Rule>,
    initial_strict: bool,
    script: &Rc<String>,
) -> Result<(), JsRuleError> {
    // The top-level program body is a `statement_list`; its leading string
    // directives form the program's directive prologue.
    let mut program_strict = initial_strict;
    for pair in pairs.clone() {
        if pair.as_rule() == Rule::statement_list && statement_list_is_strict(pair) {
            program_strict = true;
        }
    }
    for pair in pairs {
        check_pair_strict_legacy_octal(pair, program_strict, script)?;
    }
    Ok(())
}

/// Recursive worker for [`check_strict_legacy_octal`]. `strict` is the effective
/// strictness of the scope enclosing `pair`.
fn check_pair_strict_legacy_octal(
    pair: Pair<Rule>,
    strict: bool,
    script: &Rc<String>,
) -> Result<(), JsRuleError> {
    match pair.as_rule() {
        Rule::legacy_octal_like_integer_literal => {
            if strict {
                return Err(get_validation_error_with_meta(
                    "Legacy octal / non-octal-decimal integer literals are not allowed in strict mode".to_string(),
                    AstBuilderValidationErrorType::SyntaxError,
                    get_meta(&pair, script),
                ));
            }
        }
        Rule::string_literal => {
            // `string_literal` is atomic (no sub-tokens); inspect its raw text.
            if strict && string_has_strict_forbidden_escape(pair.as_str()) {
                return Err(get_validation_error_with_meta(
                    "Legacy octal / non-octal-decimal string escapes are not allowed in strict mode".to_string(),
                    AstBuilderValidationErrorType::SyntaxError,
                    get_meta(&pair, script),
                ));
            }
        }
        Rule::function_body | Rule::function_body__yield => {
            let inner_strict = strict || function_body_is_strict(pair.clone());
            for child in pair.into_inner() {
                check_pair_strict_legacy_octal(child, inner_strict, script)?;
            }
        }
        _ => {
            for child in pair.into_inner() {
                check_pair_strict_legacy_octal(child, strict, script)?;
            }
        }
    }
    Ok(())
}

/// True for any of the four grammar `statement_list` variants (plain / return /
/// yield / yield-return).
fn is_statement_list_rule(rule: Rule) -> bool {
    matches!(
        rule,
        Rule::statement_list
            | Rule::statement_list__return
            | Rule::statement_list__yield
            | Rule::statement_list__yield_return
    )
}

/// True if a `function_body` opens a strict scope via a `"use strict"` directive
/// in its own directive prologue. The body wraps a `statement_list__return`
/// (or the yield variant), reached through the silent `function_statement_list`.
fn function_body_is_strict(function_body: Pair<Rule>) -> bool {
    for inner in function_body.into_inner() {
        if is_statement_list_rule(inner.as_rule()) {
            return statement_list_is_strict(inner);
        }
    }
    false
}

/// Scan the directive prologue of a `statement_list` (its leading string-literal
/// expression statements) for an exact `use strict` directive.
fn statement_list_is_strict(statement_list: Pair<Rule>) -> bool {
    for stmt in statement_list.into_inner() {
        let text = stmt.as_str().trim();
        match directive_string_value(text) {
            Some(directive) => {
                if directive == "use strict" {
                    return true;
                }
                // Another leading string directive — keep scanning the prologue.
            }
            // First non-string-directive statement ends the prologue.
            None => return false,
        }
    }
    false
}

/// If `text` is a lone string-literal expression statement (a directive), return
/// its raw inner text (between the quotes, escapes NOT decoded — a `"use strict"`
/// directive must be spelled literally). Otherwise `None`.
fn directive_string_value(text: &str) -> Option<&str> {
    // Drop an optional trailing `;` and surrounding whitespace.
    let t = text.trim().strip_suffix(';').unwrap_or(text).trim();
    let bytes = t.as_bytes();
    if bytes.len() < 2 {
        return None;
    }
    let quote = bytes[0];
    if (quote != b'"' && quote != b'\'') || bytes[bytes.len() - 1] != quote {
        return None;
    }
    let inner = &t[1..t.len() - 1];
    // A genuine single-string directive has no unescaped closing quote inside.
    if inner.contains(quote as char) {
        return None;
    }
    Some(inner)
}

/// True if a string-literal's raw text (quotes included) contains an escape that
/// is a Syntax Error in strict mode: any `\1`..`\9`, or `\0` immediately
/// followed by a decimal digit (`\00`, `\08`). A bare `\0` is still allowed.
fn string_has_strict_forbidden_escape(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '\\' {
            i += 1;
            continue;
        }
        match chars.get(i + 1) {
            Some('0') => {
                if chars.get(i + 2).map_or(false, |c| c.is_ascii_digit()) {
                    return true;
                }
                i += 2; // `\0` alone: allowed, skip both chars.
            }
            Some(c) if ('1'..='9').contains(c) => return true,
            Some(_) => i += 2, // any other escape (incl. `\\`) — skip the pair.
            None => i += 1,
        }
    }
    false
}

fn build_ast_from_string_literal(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<LiteralData, JsRuleError> {
    let meta = get_meta(&pair, script);
    let s = pair.as_str();
    Ok(LiteralData {
        meta,
        value: LiteralType::StringLiteral(cook_string_literal(&s[1..s.len() - 1])),
    })
}

/// Decode the escape sequences in a `'…'`/`"…"` string-literal body (the
/// surrounding quotes already stripped) into their runtime code points:
/// `\n \r \t \b \f \v \0`, `\xHH`, `\uHHHH` / `\u{…}` (combining a
/// `\uD800-DBFF \uDC00-DFFF` pair into its astral code point), line
/// continuations (`\` + line terminator → nothing), and identity escapes
/// (`\' \" \\ \/ …` → the character). Mirrors `process_template_escapes` for
/// ordinary strings, which the grammar otherwise stored verbatim.
fn cook_string_literal(body: &str) -> String {
    let chars: Vec<char> = body.chars().collect();
    let mut out = String::with_capacity(body.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c != '\\' {
            out.push(c);
            i += 1;
            continue;
        }
        i += 1; // consume backslash
        let e = match chars.get(i) {
            Some(&e) => e,
            None => {
                out.push('\\');
                break;
            }
        };
        i += 1;
        match e {
            'n' => out.push('\n'),
            't' => out.push('\t'),
            'r' => out.push('\r'),
            'b' => out.push('\u{0008}'),
            'f' => out.push('\u{000C}'),
            'v' => out.push('\u{000B}'),
            '\\' => out.push('\\'),
            '\'' => out.push('\''),
            '"' => out.push('"'),
            '`' => out.push('`'),
            '$' => out.push('$'),
            // Annex-B LegacyOctalEscapeSequence: 1–3 octal digits. `\0` not
            // followed by an octal digit is the plain NUL escape. A leading
            // digit of 0–3 admits up to two more octal digits (max `\377`);
            // 4–7 admits only one more (max `\77`).
            '0'..='7' => {
                let mut v = e as u32 - '0' as u32;
                let max_more = if e <= '3' { 2 } else { 1 };
                let mut more = 0;
                while more < max_more {
                    match chars.get(i) {
                        Some(&d @ '0'..='7') => {
                            v = v * 8 + (d as u32 - '0' as u32);
                            i += 1;
                            more += 1;
                        }
                        _ => break,
                    }
                }
                out.push(core::char::from_u32(v).unwrap_or('\u{FFFD}'));
            }
            // Annex-B NonOctalDecimalEscapeSequence: `\8`/`\9` → the chars "8"/"9".
            '8' | '9' => out.push(e),
            'x' => {
                let mut v = 0u32;
                let mut n = 0;
                while n < 2 {
                    if let Some(d) = chars.get(i).and_then(|c| c.to_digit(16)) {
                        v = v * 16 + d;
                        i += 1;
                        n += 1;
                    } else {
                        break;
                    }
                }
                if n == 2 {
                    if let Some(ch) = core::char::from_u32(v) {
                        out.push(ch);
                    }
                }
            }
            'u' => {
                if let Some(ch) = read_string_unicode_escape(&chars, &mut i) {
                    out.push(ch);
                }
            }
            // Line continuation: `\` followed by a line terminator yields nothing.
            '\n' | '\u{2028}' | '\u{2029}' => {}
            '\r' => {
                if chars.get(i) == Some(&'\n') {
                    i += 1;
                }
            }
            other => out.push(other),
        }
    }
    out
}

/// Read a `\u…` escape (the leading `\u` already consumed, `*i` at the char
/// after `u`): `\u{HHHH}` or `\uHHHH`, combining a surrogate pair into its
/// astral code point. A lone surrogate degrades to U+FFFD (Rust `String`
/// cannot hold one). Advances `*i` past the escape.
fn read_string_unicode_escape(chars: &[char], i: &mut usize) -> Option<char> {
    if chars.get(*i) == Some(&'{') {
        *i += 1;
        let mut v = 0u32;
        while let Some(d) = chars.get(*i).and_then(|c| c.to_digit(16)) {
            v = v.saturating_mul(16).saturating_add(d);
            *i += 1;
        }
        if chars.get(*i) == Some(&'}') {
            *i += 1;
        }
        return core::char::from_u32(v);
    }
    let mut v = 0u32;
    for _ in 0..4 {
        if let Some(d) = chars.get(*i).and_then(|c| c.to_digit(16)) {
            v = v * 16 + d;
            *i += 1;
        } else {
            return None;
        }
    }
    if (0xD800..=0xDBFF).contains(&v) {
        // Try to combine with a following `\uXXXX` low surrogate.
        if chars.get(*i) == Some(&'\\') && chars.get(*i + 1) == Some(&'u') {
            let save = *i;
            *i += 2;
            let mut lo = 0u32;
            let mut ok = true;
            for _ in 0..4 {
                if let Some(d) = chars.get(*i).and_then(|c| c.to_digit(16)) {
                    lo = lo * 16 + d;
                    *i += 1;
                } else {
                    ok = false;
                    break;
                }
            }
            if ok && (0xDC00..=0xDFFF).contains(&lo) {
                let cp = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00);
                return core::char::from_u32(cp);
            }
            *i = save;
        }
        return Some('\u{FFFD}');
    }
    core::char::from_u32(v).or(Some('\u{FFFD}'))
}

/// Builds AST from a template literal.
/// Template literals can be:
///   - no_substitution_template: `hello world`
///   - template with substitutions: `hello ${name}!`
fn build_ast_from_template_literal(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut quasis: Vec<TemplateElementData> = vec![];
    let mut expressions: Vec<Box<ExpressionType>> = vec![];
    let mut s = Semantics::new_empty();

    for inner_pair in pair.into_inner() {
        match inner_pair.as_rule() {
            Rule::no_substitution_template => {
                // Simple template with no interpolations: `hello`
                let raw_str = inner_pair.as_str();
                // Remove the backticks
                let content = &raw_str[1..raw_str.len() - 1];
                let quasi_meta = get_meta(&inner_pair, script);
                quasis.push(TemplateElementData {
                    meta: quasi_meta,
                    tail: true,
                    cooked_value: process_template_escapes(content),
                    raw_value: content.to_string(),
                });
            }
            Rule::template_head => {
                // Head of template with substitutions: `hello ${
                let raw_str = inner_pair.as_str();
                // Remove ` from start and ${ from end
                let content = &raw_str[1..raw_str.len() - 2];
                let quasi_meta = get_meta(&inner_pair, script);
                quasis.push(TemplateElementData {
                    meta: quasi_meta,
                    tail: false,
                    cooked_value: process_template_escapes(content),
                    raw_value: content.to_string(),
                });
            }
            Rule::template_middle => {
                // Middle part between substitutions: }...${
                let raw_str = inner_pair.as_str();
                // Remove } from start and ${ from end
                let content = &raw_str[1..raw_str.len() - 2];
                let quasi_meta = get_meta(&inner_pair, script);
                quasis.push(TemplateElementData {
                    meta: quasi_meta,
                    tail: false,
                    cooked_value: process_template_escapes(content),
                    raw_value: content.to_string(),
                });
            }
            Rule::template_tail => {
                // End of template with substitutions: }...`
                let raw_str = inner_pair.as_str();
                // Remove } from start and ` from end
                let content = &raw_str[1..raw_str.len() - 1];
                let quasi_meta = get_meta(&inner_pair, script);
                quasis.push(TemplateElementData {
                    meta: quasi_meta,
                    tail: true,
                    cooked_value: process_template_escapes(content),
                    raw_value: content.to_string(),
                });
            }
            Rule::expression__in | Rule::expression__in_yield => {
                let (expr, expr_s) = build_ast_from_expression(inner_pair, script)?;
                s.merge(expr_s);
                expressions.push(Box::new(expr));
            }
            _ => {
                return Err(get_unexpected_error(
                    "build_ast_from_template_literal",
                    &inner_pair,
                ))
            }
        }
    }

    Ok((
        ExpressionType::TemplateLiteral(TemplateLiteralData {
            meta,
            quasis,
            expressions,
        }),
        s,
    ))
}

/// Process escape sequences in template literal content.
fn process_template_escapes(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('t') => result.push('\t'),
                Some('\\') => result.push('\\'),
                Some('`') => result.push('`'),
                Some('$') => result.push('$'),
                Some('0') => result.push('\0'),
                Some('\'') => result.push('\''),
                Some('"') => result.push('"'),
                Some('x') => {
                    // Hex escape: \xHH
                    let mut hex = String::new();
                    for _ in 0..2 {
                        if let Some(&c) = chars.peek() {
                            if c.is_ascii_hexdigit() {
                                hex.push(chars.next().unwrap());
                            }
                        }
                    }
                    if hex.len() == 2 {
                        if let Ok(val) = u8::from_str_radix(&hex, 16) {
                            result.push(val as char);
                        }
                    }
                }
                Some('u') => {
                    // Unicode escape: \uHHHH or \u{H...}
                    if chars.peek() == Some(&'{') {
                        chars.next(); // consume '{'
                        let mut hex = String::new();
                        while let Some(&c) = chars.peek() {
                            if c == '}' {
                                chars.next();
                                break;
                            }
                            if c.is_ascii_hexdigit() {
                                hex.push(chars.next().unwrap());
                            } else {
                                break;
                            }
                        }
                        if let Ok(val) = u32::from_str_radix(&hex, 16) {
                            if let Some(ch) = char::from_u32(val) {
                                result.push(ch);
                            }
                        }
                    } else {
                        let mut hex = String::new();
                        for _ in 0..4 {
                            if let Some(&c) = chars.peek() {
                                if c.is_ascii_hexdigit() {
                                    hex.push(chars.next().unwrap());
                                }
                            }
                        }
                        if hex.len() == 4 {
                            if let Ok(val) = u16::from_str_radix(&hex, 16) {
                                result.push(char::from_u32(val as u32).unwrap_or('\u{FFFD}'));
                            }
                        }
                    }
                }
                Some(other) => {
                    // Unknown escape, just include the character
                    result.push(other);
                }
                None => {
                    // Trailing backslash
                    result.push('\\');
                }
            }
        } else {
            result.push(c);
        }
    }

    result
}

fn build_ast_from_str_numeric_literal(
    pair: Pair<Rule>,
) -> Result<ExtendedNumberLiteralType, JsRuleError> {
    let inner_pair = pair.into_inner().next().unwrap();
    Ok(match inner_pair.as_rule() {
        Rule::binary_integer_literal => {
            ExtendedNumberLiteralType::Std(get_ast_for_binary_integer_literal(inner_pair))
        }
        Rule::octal_integer_literal => {
            ExtendedNumberLiteralType::Std(get_ast_for_octal_integer_literal(inner_pair))
        }
        Rule::hex_integer_literal => {
            ExtendedNumberLiteralType::Std(get_ast_for_hex_integer_literal(inner_pair))
        }
        Rule::str_decimal_literal => build_ast_str_decimal_literal(inner_pair)?,
        _ => {
            return Err(get_unexpected_error(
                "build_ast_from_str_numeric_literal",
                &inner_pair,
            ))
        }
    })
}

fn build_ast_str_decimal_literal(
    pair: Pair<Rule>,
) -> Result<ExtendedNumberLiteralType, JsRuleError> {
    let mut pair_iter = pair.into_inner();
    let mut inner_pair = pair_iter.next().unwrap();
    let is_negative = if inner_pair.as_rule() == Rule::str_negative_op {
        inner_pair = pair_iter.next().unwrap();
        true
    } else {
        false
    };
    build_ast_str_unsigned_decimal_literal(inner_pair, is_negative)
}

fn build_ast_str_unsigned_decimal_literal(
    pair: Pair<Rule>,
    is_negative: bool,
) -> Result<ExtendedNumberLiteralType, JsRuleError> {
    // Infinity is a dedicated token (not an f64 parse of the word).
    for decimal_pair in pair.clone().into_inner() {
        if decimal_pair.as_rule() == Rule::str_decimal_literal_infinity {
            return Ok(if is_negative {
                ExtendedNumberLiteralType::NegativeInfinity
            } else {
                ExtendedNumberLiteralType::Infinity
            });
        }
    }
    let raw = pair.as_str().replace('_', "");
    let mut lit = classify_decimal_f64(raw.parse().unwrap_or(0.0));
    if is_negative {
        lit = match lit {
            NumberLiteralType::IntegerLiteral(i) => {
                if i == 0 {
                    // `-0` from a string numeric literal must be a negative zero.
                    NumberLiteralType::FloatLiteral(-0.0)
                } else {
                    NumberLiteralType::IntegerLiteral(-i)
                }
            }
            NumberLiteralType::FloatLiteral(f) => NumberLiteralType::FloatLiteral(-f),
        };
    }
    Ok(ExtendedNumberLiteralType::Std(lit))
}

fn get_ast_for_binary_integer_literal(pair: Pair<Rule>) -> NumberLiteralType {
    let s = pair.as_str()[2..].replace('_', "");
    NumberLiteralType::IntegerLiteral(i64::from_str_radix(&s, 2).unwrap_or(i64::MAX))
}

fn get_ast_for_octal_integer_literal(pair: Pair<Rule>) -> NumberLiteralType {
    let s = pair.as_str()[2..].replace('_', "");
    NumberLiteralType::IntegerLiteral(i64::from_str_radix(&s, 8).unwrap_or(i64::MAX))
}

fn get_ast_for_hex_integer_literal(pair: Pair<Rule>) -> NumberLiteralType {
    let s = pair.as_str()[2..].replace('_', "");
    NumberLiteralType::IntegerLiteral(i64::from_str_radix(&s, 16).unwrap_or(i64::MAX))
}

/// Annex-B B.1.1 legacy leading-zero integer (`0` + digits). If every digit
/// after the leading `0` is octal it is a LegacyOctalIntegerLiteral parsed
/// base-8 (`010` → 8, `077` → 63); if any digit is `8`/`9` it is a
/// NonOctalDecimalIntegerLiteral parsed base-10 (`08` → 8, `018` → 18). Long
/// runs that overflow `i64` fall back to lossy `f64`, matching JS double
/// semantics as elsewhere in this module.
fn get_ast_for_legacy_octal_like_integer_literal(pair: Pair<Rule>) -> NumberLiteralType {
    let s = pair.as_str();
    let digits = &s[1..]; // strip the leading `0`
    let (value, radix) = if digits.bytes().all(|b| (b'0'..=b'7').contains(&b)) {
        (digits, 8u32)
    } else {
        (s, 10u32)
    };
    let v = match i64::from_str_radix(value, radix) {
        Ok(v) => v,
        Err(_) => {
            // Base-10 overflow → parse as f64; base-8 overflow is astronomically
            // unlikely for a source literal but saturate defensively.
            if radix == 10 {
                value.parse::<f64>().unwrap_or(f64::INFINITY) as i64
            } else {
                i64::MAX
            }
        }
    };
    NumberLiteralType::IntegerLiteral(v)
}

/// Classify a parsed IEEE decimal into the AST's Integer vs Float form.
/// Exact finite integers in `i64` range stay `IntegerLiteral` (so `12e2` → 1200);
/// fractions, huge magnitudes (`1e308`), and non-finite values use `FloatLiteral`
/// so they are not truncated by `as i64` (the bug behind `1e-1 === 0`).
fn classify_decimal_f64(num: f64) -> NumberLiteralType {
    if num.is_finite() {
        let as_i = num as i64;
        if (as_i as f64) == num {
            return NumberLiteralType::IntegerLiteral(as_i);
        }
    }
    NumberLiteralType::FloatLiteral(num)
}

fn build_ast_decimal_literal(pair: Pair<Rule>) -> Result<NumberLiteralType, JsRuleError> {
    // Parse the *whole* lexeme via `f64` so `1.1e-1` matches `0.11` bit-exactly.
    // Composing mantissa × 10^exp in pieces (`1.1 * 0.1`) yields a different
    // float than the literal grammar requires — that broke a dozen test262
    // numeric cases and any page that compares scientific-notation constants.
    let raw = pair.as_str().replace('_', "");
    Ok(classify_decimal_f64(raw.parse::<f64>().unwrap_or(0.0)))
}

#[allow(dead_code)]
fn build_ast_from_numeric_literal(pair: Pair<Rule>) -> Result<NumberLiteralType, JsRuleError> {
    let inner_pair = pair.into_inner().next().unwrap();
    build_ast_from_numeric_literal_inner(inner_pair)
}

/// Build a Number literal from the already-unwrapped inner pair (one of the
/// integer forms or `decimal_literal`). `bigint_literal` is handled separately.
fn build_ast_from_numeric_literal_inner(
    inner_pair: Pair<Rule>,
) -> Result<NumberLiteralType, JsRuleError> {
    Ok(match inner_pair.as_rule() {
        Rule::binary_integer_literal => get_ast_for_binary_integer_literal(inner_pair),
        Rule::octal_integer_literal => get_ast_for_octal_integer_literal(inner_pair),
        Rule::hex_integer_literal => get_ast_for_hex_integer_literal(inner_pair),
        Rule::legacy_octal_like_integer_literal => {
            get_ast_for_legacy_octal_like_integer_literal(inner_pair)
        }
        Rule::decimal_literal => build_ast_decimal_literal(inner_pair)?,
        _ => {
            return Err(get_unexpected_error(
                "build_ast_from_numeric_literal",
                &inner_pair,
            ))
        }
    })
}

/// Parse a `bigint_literal` pair (e.g. `255n`, `0xFFn`, `0b101n`, `1_000n`) into
/// a `LiteralType::BigIntLiteral` holding the value's decimal string. Numeric
/// separators (`_`) and the trailing `n` are stripped; the radix prefix selects
/// the base.
fn build_ast_from_bigint_literal(pair: Pair<Rule>) -> Result<LiteralType, JsRuleError> {
    use num_bigint::BigInt;
    // `bigint_literal` is atomic (`@`), so it has no inner tokens — parse its
    // text. Strip the trailing `n`, then select radix by prefix.
    let raw = pair.as_str();
    let body = raw.strip_suffix('n').unwrap_or(raw);
    let (radix, digits): (u32, &str) = if let Some(rest) =
        body.strip_prefix("0b").or_else(|| body.strip_prefix("0B"))
    {
        (2, rest)
    } else if let Some(rest) = body.strip_prefix("0o").or_else(|| body.strip_prefix("0O")) {
        (8, rest)
    } else if let Some(rest) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        (16, rest)
    } else {
        (10, body)
    };
    let cleaned = digits.replace('_', "");
    let value = BigInt::parse_bytes(cleaned.as_bytes(), radix)
        .ok_or_else(|| get_unexpected_error("build_ast_from_bigint_literal", &pair))?;
    Ok(LiteralType::BigIntLiteral(value.to_string()))
}

fn build_ast_from_array_literal(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut arguments = vec![];
    let mut s = Semantics::new_empty();
    for inner_pair in pair.into_inner() {
        match inner_pair.as_rule() {
            Rule::elision => {
                for _ in 0..(inner_pair.as_str().matches(',').count()) {
                    arguments.push(None);
                }
            }
            Rule::assignment_expression__in | Rule::assignment_expression__in_yield => {
                let (a, a_s) = build_ast_from_assignment_expression(inner_pair, script)?;
                s.merge(a_s);
                arguments.push(Some(ExpressionOrSpreadElement::Expression(Box::new(a))));
            }
            Rule::spread_element | Rule::spread_element__yield => {
                let spread_inner = expect_inner(inner_pair, "spread_element", script)?;
                let (a, a_s) = build_ast_from_assignment_expression(spread_inner, script)?;
                s.merge(a_s);
                arguments.push(Some(ExpressionOrSpreadElement::SpreadElement(Box::new(a))));
            }
            _ => {
                return Err(get_unexpected_error(
                    "build_ast_from_array_literal",
                    &inner_pair,
                ))
            }
        }
    }
    Ok((
        ExpressionType::ArrayExpression {
            meta,
            elements: arguments,
        },
        s,
    ))
}

fn build_ast_from_object_literal(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut s = Semantics::new_empty();
    let mut properties = vec![];
    for property_pair in pair.into_inner() {
        let (p, p_s) = build_ast_from_property_definition(property_pair, script)?;
        s.merge(p_s);
        properties.push(p);
    }
    Ok((ExpressionType::ObjectExpression { meta, properties }, s))
}

fn build_ast_from_property_definition(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(PropertyData<Box<ExpressionType>>, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut s = Semantics::new_empty();
    let mut inner_pair_iter = pair.into_inner();
    let inner_pair = inner_pair_iter.next().unwrap();
    let p = match inner_pair.as_rule() {
        Rule::spread_property | Rule::spread_property__yield => {
            // `{ ...expr }` — store the spread expr as the value of a
            // Spread-kind property (the key is an unused placeholder).
            let expr_pair = inner_pair.into_inner().next().unwrap();
            let (a, a_s) = build_ast_from_assignment_expression(expr_pair, script)?;
            s.merge(a_s);
            PropertyData::new_with_any_expression_key(
                meta.clone(),
                ExpressionType::ThisExpression { meta },
                false,
                Box::new(a),
                PropertyKind::Spread,
                false,
                false,
            )
        }
        Rule::property_name | Rule::property_name__yield => {
            let (p, p_s, p_computed) = build_ast_from_property_name(inner_pair, script)?;
            let (a, a_s) =
                build_ast_from_assignment_expression(inner_pair_iter.next().unwrap(), script)?;
            s.merge(p_s).merge(a_s);
            PropertyData::new_with_any_expression_key(
                meta,
                p,
                p_computed,
                Box::new(a),
                PropertyKind::Init,
                false,
                false,
            )
        }
        Rule::cover_initialized_name | Rule::cover_initialized_name__yield => {
            let error = format!("Initialization is only possible for object destruction pattern not in object literal: {}", inner_pair.as_str());
            return Err(get_validation_error(
                error,
                AstBuilderValidationErrorType::SyntaxError,
                &inner_pair,
                &script,
            ));
        }
        Rule::method_definition | Rule::method_definition__yield => {
            let meta = get_meta(&inner_pair, script);
            let (m, m_s) = build_ast_from_method_definition(inner_pair, script)?;
            if m_s.has_direct_super.is_true() {
                return Err(get_validation_error_with_meta(
                    "Invalid reference to super".to_string(),
                    AstBuilderValidationErrorType::SyntaxError,
                    meta,
                ));
            }
            s.merge(m_s);
            m
        }
        Rule::identifier_reference | Rule::identifier_reference__yield => {
            let id = get_identifier_data(inner_pair, script);
            let id2 = id.clone();
            PropertyData::new_with_identifier_key(
                meta,
                id,
                Box::new(ExpressionPatternType::Identifier(id2).convert_to_expression()),
                PropertyKind::Init,
                false,
                true,
            )
        }
        _ => {
            return Err(get_unexpected_error(
                "build_ast_from_property_definition",
                &inner_pair,
            ))
        }
    };
    Ok((p, s))
}

fn build_ast_from_method_definition(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(PropertyData<Box<ExpressionType>>, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut s = Semantics::new_empty();
    let mut inner_iter = pair.into_inner();
    let inner_pair = inner_iter.next().unwrap();
    let m = match inner_pair.as_rule() {
        Rule::property_name | Rule::property_name__yield => {
            let (p, p_s, p_computed) = build_ast_from_property_name(inner_pair, script)?;
            let (fp, fp_s) = build_ast_from_formal_parameters(inner_iter.next().unwrap(), script)?;
            let (fb, fb_s) = build_ast_from_function_body(inner_iter.next().unwrap(), script)?;
            validate_bound_names_have_no_duplicates_and_also_not_present_in_var_declared_names_or_lexically_declared_names(&fp_s.bound_names,&vec![],&fb_s.lexically_declared_names)?;
            s.merge(p_s).merge(fp_s).merge(fb_s);
            let meta2 = meta.clone();
            PropertyData::new_with_any_expression_key(
                meta,
                p,
                p_computed,
                Box::new(ExpressionType::FunctionOrGeneratorExpression(
                    FunctionData {
                        meta: meta2,
                        id: None,
                        params: FormalParameters::new(fp),
                        body: Box::new(fb),
                        generator: false,
                        is_async: false,
                    },
                )),
                PropertyKind::Init,
                true,
                false,
            )
        }
        Rule::generator_method | Rule::generator_method__yield => {
            let mut inner_inner_iter = inner_pair.into_inner();
            let (p, p_s, p_computed) = build_ast_from_property_name(inner_inner_iter.next().unwrap(), script)?;
            let (fp, fp_s) =
                build_ast_from_formal_parameters(inner_inner_iter.next().unwrap(), script)?;
            let (fb, fb_s) =
                build_ast_from_generator_body(inner_inner_iter.next().unwrap(), script)?;
            if fb_s.has_direct_super.is_true() {
                return Err(get_validation_error_with_meta(
                    "Invalid reference to 'super'".to_string(),
                    AstBuilderValidationErrorType::SyntaxError,
                    meta.clone(),
                ));
            }
            validate_bound_names_have_no_duplicates_and_also_not_present_in_var_declared_names_or_lexically_declared_names(&fp_s.bound_names,&vec![],&fb_s.lexically_declared_names)?;
            s.merge(p_s).merge(fp_s).merge(fb_s);
            let meta2 = meta.clone();
            PropertyData::new_with_any_expression_key(
                meta,
                p,
                p_computed,
                Box::new(ExpressionType::FunctionOrGeneratorExpression(
                    FunctionData {
                        meta: meta2,
                        id: None,
                        params: FormalParameters::new(fp),
                        body: Box::new(fb),
                        generator: true,
                        is_async: false,
                    },
                )),
                PropertyKind::Init,
                true,
                false,
            )
        }
        Rule::async_method => {
            // async_kw ~ property_name ~ "(" params ")" "{" body "}"
            let mut inner_inner_iter = inner_pair.into_inner().filter(|p| p.as_rule() != Rule::async_kw);
            let (p, p_s, p_computed) = build_ast_from_property_name(inner_inner_iter.next().unwrap(), script)?;
            let (fp, fp_s) =
                build_ast_from_formal_parameters(inner_inner_iter.next().unwrap(), script)?;
            let (fb, fb_s) =
                build_ast_from_function_body(inner_inner_iter.next().unwrap(), script)?;
            validate_bound_names_have_no_duplicates_and_also_not_present_in_var_declared_names_or_lexically_declared_names(&fp_s.bound_names,&vec![],&fb_s.lexically_declared_names)?;
            s.merge(p_s).merge(fp_s).merge(fb_s);
            let meta2 = meta.clone();
            PropertyData::new_with_any_expression_key(
                meta,
                p,
                p_computed,
                Box::new(ExpressionType::FunctionOrGeneratorExpression(
                    FunctionData {
                        meta: meta2,
                        id: None,
                        params: FormalParameters::new(fp),
                        body: Box::new(fb),
                        generator: false,
                        is_async: true,
                    },
                )),
                PropertyKind::Init,
                true,
                false,
            )
        }
        Rule::async_generator_method => {
            // async_kw ~ "*" ~ property_name ~ "(" params ")" "{" gen_body "}"
            let mut inner_inner_iter = inner_pair.into_inner().filter(|p| p.as_rule() != Rule::async_kw);
            let (p, p_s, p_computed) = build_ast_from_property_name(inner_inner_iter.next().unwrap(), script)?;
            let (fp, fp_s) =
                build_ast_from_formal_parameters(inner_inner_iter.next().unwrap(), script)?;
            let (fb, fb_s) =
                build_ast_from_generator_body(inner_inner_iter.next().unwrap(), script)?;
            validate_bound_names_have_no_duplicates_and_also_not_present_in_var_declared_names_or_lexically_declared_names(&fp_s.bound_names,&vec![],&fb_s.lexically_declared_names)?;
            s.merge(p_s).merge(fp_s).merge(fb_s);
            let meta2 = meta.clone();
            PropertyData::new_with_any_expression_key(
                meta,
                p,
                p_computed,
                Box::new(ExpressionType::FunctionOrGeneratorExpression(
                    FunctionData {
                        meta: meta2,
                        id: None,
                        params: FormalParameters::new(fp),
                        body: Box::new(fb),
                        generator: true,
                        is_async: true,
                    },
                )),
                PropertyKind::Init,
                true,
                false,
            )
        }
        Rule::getter => {
            let (p, p_s, p_computed) = build_ast_from_property_name(inner_iter.next().unwrap(), script)?;
            let (fb, fb_s) = build_ast_from_function_body(inner_iter.next().unwrap(), script)?;
            s.merge(p_s).merge(fb_s);
            let meta2 = meta.clone();
            PropertyData::new_with_any_expression_key(
                meta,
                p,
                p_computed,
                Box::new(ExpressionType::FunctionOrGeneratorExpression(
                    FunctionData {
                        meta: meta2,
                        id: None,
                        params: FormalParameters::new(vec![]),
                        body: Box::new(fb),
                        generator: false,
                        is_async: false,
                    },
                )),
                PropertyKind::Get,
                false,
                false,
            )
        }
        Rule::setter => {
            let (p, p_s, p_computed) = build_ast_from_property_name(inner_iter.next().unwrap(), script)?;
            // Setters use property_set_parameter_list which is a single formal_parameter (silent rule)
            let (fp, fp_s) = build_ast_from_single_formal_parameter(inner_iter.next().unwrap(), script)?;
            let (fb, fb_s) = build_ast_from_function_body(inner_iter.next().unwrap(), script)?;
            validate_bound_names_have_no_duplicates_and_also_not_present_in_var_declared_names_or_lexically_declared_names(&fp_s.bound_names,&vec![],&fb_s.lexically_declared_names)?;
            s.merge(p_s).merge(fp_s).merge(fb_s);
            let meta2 = meta.clone();
            PropertyData::new_with_any_expression_key(
                meta,
                p,
                p_computed,
                Box::new(ExpressionType::FunctionOrGeneratorExpression(
                    FunctionData {
                        meta: meta2,
                        id: None,
                        params: FormalParameters::new(fp),
                        body: Box::new(fb),
                        generator: false,
                        is_async: false,
                    },
                )),
                PropertyKind::Set,
                false,
                false,
            )
        }
        _ => {
            return Err(get_unexpected_error(
                "build_ast_from_method_definition",
                &inner_pair,
            ))
        }
    };
    Ok((m, s))
}

fn build_ast_from_identifier_reference(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ExpressionType, Semantics), JsRuleError> {
    let id = pair.as_str();
    if id == "yield" {
        Err(get_validation_error(
            format!("Invalid identifier reference: {}", id),
            AstBuilderValidationErrorType::SyntaxError,
            &pair,
            &script,
        ))
    } else {
        let s = Semantics::new_empty();
        Ok((
            ExpressionPatternType::Identifier(get_identifier_data(pair, script))
                .convert_to_expression(),
            s,
        ))
    }
}

/// Builds AST from a class declaration.
/// Grammar: "class" ~ binding_identifier ~ class_tail
fn build_ast_from_class_declaration(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ClassData, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut pair_iter = pair.into_inner();
    let mut s = Semantics::new_empty();

    // binding_identifier is required for class declarations
    let id_pair = pair_iter.next().ok_or_else(|| {
        get_validation_error_with_meta(
            "Expected class name".to_string(),
            AstBuilderValidationErrorType::SyntaxError,
            meta.clone(),
        )
    })?;
    let (id, id_s) = get_binding_identifier_data(id_pair, script)?;
    s.merge(id_s);

    // Parse class_tail
    let class_tail_pair = pair_iter.next().ok_or_else(|| {
        get_validation_error_with_meta(
            "Expected class body".to_string(),
            AstBuilderValidationErrorType::SyntaxError,
            meta.clone(),
        )
    })?;
    let (super_class, body, tail_s) = build_ast_from_class_tail(class_tail_pair, script)?;
    s.merge(tail_s);

    Ok((
        ClassData {
            meta,
            id: Some(id),
            super_class,
            body,
        },
        s,
    ))
}

/// Builds AST from a class expression.
/// Grammar: "class" ~ binding_identifier? ~ class_tail
fn build_ast_from_class_expression(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(ClassData, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut pair_iter = pair.into_inner();
    let mut s = Semantics::new_empty();

    // Check if there's an optional binding_identifier
    let first_pair = pair_iter.next();
    let (id, class_tail_pair) = if let Some(fp) = first_pair {
        if fp.as_rule() == Rule::binding_identifier || fp.as_rule() == Rule::binding_identifier__yield
        {
            let (id, _id_s) = get_binding_identifier_data(fp, script)?;
            // For class expressions, we don't include the name in bound_names
            let tail = pair_iter.next().ok_or_else(|| {
                get_validation_error_with_meta(
                    "Expected class body".to_string(),
                    AstBuilderValidationErrorType::SyntaxError,
                    meta.clone(),
                )
            })?;
            (Some(id), tail)
        } else {
            // It's the class_tail
            (None, fp)
        }
    } else {
        return Err(get_validation_error_with_meta(
            "Expected class body".to_string(),
            AstBuilderValidationErrorType::SyntaxError,
            meta.clone(),
        ));
    };

    let (super_class, body, tail_s) = build_ast_from_class_tail(class_tail_pair, script)?;
    s.merge(tail_s);

    Ok((
        ClassData {
            meta,
            id,
            super_class,
            body,
        },
        s,
    ))
}

/// Builds AST from class_tail.
/// Grammar: class_heritage? ~ "{" ~ class_body? ~ "}"
fn build_ast_from_class_tail(
    pair: Pair<Rule>,
    script: &Rc<String>,
) -> Result<(Option<Box<ExpressionType>>, ClassBodyData, Semantics), JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut s = Semantics::new_empty();
    let mut super_class = None;
    let mut methods: Vec<MethodDefinitionData> = vec![];
    let mut fields: Vec<ClassFieldData> = vec![];
    let mut static_blocks: Vec<StaticBlockData> = vec![];

    for inner_pair in pair.into_inner() {
        match inner_pair.as_rule() {
            Rule::class_heritage | Rule::class_heritage__yield => {
                // class_heritage = "extends" ~ left_hand_side_expression
                let lhs_pair = inner_pair.into_inner().next().ok_or_else(|| {
                    get_validation_error_with_meta(
                        "Expected superclass expression".to_string(),
                        AstBuilderValidationErrorType::SyntaxError,
                        meta.clone(),
                    )
                })?;
                let (expr, expr_s) = build_ast_from_left_hand_side_expression(lhs_pair, script)?;
                s.merge(expr_s);
                super_class = Some(Box::new(expr));
            }
            Rule::class_body | Rule::class_body__yield => {
                // class_body = class_element_list
                for element_pair in inner_pair.into_inner() {
                    match build_ast_from_class_element(element_pair, script, &mut s)? {
                        Some(ClassElement::Method(m)) => methods.push(m),
                        Some(ClassElement::Field(f)) => fields.push(f),
                        Some(ClassElement::StaticBlock(b)) => static_blocks.push(b),
                        None => {}
                    }
                }
            }
            _ => {
                return Err(get_unexpected_error("build_ast_from_class_tail", &inner_pair))
            }
        }
    }

    Ok((
        super_class,
        ClassBodyData {
            meta,
            body: methods,
            fields,
            static_blocks,
        },
        s,
    ))
}

/// ChittiOS: a parsed class element — a method, a field, or a static block.
enum ClassElement {
    Method(MethodDefinitionData),
    Field(ClassFieldData),
    StaticBlock(StaticBlockData),
}

/// Builds AST from a class_element.
/// Grammar: `static`-prefixed or plain method_definition / field_definition /
/// static block, or a lone `;`.
fn build_ast_from_class_element(
    pair: Pair<Rule>,
    script: &Rc<String>,
    s: &mut Semantics,
) -> Result<Option<ClassElement>, JsRuleError> {
    let meta = get_meta(&pair, script);
    let mut inner_iter = pair.into_inner();

    // Check if empty (semicolon-only element)
    let first_pair = match inner_iter.next() {
        Some(p) => p,
        None => return Ok(None), // Empty class element (;)
    };

    let (is_static, member_pair) = if first_pair.as_rule() == Rule::class_static {
        let mp = inner_iter.next().ok_or_else(|| {
            get_validation_error_with_meta(
                "Expected member after 'static'".to_string(),
                AstBuilderValidationErrorType::SyntaxError,
                meta.clone(),
            )
        })?;
        (true, mp)
    } else {
        (false, first_pair)
    };

    // A `static { … }` initialization block.
    if member_pair.as_rule() == Rule::class_static_block {
        let mut body = vec![];
        for sp in member_pair.into_inner() {
            if sp.as_rule() == Rule::statement_list {
                let (stmts, st_s) = build_ast_from_statement_list(sp, script)?;
                s.merge(st_s);
                body = stmts;
            }
        }
        return Ok(Some(ClassElement::StaticBlock(StaticBlockData { meta, body })));
    }

    // A field definition: `name`, `name = init`, `#priv = init`, `[computed] = …`.
    if member_pair.as_rule() == Rule::field_definition {
        let mut fi = member_pair.into_inner();
        let name_pair = fi.next().unwrap();
        // class_element_name = property_name | private_name
        let name_inner = name_pair.into_inner().next().unwrap();
        let (key, computed) = match name_inner.as_rule() {
            Rule::property_name | Rule::property_name__yield => {
                // The builder reports `computed` itself now — this site used to
                // re-derive it by re-walking the pair, which was the only place
                // that got it right.
                let (k, k_s, computed) =
                    build_ast_from_property_name(name_inner.clone(), script)?;
                s.merge(k_s);
                (Box::new(k), computed)
            }
            Rule::private_name => {
                let id = get_identifier_data(name_inner, script);
                (
                    Box::new(ExpressionPatternType::Identifier(id).convert_to_expression()),
                    false,
                )
            }
            _ => return Err(get_unexpected_error("build_ast_from_class_element:field", &name_inner)),
        };
        let value = match fi.next() {
            Some(init_pair) => {
                // initializer__in = "=" ~ assignment_expression__in
                let expr_pair = init_pair.into_inner().next().unwrap();
                let (e, e_s) = build_ast_from_assignment_expression(expr_pair, script)?;
                s.merge(e_s);
                Some(Box::new(e))
            }
            None => None,
        };
        return Ok(Some(ClassElement::Field(ClassFieldData {
            meta,
            key,
            computed,
            is_static,
            value,
        })));
    }

    let method_pair = member_pair;

    // Parse the method definition
    let (prop_data, method_s) = build_ast_from_method_definition(method_pair, script)?;
    s.merge(method_s);

    // Convert PropertyData to MethodDefinitionData
    let PropertyData {
        meta: prop_meta,
        key,
        value,
        kind,
        method: _,
        shorthand: _,
        computed,
    } = prop_data;

    // Extract FunctionData from the value expression
    let func_data = match *value {
        ExpressionType::FunctionOrGeneratorExpression(f) => f,
        _ => {
            return Err(get_validation_error_with_meta(
                "Expected function in method definition".to_string(),
                AstBuilderValidationErrorType::SyntaxError,
                prop_meta.clone(),
            ));
        }
    };

    // Determine method kind
    let method_kind = match kind {
        PropertyKind::Init => {
            // Check if it's a constructor
            if let ExpressionType::ExpressionWhichCanBePattern(ExpressionPatternType::Identifier(
                ref id,
            )) = *key
            {
                if id.name == "constructor" && !is_static {
                    MethodDefinitionKind::Constructor
                } else {
                    MethodDefinitionKind::Method
                }
            } else {
                MethodDefinitionKind::Method
            }
        }
        PropertyKind::Get => MethodDefinitionKind::Get,
        PropertyKind::Set => MethodDefinitionKind::Set,
        // A method definition is never spread; treat defensively as a method.
        PropertyKind::Spread => MethodDefinitionKind::Method,
    };

    Ok(Some(ClassElement::Method(MethodDefinitionData {
        meta: prop_meta,
        key,
        value: func_data,
        kind: method_kind,
        computed,
        static_flag: is_static,
    })))
}

#[cfg(test)]
mod annex_b_legacy_octal_tests {
    use super::*;

    /// Parse an expression `expr` as `var _x = <expr>;` and return the integer
    /// value of the resulting numeric literal (panics if it is not one).
    fn int_value(expr: &str) -> i64 {
        let src = format!("var _x = {};", expr);
        let prog = JsParser::parse_to_ast_from_str(&src)
            .unwrap_or_else(|e| panic!("parse `{}` failed: {:?}", expr, e));
        // Walk to the initializer literal via the formatted-string is fragile;
        // instead re-parse just the literal through the numeric builder path.
        let _ = prog; // parsing succeeded — value checked via cook/int helpers below
        // Directly exercise the numeric literal builder through the grammar.
        let mut pairs = JsParser::parse(Rule::numeric_literal, expr).expect("numeric parse");
        let pair = pairs.next().unwrap();
        match build_ast_from_numeric_literal_inner(pair.into_inner().next().unwrap()).unwrap() {
            NumberLiteralType::IntegerLiteral(v) => v,
            other => panic!("expected integer, got {:?} for {}", other, expr),
        }
    }

    #[test]
    fn legacy_octal_integer_values() {
        assert_eq!(int_value("00"), 0);
        assert_eq!(int_value("07"), 7);
        assert_eq!(int_value("010"), 8);
        assert_eq!(int_value("077"), 63);
        assert_eq!(int_value("0123"), 83);
    }

    #[test]
    fn non_octal_decimal_integer_values() {
        assert_eq!(int_value("08"), 8);
        assert_eq!(int_value("09"), 9);
        assert_eq!(int_value("018"), 18);
        assert_eq!(int_value("019"), 19);
        assert_eq!(int_value("0789"), 789);
        assert_eq!(int_value("088"), 88);
    }

    #[test]
    fn plain_numbers_unchanged() {
        assert_eq!(int_value("0"), 0);
        assert_eq!(int_value("42"), 42);
        assert_eq!(int_value("0x1f"), 31);
        assert_eq!(int_value("0o17"), 15);
        assert_eq!(int_value("0b11"), 3);
    }

    #[test]
    fn legacy_octal_string_escapes() {
        // LegacyOctalEscapeSequence: 1–3 octal digits.
        assert_eq!(cook_string_literal("\\1"), "\u{01}");
        assert_eq!(cook_string_literal("\\7"), "\u{07}");
        assert_eq!(cook_string_literal("\\40"), " "); // 0x20
        assert_eq!(cook_string_literal("\\101"), "A"); // 0x41
        assert_eq!(cook_string_literal("\\377"), "\u{ff}");
        assert_eq!(cook_string_literal("\\251"), "\u{a9}");
        // ZeroToThree admits up to two more; FourToSeven admits only one.
        assert_eq!(cook_string_literal("\\400"), "\u{20}0"); // \40 then '0'
        assert_eq!(cook_string_literal("\\00"), "\u{00}");
    }

    #[test]
    fn non_octal_decimal_and_nul_escapes() {
        // NonOctalDecimalEscapeSequence \8 \9 -> the chars "8"/"9".
        assert_eq!(cook_string_literal("\\8"), "8");
        assert_eq!(cook_string_literal("\\9"), "9");
        // \0 not followed by an octal digit is NUL; \08 is NUL then '8'.
        assert_eq!(cook_string_literal("\\0"), "\u{00}");
        assert_eq!(cook_string_literal("\\08"), "\u{00}8");
        assert_eq!(cook_string_literal("\\18"), "\u{01}8");
    }

    #[test]
    fn strict_forbidden_escape_detection() {
        assert!(string_has_strict_forbidden_escape("'\\1'"));
        assert!(string_has_strict_forbidden_escape("'\\052'"));
        assert!(string_has_strict_forbidden_escape("'\\8'"));
        assert!(string_has_strict_forbidden_escape("'\\08'"));
        // A bare \0, escaped backslash, or ordinary escapes are allowed.
        assert!(!string_has_strict_forbidden_escape("'\\0'"));
        assert!(!string_has_strict_forbidden_escape("'\\\\1'"));
        assert!(!string_has_strict_forbidden_escape("'\\n\\t\\x41'"));
        assert!(!string_has_strict_forbidden_escape("'plain'"));
    }

    #[test]
    fn directive_value_extraction() {
        assert_eq!(directive_string_value("\"use strict\";"), Some("use strict"));
        assert_eq!(directive_string_value("'use strict'"), Some("use strict"));
        assert_eq!(directive_string_value("  \"use strict\" ; "), Some("use strict"));
        assert_eq!(directive_string_value("\"other\";"), Some("other"));
        assert_eq!(directive_string_value("var x = 1;"), None);
        assert_eq!(directive_string_value("1 + 2;"), None);
    }

    #[test]
    fn non_strict_accepts_legacy_octal() {
        assert!(JsParser::parse_to_ast_from_str("var x = 010;").is_ok());
        assert!(JsParser::parse_to_ast_from_str("var x = 08;").is_ok());
        assert!(JsParser::parse_to_ast_from_str("var s = '\\1';").is_ok());
        assert!(JsParser::parse_to_ast_from_str("var s = '\\8';").is_ok());
    }

    #[test]
    fn strict_rejects_legacy_octal() {
        // Forced top-level strict (as onlyStrict).
        assert!(JsParser::parse_to_ast_from_str_strict("010;", true).is_err());
        assert!(JsParser::parse_to_ast_from_str_strict("08;", true).is_err());
        assert!(JsParser::parse_to_ast_from_str_strict("var s = '\\1';", true).is_err());
        assert!(JsParser::parse_to_ast_from_str_strict("var s = '\\8';", true).is_err());
        // A bare \0 and ordinary numbers are still fine under strict.
        assert!(JsParser::parse_to_ast_from_str_strict("var s = '\\0';", true).is_ok());
        assert!(JsParser::parse_to_ast_from_str_strict("var n = 42;", true).is_ok());
    }

    #[test]
    fn strict_via_directive_prologue() {
        // A program-level "use strict" prologue makes legacy octal an error.
        assert!(JsParser::parse_to_ast_from_str("\"use strict\"; 010;").is_err());
        // "use strict" inside a function only makes THAT function strict.
        assert!(
            JsParser::parse_to_ast_from_str("function f(){ \"use strict\"; return '\\1'; }")
                .is_err()
        );
        // Octal in a sloppy function is fine even if another function is strict.
        assert!(JsParser::parse_to_ast_from_str(
            "function g(){ return 010; } function f(){ \"use strict\"; return 1; }"
        )
        .is_ok());
    }

    #[test]
    fn template_legacy_octal_always_rejected() {
        // Legacy octal escapes are a SyntaxError in templates, even sloppy.
        assert!(JsParser::parse_to_ast_from_str("var t = `\\1`;").is_err());
        assert!(JsParser::parse_to_ast_from_str("var t = `\\8`;").is_err());
    }
}
