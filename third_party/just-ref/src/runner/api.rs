
// pub struct JsRunner {}
//
// impl JsRunner {}
//
// fn run_script<'a>(
//     ast: &'a ProgramData,
//     mut env: JsEnvironment<'a>,
// ) -> Result<JsEnvironment<'a>, JsExecutionError> {
//     let statements = &ast.body;
//     let scope = env.get_scope_mut(GLOBAL_SCOPE_ID).unwrap();
//     scope.add_declarations(get_declarations_from_block_of_statements(
//         statements,
//         GLOBAL_SCOPE_ID,
//     )?);
//
//     Ok(run_block_of_statements(
//         statements,
//         env,
//         GLOBAL_SCOPE_ID,
//         None,
//     )?)
// }
//
// fn run_block_of_statements<'a>(
//     statements: &'a Vec<StatementType>,
//     mut env: JsEnvironment<'a>,
//     scope_id: ScopeId,
//     ctx: Option<&'a Lambda<'a>>,
// ) -> Result<JsEnvironment<'a>, JsExecutionError> {
//     // Let's execute!
//     for statement in statements {
//         match statement {
//             StatementType::ExpressionStatement { meta, expression } => {
//                 run_an_expression(expression.deref(), env.get_scope_mut(scope_id).unwrap())?;
//             }
//             StatementType::BlockStatement(d) => {
//                 let new_scope_id = env.new_scope(scope_id);
//                 // TODO add protection against infinite loops.
//                 env = run_block_of_statements(d.body.deref(), env, new_scope_id, ctx)?;
//             }
//             StatementType::FunctionBody(d) => {
//                 panic!("Use case is not clear. Hence not implemented.")
//             }
//             StatementType::EmptyStatement { meta } => {}
//             StatementType::DebuggerStatement { meta } => unimplemented!(),
//             StatementType::ReturnStatement { meta, argument } => {}
//             StatementType::BreakStatement { meta } => {}
//             StatementType::ContinueStatement { meta } => {}
//             StatementType::IfStatement {
//                 meta,
//                 test,
//                 consequent,
//                 alternate,
//             } => {}
//             StatementType::SwitchStatement {
//                 meta,
//                 discriminant,
//                 cases,
//             } => {}
//             StatementType::ThrowStatement { meta, argument } => {}
//             StatementType::TryStatement {
//                 meta,
//                 block,
//                 handler,
//                 finalizer,
//             } => {}
//             StatementType::WhileStatement { meta, test, body } => {}
//             StatementType::DoWhileStatement { meta, test, body } => {}
//             StatementType::ForStatement {
//                 meta,
//                 init,
//                 test,
//                 update,
//                 body,
//             } => {}
//             StatementType::ForInStatement(d) => {}
//             StatementType::ForOfStatement(d) => {}
//             StatementType::DeclarationStatement(d) => {}
//         }
//     }
//
//     Ok(env)
// }
//
// fn get_declarations_from_block_of_statements(
//     statements: &Vec<StatementType>,
//     scope_id: ScopeId,
// ) -> Result<HashMap<&IdentifierData, JsObject>, JsExecutionError> {
//     let mut declarations = HashMap::new();
//     // Scan for declarations
//     for statement in statements {
//         declarations.extend(get_declarations_from_a_statement(statement, scope_id)?);
//     }
//     Ok(declarations)
// }
//
// fn get_declarations_from_a_statement(
//     statement: &StatementType,
//     scope_id: ScopeId,
// ) -> Result<HashMap<&IdentifierData, JsObject>, JsExecutionError> {
//     let mut declarations = HashMap::new();
//     // Except for Function and Class declarations, we record just the identifiers for all others.
//     // Their init values (if exist) will be evaluated during execution phase.
//     match statement {
//         StatementType::ExpressionStatement { .. } => {}
//         StatementType::BlockStatement(d) => {
//             let more_declarations =
//                 get_declarations_from_block_of_statements(d.body.deref(), scope_id)?;
//             declarations.extend(more_declarations);
//         }
//         StatementType::FunctionBody(_) => {}
//         StatementType::EmptyStatement { .. } => {}
//         StatementType::DebuggerStatement { .. } => {}
//         StatementType::ReturnStatement { .. } => {}
//         StatementType::BreakStatement { .. } => {}
//         StatementType::ContinueStatement { .. } => {}
//         StatementType::IfStatement {
//             consequent,
//             alternate,
//             ..
//         } => {
//             let mut more_declarations =
//                 get_declarations_from_a_statement(consequent.deref(), scope_id)?;
//             if let Some(alternate) = alternate {
//                 more_declarations.extend(get_declarations_from_a_statement(
//                     alternate.deref(),
//                     scope_id,
//                 )?);
//             }
//             declarations.extend(more_declarations);
//         }
//         StatementType::SwitchStatement { cases, .. } => {
//             for case in cases.deref() {
//                 declarations.extend(get_declarations_from_block_of_statements(
//                     case.consequent.deref(),
//                     scope_id,
//                 )?);
//             }
//         }
//         StatementType::ThrowStatement { .. } => {}
//         StatementType::TryStatement { .. } => {}
//         StatementType::WhileStatement { body, .. } => {
//             declarations.extend(get_declarations_from_a_statement(body, scope_id)?);
//         }
//         StatementType::DoWhileStatement { body, .. } => {
//             declarations.extend(get_declarations_from_a_statement(body, scope_id)?);
//         }
//         StatementType::ForStatement { init, body, .. } => {
//             if let Some(init) = init {
//                 match init {
//                     VariableDeclarationOrExpression::VariableDeclaration(d) => {
//                         declarations.extend(get_declarations_from_a_variable_declaration_data(
//                             d, scope_id,
//                         )?);
//                     }
//                     VariableDeclarationOrExpression::Expression(_) => {}
//                 }
//             }
//             declarations.extend(get_declarations_from_a_statement(body.deref(), scope_id)?);
//         }
//         StatementType::ForInStatement(fid) | StatementType::ForOfStatement(fid) => {
//             match &fid.left {
//                 VariableDeclarationOrExpression::VariableDeclaration(d) => {
//                     declarations.extend(get_declarations_from_a_variable_declaration_data(
//                         d, scope_id,
//                     )?);
//                 }
//                 VariableDeclarationOrExpression::Expression(_) => {}
//             }
//             declarations.extend(get_declarations_from_a_statement(
//                 fid.body.deref(),
//                 scope_id,
//             )?);
//         }
//         StatementType::DeclarationStatement(declaration) => {
//             match declaration {
//                 DeclarationType::FunctionDeclaration(f) => {
//                     declarations.insert(
//                         f.id.as_ref().unwrap(),
//                         JsObject::Function(Lambda {
//                             def: f,
//                             scope: scope_id,
//                         }),
//                     );
//                 }
//                 DeclarationType::VariableDeclaration(v) => {
//                     declarations.extend(get_declarations_from_a_variable_declaration_data(
//                         v, scope_id,
//                     )?);
//                 }
//                 DeclarationType::ClassDeclaration(c) => unimplemented!(),
//             };
//         }
//     };
//     Ok(declarations)
// }
//
// fn get_declarations_from_a_variable_declaration_data(
//     v_data: &VariableDeclarationData,
//     _scope_id: ScopeId,
// ) -> Result<HashMap<&IdentifierData, JsObject>, JsExecutionError> {
//     let mut declarations = HashMap::new();
//     match v_data.kind {
//         VariableDeclarationKind::Var => {
//             for declaration in &v_data.declarations {
//                 let id = declaration.id.as_ref();
//                 match id {
//                     PatternType::PatternWhichCanBeExpression(d) => match d {
//                         ExpressionPatternType::Identifier(id) => {
//                             declarations.insert(id, JsObject::Undefined);
//                         }
//                         ExpressionPatternType::MemberExpression(d) => {
//                             unimplemented!()
//                         }
//                     },
//                     PatternType::ObjectPattern { properties, .. } => {
//                         for prop in properties {
//                             match &prop.0.key {
//                                 LiteralOrIdentifier::Identifier(id) => {
//                                     declarations.insert(id, JsObject::Undefined);
//                                 }
//                                 LiteralOrIdentifier::Literal(_) => {}
//                             }
//                         }
//                     }
//                     PatternType::ArrayPattern { .. } => unimplemented!(),
//                     PatternType::RestElement { .. } => unimplemented!(),
//                     PatternType::AssignmentPattern { .. } => unimplemented!(),
//                 }
//             }
//         }
//         _ => { /* Ignore Let & Const */ }
//     };
//     Ok(declarations)
// }
//
// fn get_execution_error(meta: &Meta, message: String) -> JsExecutionError {
//     JsExecutionError {
//         meta: meta.clone(),
//         message,
//     }
// }
//
// fn run_an_expression<'a>(
//     expression: &'a ExpressionType,
//     scope: &'a mut JsScope,
// ) -> Result<JsObject<'a>, JsExecutionError> {
//     match expression {
//         ExpressionType::ExpressionWhichCanBePattern(p) => match p {
//             ExpressionPatternType::Identifier(id) => match scope.get_value_by_identifier_data(id) {
//                 None => {
//                     return Err(get_execution_error(
//                         id.get_meta(),
//                         format!("Uncaught reference error: '{}' is not defined.", id.name),
//                     ));
//                 }
//                 Some(v) => v.clone(),
//             },
//             ExpressionPatternType::MemberExpression(m) => match m {
//                 MemberExpressionType::SimpleMemberExpression {
//                     object,
//                     property,
//                     meta,
//                 } => {
//                     let object = match object {
//                         ExpressionOrSuper::Expression(e) => run_an_expression(e.deref(), scope)?,
//                         ExpressionOrSuper::Super => {
//                             let o = scope.get_value(&"super".to_string());
//                             match o {
//                                 None => {
//                                     return Err(get_execution_error(
//                                         meta,
//                                         format!(
//                                             "Uncaught reference error: 'super' is not defined."
//                                         ),
//                                     ));
//                                 }
//                                 Some(s) => s.clone(),
//                             }
//                         }
//                     };
//                 }
//                 MemberExpressionType::ComputedMemberExpression { .. } => {}
//             },
//         },
//         ExpressionType::Literal(_) => {}
//         ExpressionType::ThisExpression { .. } => {}
//         ExpressionType::ArrayExpression { .. } => {}
//         ExpressionType::ObjectExpression { .. } => {}
//         ExpressionType::FunctionExpression(_) => {}
//         ExpressionType::UnaryExpression { .. } => {}
//         ExpressionType::UpdateExpression { .. } => {}
//         ExpressionType::BinaryExpression { .. } => {}
//         ExpressionType::AssignmentExpression { .. } => {}
//         ExpressionType::LogicalExpression { .. } => {}
//         ExpressionType::ConditionalExpression { .. } => {}
//         ExpressionType::CallExpression { .. } => {}
//         ExpressionType::NewExpression { .. } => {}
//         ExpressionType::SequenceExpression { .. } => {}
//         ExpressionType::ArrowFunctionExpression { .. } => {}
//         ExpressionType::yield_expression { .. } => {}
//         ExpressionType::TemplateLiteral(_) => {}
//         ExpressionType::TaggedTemplateExpression { .. } => {}
//         ExpressionType::ClassExpression(_) => {}
//         ExpressionType::MetaProperty { .. } => {}
//     }
//     Ok()
// }
