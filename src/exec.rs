// Copyright 2018 Ethan Pailes.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::HashMap;

use regex_syntax;
use regex_syntax::ast::{RepetitionKind, GroupKind};

use error::{InternalError, ErrorKind};
use ast::{Expr, ExprKind, BOp, UOp, Statement, StatementKind};
use operators;
use operators::noncapturing_group;
use util::POISON_SPAN;

/// A remake runtime value.
///
/// Remake values are expected to be wrapped in an Rc<RefCell<_>>
/// in order to provide garbage collection.
#[derive(Debug, Clone)]
pub enum Value {
    Regex(Box<regex_syntax::ast::Ast>),
}

impl Value {
    pub fn type_of(&self) -> &str {
        match self {
            &Value::Regex(_) => "regex",
        }
    }
}

pub fn eval(expr: Expr) -> Result<Value, InternalError> {
    eval_(&mut EvalEnv::new(), expr)
}

/// Evaluate an expression and return the inner type of the
/// resulting value if it matches the given type.
macro_rules! expect_type {
    ($env:expr, $expr:expr, "regex") => {{
        let e = $expr;
        let expr_span = e.span.clone();
        match eval_($env, e)? {
            Value::Regex(re) => re,
            val => return Err(InternalError::new(
                ErrorKind::TypeError {
                    actual: val.type_of().to_string(),
                    expected: "regex".to_string(),
                },
                expr_span,
            ))
        }
    }}
}

fn eval_(env: &mut EvalEnv, expr: Expr)
    -> Result<Value, InternalError>
{
    match expr.kind {
        ExprKind::RegexLiteral(r) => Ok(Value::Regex(r)),

        ExprKind::BinOp(lhs, op, rhs) => {
            match op {
                BOp::Concat => Ok(Value::Regex(operators::concat(
                    expect_type!(env, *lhs, "regex"),
                    expect_type!(env, *rhs, "regex"),
                ))),
                BOp::Alt => {
                    Ok(Value::Regex(operators::alt(
                        expect_type!(env, *lhs, "regex"),
                        expect_type!(env, *rhs, "regex"),
                    )))
                }
            }
        }

        ExprKind::UnaryOp(op, e) => match op {
            UOp::RepeatZeroOrMore(greedy) => {
                Ok(Value::Regex(Box::new(regex_syntax::ast::Ast::Repetition(
                    regex_syntax::ast::Repetition {
                        span: POISON_SPAN,
                        op: regex_syntax::ast::RepetitionOp {
                            span: POISON_SPAN,
                            kind: RepetitionKind::ZeroOrMore,
                        },
                        greedy: greedy,
                        ast: Box::new(noncapturing_group(
                                expect_type!(env, *e, "regex"))),
                    },
                ))))
            }
            UOp::RepeatOneOrMore(greedy) => {
                Ok(Value::Regex(Box::new(regex_syntax::ast::Ast::Repetition(
                    regex_syntax::ast::Repetition {
                        span: POISON_SPAN,
                        op: regex_syntax::ast::RepetitionOp {
                            span: POISON_SPAN,
                            kind: RepetitionKind::OneOrMore,
                        },
                        greedy: greedy,
                        ast: Box::new(noncapturing_group(
                                expect_type!(env, *e, "regex"))),
                    },
                ))))
            }
            UOp::RepeatZeroOrOne(greedy) => {
                Ok(Value::Regex(Box::new(regex_syntax::ast::Ast::Repetition(
                    regex_syntax::ast::Repetition {
                        span: POISON_SPAN,
                        op: regex_syntax::ast::RepetitionOp {
                            span: POISON_SPAN,
                            kind: RepetitionKind::ZeroOrOne,
                        },
                        greedy: greedy,
                        ast: Box::new(noncapturing_group(
                                expect_type!(env, *e, "regex"))),
                    },
                ))))
            }
            UOp::RepeatRange(greedy, range) => {
                Ok(Value::Regex(Box::new(regex_syntax::ast::Ast::Repetition(
                    regex_syntax::ast::Repetition {
                        span: POISON_SPAN,
                        op: regex_syntax::ast::RepetitionOp {
                            span: POISON_SPAN,
                            kind: RepetitionKind::Range(range),
                        },
                        greedy: greedy,
                        ast: Box::new(noncapturing_group(
                                expect_type!(env, *e, "regex"))),
                    },
                ))))
            }
        },

        ExprKind::Capture(e, name) => Ok(Value::Regex(Box::new(
            regex_syntax::ast::Ast::Group(regex_syntax::ast::Group {
                span: POISON_SPAN,
                kind: match name {
                    Some(n) => GroupKind::CaptureName(
                        regex_syntax::ast::CaptureName {
                            span: POISON_SPAN,
                            name: n,
                            index: BOGUS_GROUP_INDEX,
                        },
                    ),
                    None => GroupKind::CaptureIndex(BOGUS_GROUP_INDEX),
                },
                ast: expect_type!(env, *e, "regex"),
            }),
        ))),

        ExprKind::Block(statements, value) => {
            env.push_block_env();
            for s in statements {
                exec(env, s)?;
                // s.eval(env)?;
            }
            let res = eval_(env, *value)?;
            env.pop_block_env();

            Ok(res)
        }

        ExprKind::Var(var) => {
            let span = expr.span;
            env.lookup(var)
               .map_err(|e| InternalError::new(e, span))
        }

        ExprKind::ExprPoison => panic!("Bug in remake."),
    }

}

fn exec(env: &mut EvalEnv, s: Statement) -> Result<(), InternalError> {
    match s.kind {
        StatementKind::LetBinding(id, e) => {
            let v = eval_(env, *e)?;
            env.bind(id.clone(), v);
            Ok(())
        }
    }
}

//
// The evaluation environment
//

struct EvalEnv {
    block_envs: Vec<HashMap<String, Value>>,
}
impl EvalEnv {
    fn new() -> Self {
        EvalEnv {
            block_envs: vec![],
        }
    }

    fn push_block_env(&mut self) {
        self.block_envs.push(HashMap::new());
    }

    fn pop_block_env(&mut self) {
        self.block_envs.pop();
    }

    fn bind(&mut self, var: String, v: Value) {
        let idx = self.block_envs.len() - 1;
        self.block_envs[idx].insert(var, v);
    }

    fn lookup(&self, var: String) -> Result<Value, ErrorKind> {
        for env in self.block_envs.iter().rev() {
            match env.get(&var) {
                None => {}
                // TODO(ethan): drop the clone
                Some(val) => return Ok(val.clone()),
            }
        }

        Err(ErrorKind::NameError { name: var })
    }
}

/// We don't have to spend any effort assigning indicies to groups because
/// we are going to pretty-print the AST and have regex just parse it.
/// If we passed the AST to the regex crate directly, we would need some
/// way to thread the group index through its parser. This way we can
/// just ignore the whole problem.
const BOGUS_GROUP_INDEX: u32 = 0;
