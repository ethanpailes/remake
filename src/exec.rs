// Copyright 2018 Ethan Pailes.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::HashMap;

use regex_syntax;
use regex_syntax::ast::{GroupKind, RepetitionKind};

use ast::{BOp, Expr, ExprKind, Statement, StatementKind, UOp};
use error::{ErrorKind, InternalError};
use re_operators;
use re_operators::noncapturing_group;
use util::POISON_SPAN;

/// A remake runtime value.
///
/// Remake values are expected to be wrapped in an Rc<RefCell<_>>
/// in order to provide garbage collection.
#[derive(Debug, Clone)]
pub enum Value {
    Regex(Box<regex_syntax::ast::Ast>),
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

impl Value {
    pub fn type_of(&self) -> &str {
        match self {
            &Value::Regex(_) => "regex",
            &Value::Int(_) => "int",
            &Value::Float(_) => "float",
            &Value::Str(_) => "str",
            &Value::Bool(_) => "bool",
        }
    }
}

pub fn eval(expr: Expr) -> Result<Value, InternalError> {
    eval_(&mut EvalEnv::new(), expr)
}

macro_rules! type_error {
    ($val:expr, $span:expr, $($expected:expr),* ) => {
        Err(InternalError::new(
            ErrorKind::TypeError {
                actual: $val.type_of().to_string(),
                expected: vec![$($expected.to_string()),*],
            },
            $span
            ))
    }
}

/// Evaluate an expression and return the inner type of the
/// resulting value if it matches the given type.
macro_rules! expect_type {
    ($env:expr, $expr:expr,"regex") => {{
        let e = $expr;
        let expr_span = e.span.clone();
        match eval_($env, e)? {
            Value::Regex(re) => re,
            val => return type_error!(val, expr_span, "regex"),
        }
    }};
    ($env:expr, $expr:expr,"str") => {{
        let e = $expr;
        let expr_span = e.span.clone();
        match eval_($env, e)? {
            Value::Str(s) => s,
            val => return type_error!(val, expr_span, "str"),
        }
    }};
    ($env:expr, $expr:expr,"int") => {{
        let e = $expr;
        let expr_span = e.span.clone();
        match eval_($env, e)? {
            Value::Int(i) => i,
            val => return type_error!(val, expr_span, "int"),
        }
    }};
    ($env:expr, $expr:expr,"float") => {{
        let e = $expr;
        let expr_span = e.span.clone();
        match eval_($env, e)? {
            Value::Float(i) => i,
            val => return type_error!(val, expr_span, "float"),
        }
    }};
    ($env:expr, $expr:expr,"bool") => {{
        let e = $expr;
        let expr_span = e.span.clone();
        match eval_($env, e)? {
            Value::Bool(b) => b,
            val => return type_error!(val, expr_span, "bool"),
        }
    }};
}

fn eval_(env: &mut EvalEnv, expr: Expr) -> Result<Value, InternalError> {
    match expr.kind {
        ExprKind::RegexLiteral(r) => Ok(Value::Regex(r)),

        ExprKind::BinOp(lhs, op, rhs) => match op {
            BOp::Concat => {
                let expr_span = lhs.span.clone();
                match eval_(env, *lhs)? {
                    Value::Regex(re) => Ok(Value::Regex(re_operators::concat(
                        re,
                        expect_type!(env, *rhs, "regex"),
                    ))),
                    Value::Str(s1) => {
                        let s2 = expect_type!(env, *rhs, "str");
                        let mut s = String::with_capacity(s1.len() + s2.len());
                        s.push_str(&s1);
                        s.push_str(&s2);
                        Ok(Value::Str(s))
                    }
                    val => type_error!(val, expr_span, "regex", "str"),
                }
            }
            BOp::Alt => Ok(Value::Regex(re_operators::alt(
                expect_type!(env, *lhs, "regex"),
                expect_type!(env, *rhs, "regex"),
            ))),

            // comparison operators
            BOp::Equals => Ok(Value::Bool(eval_equals(env, lhs, rhs)?)),
            BOp::Ne => Ok(Value::Bool(!eval_equals(env, lhs, rhs)?)),
            BOp::Lt => Ok(Value::Bool(eval_lt(env, lhs, rhs)?)),
            BOp::Gt => Ok(Value::Bool(eval_gt(env, lhs, rhs)?)),
            BOp::Le => Ok(Value::Bool(eval_le(env, lhs, rhs)?)),
            BOp::Ge => Ok(Value::Bool(eval_ge(env, lhs, rhs)?)),
            BOp::Or => Ok(Value::Bool(
                expect_type!(env, *lhs, "bool")
                    || expect_type!(env, *rhs, "bool"),
            )),
            BOp::And => Ok(Value::Bool(
                expect_type!(env, *lhs, "bool")
                    && expect_type!(env, *rhs, "bool"),
            )),

            // arith operators
            BOp::Plus => {
                let span = lhs.span.clone();
                match eval_(env, *lhs)? {
                    Value::Int(i) => {
                        Ok(Value::Int(i + expect_type!(env, *rhs, "int")))
                    }
                    Value::Float(f) => Ok(Value::Float(
                        f + expect_type!(env, *rhs, "float"),
                    )),
                    val => type_error!(val, span, "int", "float"),
                }
            }
            BOp::Minus => {
                let span = lhs.span.clone();
                match eval_(env, *lhs)? {
                    Value::Int(i) => {
                        Ok(Value::Int(i - expect_type!(env, *rhs, "int")))
                    }
                    Value::Float(f) => Ok(Value::Float(
                        f - expect_type!(env, *rhs, "float"),
                    )),
                    val => type_error!(val, span, "int", "float"),
                }
            }
            BOp::Div => {
                let span = lhs.span.clone();
                match eval_(env, *lhs)? {
                    Value::Int(i) => {
                        Ok(Value::Int(i / expect_type!(env, *rhs, "int")))
                    }
                    Value::Float(f) => Ok(Value::Float(
                        f / expect_type!(env, *rhs, "float"),
                    )),
                    val => type_error!(val, span, "int", "float"),
                }
            }
            BOp::Times => {
                let span = lhs.span.clone();
                match eval_(env, *lhs)? {
                    Value::Int(i) => {
                        Ok(Value::Int(i * expect_type!(env, *rhs, "int")))
                    }
                    Value::Float(f) => Ok(Value::Float(
                        f * expect_type!(env, *rhs, "float"),
                    )),
                    val => type_error!(val, span, "int", "float"),
                }
            }
            BOp::Mod => {
                let span = lhs.span.clone();
                match eval_(env, *lhs)? {
                    Value::Int(i) => {
                        Ok(Value::Int(i % expect_type!(env, *rhs, "int")))
                    }
                    Value::Float(f) => Ok(Value::Float(
                        f % expect_type!(env, *rhs, "float"),
                    )),
                    val => type_error!(val, span, "int", "float"),
                }
            }
        },

        ExprKind::UnaryOp(op, e) => match op {
            UOp::Not => Ok(Value::Bool(!expect_type!(env, *e, "bool"))),
            UOp::Neg => {
                let span = e.span.clone();
                match eval_(env, *e)? {
                    Value::Int(i) => Ok(Value::Int(-i)),
                    Value::Float(f) => Ok(Value::Float(-f)),
                    val => type_error!(val, span, "int", "float"),
                }
            }
            UOp::RepeatZeroOrMore(greedy) => Ok(Value::Regex(Box::new(
                regex_syntax::ast::Ast::Repetition(
                    regex_syntax::ast::Repetition {
                        span: POISON_SPAN,
                        op: regex_syntax::ast::RepetitionOp {
                            span: POISON_SPAN,
                            kind: RepetitionKind::ZeroOrMore,
                        },
                        greedy: greedy,
                        ast: Box::new(noncapturing_group(expect_type!(
                            env, *e, "regex"
                        ))),
                    },
                ),
            ))),
            UOp::RepeatOneOrMore(greedy) => Ok(Value::Regex(Box::new(
                regex_syntax::ast::Ast::Repetition(
                    regex_syntax::ast::Repetition {
                        span: POISON_SPAN,
                        op: regex_syntax::ast::RepetitionOp {
                            span: POISON_SPAN,
                            kind: RepetitionKind::OneOrMore,
                        },
                        greedy: greedy,
                        ast: Box::new(noncapturing_group(expect_type!(
                            env, *e, "regex"
                        ))),
                    },
                ),
            ))),
            UOp::RepeatZeroOrOne(greedy) => Ok(Value::Regex(Box::new(
                regex_syntax::ast::Ast::Repetition(
                    regex_syntax::ast::Repetition {
                        span: POISON_SPAN,
                        op: regex_syntax::ast::RepetitionOp {
                            span: POISON_SPAN,
                            kind: RepetitionKind::ZeroOrOne,
                        },
                        greedy: greedy,
                        ast: Box::new(noncapturing_group(expect_type!(
                            env, *e, "regex"
                        ))),
                    },
                ),
            ))),
            UOp::RepeatRange(greedy, range) => Ok(Value::Regex(Box::new(
                regex_syntax::ast::Ast::Repetition(
                    regex_syntax::ast::Repetition {
                        span: POISON_SPAN,
                        op: regex_syntax::ast::RepetitionOp {
                            span: POISON_SPAN,
                            kind: RepetitionKind::Range(range),
                        },
                        greedy: greedy,
                        ast: Box::new(noncapturing_group(expect_type!(
                            env, *e, "regex"
                        ))),
                    },
                ),
            ))),
        },

        ExprKind::Capture(e, name) => Ok(Value::Regex(Box::new(
            regex_syntax::ast::Ast::Group(regex_syntax::ast::Group {
                span: POISON_SPAN,
                kind: match name {
                    Some(n) => {
                        GroupKind::CaptureName(regex_syntax::ast::CaptureName {
                            span: POISON_SPAN,
                            name: n,
                            index: BOGUS_GROUP_INDEX,
                        })
                    }
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

        ExprKind::IntLiteral(i) => Ok(Value::Int(i)),

        ExprKind::FloatLiteral(f) => Ok(Value::Float(f)),

        ExprKind::StringLiteral(s) => Ok(Value::Str(s)),

        ExprKind::BoolLiteral(b) => Ok(Value::Bool(b)),

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

//
// Utils
//

fn eval_equals(
    env: &mut EvalEnv,
    lhs: Box<Expr>,
    rhs: Box<Expr>,
) -> Result<bool, InternalError> {
    match eval_(env, *lhs)? {
        Value::Regex(re) => Ok(re == expect_type!(env, *rhs, "regex")),
        Value::Int(i) => Ok(i == expect_type!(env, *rhs, "int")),
        Value::Float(f) => {
            Ok((f - expect_type!(env, *rhs, "float")).abs() < FLOAT_EQ_EPSILON)
        }
        Value::Str(s) => Ok(s == expect_type!(env, *rhs, "str")),
        Value::Bool(s) => Ok(s == expect_type!(env, *rhs, "bool")),
    }
}

fn eval_lt(
    env: &mut EvalEnv,
    lhs: Box<Expr>,
    rhs: Box<Expr>,
) -> Result<bool, InternalError> {
    let span = lhs.span.clone();
    match eval_(env, *lhs)? {
        Value::Int(i) => Ok(i < expect_type!(env, *rhs, "int")),
        Value::Float(f) => Ok(f < expect_type!(env, *rhs, "float")),
        Value::Str(s) => Ok(s < expect_type!(env, *rhs, "str")),
        Value::Bool(s) => Ok(s < expect_type!(env, *rhs, "bool")),
        val => type_error!(val, span, "int", "float", "str", "bool"),
    }
}

fn eval_gt(
    env: &mut EvalEnv,
    lhs: Box<Expr>,
    rhs: Box<Expr>,
) -> Result<bool, InternalError> {
    let span = lhs.span.clone();
    match eval_(env, *lhs)? {
        Value::Int(i) => Ok(i > expect_type!(env, *rhs, "int")),
        Value::Float(f) => Ok(f > expect_type!(env, *rhs, "float")),
        Value::Str(s) => Ok(s > expect_type!(env, *rhs, "str")),
        Value::Bool(s) => Ok(s > expect_type!(env, *rhs, "bool")),
        val => type_error!(val, span, "int", "float", "str", "bool"),
    }
}

fn eval_le(
    env: &mut EvalEnv,
    lhs: Box<Expr>,
    rhs: Box<Expr>,
) -> Result<bool, InternalError> {
    let span = lhs.span.clone();
    match eval_(env, *lhs)? {
        Value::Int(i) => Ok(i <= expect_type!(env, *rhs, "int")),
        Value::Float(f) => Ok(f <= expect_type!(env, *rhs, "float")),
        Value::Str(s) => Ok(s <= expect_type!(env, *rhs, "str")),
        Value::Bool(s) => Ok(s <= expect_type!(env, *rhs, "bool")),
        val => type_error!(val, span, "int", "float", "str", "bool"),
    }
}

fn eval_ge(
    env: &mut EvalEnv,
    lhs: Box<Expr>,
    rhs: Box<Expr>,
) -> Result<bool, InternalError> {
    let span = lhs.span.clone();
    match eval_(env, *lhs)? {
        Value::Int(i) => Ok(i >= expect_type!(env, *rhs, "int")),
        Value::Float(f) => Ok(f >= expect_type!(env, *rhs, "float")),
        Value::Str(s) => Ok(s >= expect_type!(env, *rhs, "str")),
        Value::Bool(s) => Ok(s >= expect_type!(env, *rhs, "bool")),
        val => type_error!(val, span, "int", "float", "str", "bool"),
    }
}

/// We don't have to spend any effort assigning indicies to groups because
/// we are going to pretty-print the AST and have regex just parse it.
/// If we passed the AST to the regex crate directly, we would need some
/// way to thread the group index through its parser. This way we can
/// just ignore the whole problem.
const BOGUS_GROUP_INDEX: u32 = 0;

const FLOAT_EQ_EPSILON: f64 = 0.000001;

#[cfg(test)]
mod tests {
    use super::*;
    use lex;
    use parse::BlockBodyParser;

    /// An equality function for remake runtime values.
    ///
    /// We give this guys an awkward name rather than just adding
    /// an Eq impl to avoid taking a position about the right way to
    /// compare floating point values in all cases. For testing this
    /// function is good enough.
    fn test_eq(lhs: &Value, rhs: &Value) -> bool {
        match (lhs, rhs) {
            (&Value::Regex(ref l), &Value::Regex(ref r)) => *l == *r,
            (&Value::Int(ref l), &Value::Int(ref r)) => *l == *r,
            (&Value::Str(ref l), &Value::Str(ref r)) => *l == *r,
            (&Value::Bool(ref l), &Value::Bool(ref r)) => *l == *r,

            // stupid fixed-epsilon test
            (&Value::Float(ref l), &Value::Float(ref r)) => {
                (*l - *r).abs() < 0.0000001
            }

            (_, _) => false,
        }
    }

    macro_rules! eval_to {
        ($test_name:ident, $remake_src:expr, $expected_value:expr) => {
            #[test]
            fn $test_name() {
                let parser = BlockBodyParser::new();
                let lexer = lex::Lexer::new($remake_src);
                let expr = eval(parser.parse(lexer).unwrap()).unwrap();

                assert!(
                    test_eq(&$expected_value, &expr),
                    "The expr '{}' evaluates to {:?} not {:?}",
                    $remake_src,
                    expr,
                    $expected_value,
                );
            }
        };
    }

    macro_rules! eval_fail {
        ($test_name:ident, $remake_src:expr) => {
            #[test]
            fn $test_name() {
                let parser = BlockBodyParser::new();
                let lexer = lex::Lexer::new($remake_src);
                let expr = eval(parser.parse(lexer).unwrap());

                assert!(
                    !expr.is_ok(),
                    "The expr '{}' should not evaulate to anything",
                    $remake_src
                );
            }
        };
        ($test_name:ident, $remake_src:expr, $error_frag:expr) => {
            #[test]
            fn $test_name() {
                let parser = BlockBodyParser::new();
                let lexer = lex::Lexer::new($remake_src);
                let expr = eval(parser.parse(lexer).unwrap());

                match expr {
                    Ok(_) => panic!(
                        "The expr '{}' should not evaulate to anything",
                        $remake_src
                    ),
                    Err(e) => assert!(
                        format!("{}", e.overlay($remake_src))
                            .contains($error_frag),
                        "The expr '{}' must have an error containing '{}'",
                        $remake_src,
                        $error_frag,
                    ),
                }
            }
        };
    }

    eval_to!(basic_int_1_, " 5", Value::Int(5));
    eval_to!(basic_int_2_, " 8  ", Value::Int(8));

    eval_to!(basic_float_1_, " 5.0   ", Value::Float(5.0));
    eval_to!(basic_float_2_, " 5.9", Value::Float(5.9));

    eval_fail!(basic_float_3_, " 5 .9");

    eval_to!(
        basic_str_1_,
        " \"hello\"",
        Value::Str("hello".to_string())
    );
    eval_to!(
        basic_str_2_,
        " \"\" ",
        Value::Str("".to_string())
    );

    eval_to!(basic_bool_1_, " true", Value::Bool(true));
    eval_to!(basic_bool_2_, " false ", Value::Bool(false));

    eval_to!(
        str_1_,
        " \"hello \" . \"world\"",
        Value::Str("hello world".to_string())
    );

    eval_fail!(str_2_, " \"hello \" . 'sup' ", "TypeError");
    eval_fail!(str_3_, " 'regex' . \"str\" ", "TypeError");

    //
    // Primitive Comparisons
    //

    eval_to!(
        prim_cmp_1_,
        " \"aaa\" < \"zzz\" ",
        Value::Bool(true)
    );
    eval_to!(
        prim_cmp_2_,
        " \"aaa\" > \"zzz\" ",
        Value::Bool(false)
    );
    eval_to!(
        prim_cmp_3_,
        " \"aaa\" <= \"zzz\" ",
        Value::Bool(true)
    );
    eval_to!(
        prim_cmp_4_,
        " \"aaa\" >= \"zzz\" ",
        Value::Bool(false)
    );
    eval_to!(
        prim_cmp_5_,
        " \"aaa\" == \"zzz\" ",
        Value::Bool(false)
    );
    eval_to!(
        prim_cmp_6_,
        " \"aaa\" != \"zzz\" ",
        Value::Bool(true)
    );

    eval_to!(prim_cmp_7_, " 9 < 15 ", Value::Bool(true));
    eval_to!(prim_cmp_8_, " 9 > 15 ", Value::Bool(false));
    eval_to!(prim_cmp_9_, " 9 <= 15 ", Value::Bool(true));
    eval_to!(prim_cmp_10_, " 9 >= 15 ", Value::Bool(false));
    eval_to!(prim_cmp_11_, " 9 == 15 ", Value::Bool(false));
    eval_to!(prim_cmp_12_, " 9 != 15 ", Value::Bool(true));

    eval_to!(prim_cmp_13_, " 9.0 < 15.0 ", Value::Bool(true));
    eval_to!(prim_cmp_14_, " 9.0 > 15.0 ", Value::Bool(false));
    eval_to!(prim_cmp_15_, " 9.0 <= 15.0 ", Value::Bool(true));
    eval_to!(
        prim_cmp_16_,
        " 9.0 >= 15.0 ",
        Value::Bool(false)
    );
    eval_to!(
        prim_cmp_17_,
        " 9.0 == 15.0 ",
        Value::Bool(false)
    );
    eval_to!(prim_cmp_18_, " 9.0 != 15.0 ", Value::Bool(true));

    eval_to!(
        prim_cmp_19_,
        " /test/ == 'data' ",
        Value::Bool(false)
    );
    eval_to!(
        prim_cmp_20_,
        " /test/ != /data/ ",
        Value::Bool(true)
    );

    eval_to!(
        prim_cmp_21_,
        " false < true ",
        Value::Bool(true)
    );
    eval_to!(
        prim_cmp_22_,
        " false > true ",
        Value::Bool(false)
    );
    eval_to!(
        prim_cmp_23_,
        " false <= true ",
        Value::Bool(true)
    );
    eval_to!(
        prim_cmp_24_,
        " false >= true ",
        Value::Bool(false)
    );
    eval_to!(
        prim_cmp_25_,
        " false == true ",
        Value::Bool(false)
    );
    eval_to!(
        prim_cmp_26_,
        " false != true ",
        Value::Bool(true)
    );

    eval_fail!(prim_cmp_27_, " false < 1 ", "TypeError");
    eval_fail!(prim_cmp_28_, " false > 1 ", "TypeError");
    eval_fail!(prim_cmp_29_, " false <= 1 ", "TypeError");
    eval_fail!(prim_cmp_30_, " false >= 1 ", "TypeError");
    eval_fail!(prim_cmp_31_, " false == 1 ", "TypeError");
    eval_fail!(prim_cmp_32_, " false != 1 ", "TypeError");

    eval_to!(
        prim_cmp_33_,
        " false || true ",
        Value::Bool(true)
    );
    eval_to!(
        prim_cmp_34_,
        " true && false ",
        Value::Bool(false)
    );

    eval_fail!(prim_cmp_35_, " /regex/ > /regex/ ", "TypeError");
    eval_fail!(prim_cmp_36_, " /regex/ < /regex/ ", "TypeError");
    eval_fail!(
        prim_cmp_37_,
        " /regex/ <= /regex/ ",
        "TypeError"
    );
    eval_fail!(
        prim_cmp_48_,
        " /regex/ >= /regex/ ",
        "TypeError"
    );

    //
    // Arith Ops
    //

    eval_to!(arith_1_, " 1 <+> 2 ", Value::Int(3));
    eval_to!(arith_2_, " 1 - 2 ", Value::Int(-1));
    eval_to!(arith_3_, " 1 </> 2 ", Value::Int(0));
    eval_to!(arith_4_, " 3 % 2 ", Value::Int(1));
    eval_to!(arith_5_, " 1 <*> 2 ", Value::Int(2));

    eval_to!(arith_6_, " 1.0 <+> 2.0 ", Value::Float(3.0));
    eval_to!(arith_7_, " 1.0 - 2.0 ", Value::Float(-1.0));
    eval_to!(arith_8_, " 1.0 </> 2.0 ", Value::Float(0.5));
    eval_to!(arith_9_, " 3.0 % 2.0 ", Value::Float(1.0));
    eval_to!(arith_10_, " 1.0 <*> 2.0 ", Value::Float(2.0));

    eval_fail!(arith_11_, " 1 <+> 2.0 ", "TypeError");
    eval_fail!(arith_12_, " 1.0 - 2 ", "TypeError");
    eval_fail!(arith_13_, " 1 </> 2.0 ", "TypeError");
    eval_fail!(arith_14_, " 3 % 2.0 ", "TypeError");
    eval_fail!(arith_15_, " 1.0 <*> 2 ", "TypeError");
    eval_fail!(arith_16_, " \"str\" <+> 2 ", "TypeError");

    eval_to!(arith_17_, " -2 ", Value::Int(-2));
    eval_to!(arith_18_, " -2.0 ", Value::Float(-2.0));

    eval_fail!(arith_19_, " /re/ <+> 2.0 ", "TypeError");
    eval_fail!(arith_20_, " /re/ - 2 ", "TypeError");
    eval_fail!(arith_21_, " /re/ </> 2.0 ", "TypeError");
    eval_fail!(arith_22_, " 're' % 2.0 ", "TypeError");
    eval_fail!(arith_23_, " 're' <*> 2 ", "TypeError");
    eval_fail!(arith_24_, " 're' <+> 2 ", "TypeError");
}
