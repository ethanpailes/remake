// Copyright 2018 Ethan Pailes.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use regex_syntax;

use error::{ErrorKind, InternalError};
use exec;

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

type Value = Box<regex_syntax::ast::Ast>;

impl Expr {
    pub fn new(kind: ExprKind, span: Span) -> Self {
        Self {
            kind: kind,
            span: span,
        }
    }

    pub fn eval(&self) -> Result<Value, InternalError> {
        let span = self.span.clone();
        match exec::eval(self)? {
            exec::Value::Regex(re) => Ok(re),
            val => Err(InternalError::new(
                ErrorKind::FinalValueNotRegex {
                    actual: val.type_of().to_string(),
                },
                span,
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    BinOp(Box<Expr>, BOp, Box<Expr>),
    UnaryOp(UOp, Box<Expr>),
    Capture(Box<Expr>, Option<String>),
    Block(Vec<Statement>, Box<Expr>),
    Var(String),
    Index(Box<Expr>, Box<Expr>),

    RegexLiteral(Box<regex_syntax::ast::Ast>),
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    BoolLiteral(bool),
    DictLiteral(Vec<(Box<Expr>, Box<Expr>)>),
    TupleLiteral(Vec<Box<Expr>>),

    /// A poison expression is never valid, but it lets us avoid copying
    /// the source string and still please the borrow checker.
    #[doc(hidden)]
    ExprPoison,
}

#[derive(Debug, Clone)]
pub enum BOp {
    Concat,
    Alt,

    Equals,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
    And,
    Or,

    Plus,
    Minus,
    Div,
    Times,
    Mod,
}

#[derive(Debug, Clone)]
pub enum UOp {
    Not,
    Neg,
    RepeatZeroOrMore(bool),
    RepeatOneOrMore(bool),
    RepeatZeroOrOne(bool),
    RepeatRange(bool, regex_syntax::ast::RepetitionRange),
}

#[derive(Debug, Clone)]
pub struct Statement {
    pub kind: StatementKind,
    pub span: Span,
}

impl Statement {
    pub fn new(kind: StatementKind, span: Span) -> Self {
        Statement {
            kind: kind,
            span: span,
        }
    }
}

#[derive(Debug, Clone)]
pub enum StatementKind {
    LetBinding(String, Box<Expr>),
    Assign(String, Box<Expr>),
}

#[derive(Debug, Clone)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}
