// Copyright 2018 Ethan Pailes.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::rc::Rc;

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
    RegexLiteral(Box<regex_syntax::ast::Ast>),
    IntLiteral(i64),
    Lambda {
        expr: Rc<Lambda>,
        // We pre-compute the free variables so that we don't have to
        // do it every time we stamp out a new closure.
        free_vars: Vec<String>,
    },
    Apply {
        func: Box<Expr>,
        args: Vec<Box<Expr>>,
    },

    /// A poison expression is never valid, but it lets us avoid copying
    /// the remake source string and still please the borrow checker.
    #[doc(hidden)]
    ExprPoison,
}

#[derive(Debug, Clone)]
pub struct Lambda {
    pub args: Vec<String>,
    pub body: Box<Expr>,
}

#[derive(Debug, Clone)]
pub enum BOp {
    Concat,
    Alt,

    Plus,
    Minus,
    Div,
    Times,
    Mod,
}

#[derive(Debug, Clone)]
pub enum UOp {
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

// TODO(ethan): represent blocks as single expressions or statements
//              to reduce the surface area where block scope handling
//              needs to be done correctly in the interpreter (spoiler:
//              it's not done right as things stand).
#[derive(Debug, Clone)]
pub enum StatementKind {
    LetBinding(String, Box<Expr>),
    Assign(Box<Expr>, Box<Expr>),
    Expr(Box<Expr>),
    #[allow(dead_code)]
    Block(Vec<Statement>),
}

#[derive(Debug, Clone)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

//
// Visitor
//
// The design is mostly just stolen from the regex crate's Hir
// visitor.
//

pub fn visit_expr<'expr, V: Visitor<'expr>>(
    mut visitor: V,
    expr: &'expr Expr,
) -> Result<V::Output, V::Err> {
    visitor.start();
    let mut heap_visitor = HeapVisitor { stack: vec![] };
    heap_visitor.push_expr(expr);
    visitor.visit_expr_pre(expr)?;

    heap_visitor.run(visitor)
}

#[allow(dead_code)]
pub fn visit_stmt<'stmt, V: Visitor<'stmt>>(
    mut visitor: V,
    stmt: &'stmt Statement,
) -> Result<V::Output, V::Err> {
    visitor.start();
    let mut heap_visitor = HeapVisitor { stack: vec![] };
    heap_visitor.push_stmt(stmt);
    visitor.visit_stmt_pre(stmt)?;

    heap_visitor.run(visitor)
}

/// A trait to encode traversals of the remake AST
pub trait Visitor<'expr> {
    /// The result of the traversal.
    type Output;
    /// An error which might happen during traversal. Traversal
    /// stops as soon as any error is returned.
    type Err;

    /// Finalize the state of the visitor object to get the output
    fn finish(self) -> Result<Self::Output, Self::Err>;

    /// A hook called before beginning traversal of the AST
    fn start(&mut self) {}

    /// Called before visiting expressions
    fn visit_expr_pre(&mut self, _expr: &'expr Expr) -> Result<(), Self::Err> {
        Ok(())
    }
    /// Called after visiting expressions
    fn visit_expr_post(&mut self, _expr: &'expr Expr) -> Result<(), Self::Err> {
        Ok(())
    }

    /// Called before visiting statements
    fn visit_stmt_pre(
        &mut self,
        _expr: &'expr Statement,
    ) -> Result<(), Self::Err> {
        Ok(())
    }
    /// Called after visiting statements
    fn visit_stmt_post(
        &mut self,
        _expr: &'expr Statement,
    ) -> Result<(), Self::Err> {
        Ok(())
    }
}

struct HeapVisitor<'expr> {
    stack: Vec<Frame<'expr>>,
}
enum Frame<'expr> {
    PostExpr(&'expr Expr),
    PreExpr(&'expr Expr),
    PostStmt(&'expr Statement),
    PreStmt(&'expr Statement),
}

impl<'expr> HeapVisitor<'expr> {
    fn run<V: Visitor<'expr>>(
        &mut self,
        mut visitor: V,
    ) -> Result<V::Output, V::Err> {
        loop {
            match self.stack.pop() {
                None => return visitor.finish(),
                Some(Frame::PostExpr(e)) => visitor.visit_expr_post(e)?,
                Some(Frame::PreExpr(e)) => {
                    self.push_expr(e);
                    visitor.visit_expr_pre(e)?;
                }
                Some(Frame::PostStmt(s)) => visitor.visit_stmt_post(s)?,
                Some(Frame::PreStmt(s)) => {
                    self.push_stmt(s);
                    visitor.visit_stmt_pre(s)?;
                }
            }
        }
    }

    /// Push a Post node for the given expression, then a Pre node
    /// for all of its children in reverse order.
    fn push_expr(&mut self, expr: &'expr Expr) {
        self.stack.push(Frame::PostExpr(expr));
        match &expr.kind {
            &ExprKind::BinOp(ref l, _, ref r) => {
                self.stack.push(Frame::PreExpr(&r));
                self.stack.push(Frame::PreExpr(&l));
            }
            &ExprKind::UnaryOp(_, ref e) => {
                self.stack.push(Frame::PreExpr(&e));
            }
            &ExprKind::Capture(ref e, _) => {
                self.stack.push(Frame::PreExpr(&e));
            }
            &ExprKind::Block(ref ss, ref e) => {
                self.stack.push(Frame::PreExpr(&e));
                for s in ss.iter().rev() {
                    self.stack.push(Frame::PreStmt(&s));
                }
            }

            &ExprKind::Lambda {
                ref expr,
                free_vars: _,
            } => {
                self.stack.push(Frame::PreExpr(&expr.body));
            }

            &ExprKind::Apply { ref func, ref args } => {
                for ref a in args.iter().rev() {
                    self.stack.push(Frame::PreExpr(&a));
                }
                self.stack.push(Frame::PreExpr(&func));
            }

            &ExprKind::Var(_)
            | &ExprKind::RegexLiteral(_)
            | &ExprKind::IntLiteral(_)
            | &ExprKind::ExprPoison => {}
        }
    }

    fn push_stmt(&mut self, stmt: &'expr Statement) {
        self.stack.push(Frame::PostStmt(stmt));
        match &stmt.kind {
            &StatementKind::LetBinding(_, ref e) => {
                self.stack.push(Frame::PreExpr(&e));
            }
            &StatementKind::Assign(ref l, ref r) => {
                self.stack.push(Frame::PreExpr(&r));
                self.stack.push(Frame::PreExpr(&l));
            }
            &StatementKind::Expr(ref e) => {
                self.stack.push(Frame::PreExpr(&e));
            }
            &StatementKind::Block(ref body) => {
                for s in body.iter().rev() {
                    self.stack.push(Frame::PreStmt(&s));
                }
            }
        }
    }
}
