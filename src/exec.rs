// Copyright 2018 Ethan Pailes.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::fmt;

use regex_syntax;
use regex_syntax::ast::{GroupKind, RepetitionKind};

use ast;
use ast::{Expr, ExprKind, Statement, StatementKind};
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
    Dict(HashMap<Value, Rc<RefCell<Value>>>),
    Tuple(Vec<Rc<RefCell<Value>>>),
    Vector(Vec<Rc<RefCell<Value>>>),
}

impl Value {
    pub fn type_of(&self) -> &str {
        match self {
            &Value::Regex(_) => "regex",
            &Value::Int(_) => "int",
            &Value::Float(_) => "float",
            &Value::Str(_) => "str",
            &Value::Bool(_) => "bool",
            &Value::Dict(_) => "dict",
            &Value::Tuple(_) => "tuple",
            &Value::Vector(_) => "vec",
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &Value::Int(ref i) => write!(f, "{}", i)?,
            &Value::Float(ref x) => write!(f, "{}", x)?,
            &Value::Str(ref s) => write!(f, "{:?}", s)?,
            &Value::Bool(ref b) => write!(f, "{}", b)?,
            // TODO: real formatting
            &Value::Regex(_) => write!(f, "TODO regex")?,
            &Value::Dict(_) => write!(f, "TODO dict")?,
            &Value::Tuple(_) => write!(f, "TODO tuple")?,
            &Value::Vector(_) => write!(f, "TODO vec")?,
        }

        Ok(())
    }
}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            &Value::Int(ref i) => i.hash(state),
            &Value::Float(ref f) => {
                // Allowing floats as keys because Worse is Better.
                // When performing a lookup with nan as a key we will
                // throw a TypeError just like python does.
                //
                // Argument for saftey: f64 always has 8 bytes.
                let bytes: [u8; 8] = unsafe { ::std::mem::transmute(*f) };
                bytes.hash(state);
            }
            &Value::Str(ref s) => s.hash(state),
            &Value::Bool(ref b) => b.hash(state),

            val => unreachable!("Bug in remake - hash({})", val.type_of()),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        match (self, other) {
            (&Value::Int(ref i1), &Value::Int(ref i2)) => *i1 == *i2,
            (&Value::Float(ref f1), &Value::Float(ref f2)) => {
                // Argument for saftey: f64 always has 8 bytes
                let bytes1: [u8; 8] = unsafe { ::std::mem::transmute(*f1) };
                let bytes2: [u8; 8] = unsafe { ::std::mem::transmute(*f2) };
                bytes1 == bytes2
            }
            (&Value::Str(ref s1), &Value::Str(ref s2)) => *s1 == *s2,
            (&Value::Bool(ref b1), &Value::Bool(ref b2)) => *b1 == *b2,

            _ => unreachable!("Bug in remake - bogus eq"),
        }
    }
}

// bogus Eq implimentation to facilitate our dict implimentation.
impl Eq for Value {}

pub fn eval(expr: &Expr) -> Result<Value, InternalError> {
    eval_(&mut EvalEnv::new(), expr)
        .map(|v| Rc::try_unwrap(v).unwrap().into_inner())
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

macro_rules! type_guard {
    ($val:expr, $span:expr, $($expected:expr),* ) => {
        let ty = $val.type_of();
        if $((ty != $expected) &&)* true {
            return type_error!($val, $span, $($expected),*);
        }
    }
}

macro_rules! key_error {
    ($key:expr, $span:expr) => {
        Err(InternalError::new(
            ErrorKind::KeyError {
                key: $key.to_string()
            },
            $span
            ))
    }
}

fn eval_(
    env: &mut EvalEnv,
    expr: &Expr,
) -> Result<Rc<RefCell<Value>>, InternalError> {
    match expr.kind {
        ExprKind::RegexLiteral(ref r) => ok(Value::Regex(r.clone())),
        ExprKind::BoolLiteral(ref b) => ok(Value::Bool(b.clone())),
        ExprKind::IntLiteral(ref i) => ok(Value::Int(i.clone())),
        ExprKind::FloatLiteral(ref f) => ok(Value::Float(f.clone())),
        ExprKind::StringLiteral(ref s) => ok(Value::Str(s.clone())),
        ExprKind::DictLiteral(ref pairs) => {
            let mut h = if pairs.len() == 0 {
                HashMap::new()
            } else {
                HashMap::with_capacity(pairs.len() * 2)
            };

            for &(ref k_expr, ref v_expr) in pairs.iter() {
                let k_val = eval_(env, &k_expr)?;
                let k_val = k_val.borrow();
                type_guard!(
                    k_val,
                    k_expr.span.clone(),
                    "str",
                    "int",
                    "float",
                    "bool"
                );

                let v_val = eval_(env, &v_expr)?;

                h.insert(k_val.deref().clone(), v_val);
            }

            ok(Value::Dict(h))
        }
        ExprKind::TupleLiteral(ref es) => {
            debug_assert!(es.len() != 0);
            let mut vs = Vec::with_capacity(es.len());

            for v_expr in es.iter() {
                vs.push(eval_(env, &v_expr)?);
            }

            ok(Value::Tuple(vs))
        }
        ExprKind::VectorLiteral(ref es) => {
            let mut vs = if es.len() == 0 {
                Vec::new()
            } else {
                Vec::with_capacity(es.len())
            };

            for v_expr in es.iter() {
                vs.push(eval_(env, &v_expr)?);
            }

            ok(Value::Vector(vs))
        }

        ExprKind::BinOp(ref l_expr, ref op, ref r_expr) => match op {
            &ast::BOp::Concat => {
                let l_val = eval_(env, l_expr)?;
                let l_val = l_val.borrow();
                type_guard!(l_val, l_expr.span.clone(), "regex", "str");

                let r_val = eval_(env, r_expr)?;
                let r_val = r_val.borrow();

                match (l_val.deref(), r_val.deref()) {
                    (&Value::Regex(ref l), &Value::Regex(ref r)) => {
                        ok(Value::Regex(re_operators::concat(
                            l.clone(),
                            r.clone(),
                        )))
                    }
                    (&Value::Regex(_), _) => {
                        type_error!(r_val, r_expr.span.clone(), "regex")
                    }

                    (&Value::Str(ref l), &Value::Str(ref r)) => {
                        let mut s = String::with_capacity(l.len() + r.len());
                        s.push_str(l);
                        s.push_str(r);
                        ok(Value::Str(s))
                    }
                    (&Value::Str(_), _) => {
                        type_error!(r_val, r_expr.span.clone(), "str")
                    }

                    _ => unreachable!("Bug in remake - concat."),
                }
            }

            &ast::BOp::Alt => {
                let l_val = eval_(env, l_expr)?;
                let l_val = l_val.borrow();
                type_guard!(l_val, l_expr.span.clone(), "regex");

                let r_val = eval_(env, r_expr)?;
                let r_val = r_val.borrow();

                match (l_val.deref(), r_val.deref()) {
                    (&Value::Regex(ref l), &Value::Regex(ref r)) => ok(
                        Value::Regex(re_operators::alt(l.clone(), r.clone())),
                    ),
                    (&Value::Regex(_), _) => {
                        type_error!(r_val, r_expr.span.clone(), "regex")
                    }

                    _ => unreachable!("Bug in remake - alt"),
                }
            }

            // comparison operators
            &ast::BOp::Equals => {
                ok(Value::Bool(eval_equals(env, l_expr, r_expr)?))
            }
            &ast::BOp::Ne => {
                ok(Value::Bool(!eval_equals(env, l_expr, r_expr)?))
            }
            &ast::BOp::Lt => ok(Value::Bool(eval_lt(env, l_expr, r_expr)?)),
            &ast::BOp::Gt => ok(Value::Bool(eval_gt(env, l_expr, r_expr)?)),
            &ast::BOp::Le => ok(Value::Bool(eval_le(env, l_expr, r_expr)?)),
            &ast::BOp::Ge => ok(Value::Bool(eval_ge(env, l_expr, r_expr)?)),
            &ast::BOp::Or => {
                let l_val = eval_(env, l_expr)?;
                let l_val = l_val.borrow();
                type_guard!(l_val, l_expr.span.clone(), "bool");

                let r_val = eval_(env, r_expr)?;
                let r_val = r_val.borrow();

                match (l_val.deref(), r_val.deref()) {
                    (&Value::Bool(ref l), &Value::Bool(ref r)) => {
                        ok(Value::Bool(*l || *r))
                    }
                    (&Value::Bool(_), _) => {
                        type_error!(r_val, r_expr.span.clone(), "bool")
                    }

                    _ => unreachable!("Bug in remake - or"),
                }
            }
            &ast::BOp::And => {
                let l_val = eval_(env, l_expr)?;
                let l_val = l_val.borrow();
                type_guard!(l_val, l_expr.span.clone(), "bool");

                let r_val = eval_(env, r_expr)?;
                let r_val = r_val.borrow();

                match (l_val.deref(), r_val.deref()) {
                    (&Value::Bool(ref l), &Value::Bool(ref r)) => {
                        ok(Value::Bool(*l && *r))
                    }
                    (&Value::Bool(_), _) => {
                        type_error!(r_val, r_expr.span.clone(), "bool")
                    }

                    _ => unreachable!("Bug in remake - and"),
                }
            }

            &ast::BOp::Plus => {
                let l_val = eval_(env, l_expr)?;
                let l_val = l_val.borrow();
                type_guard!(l_val, l_expr.span.clone(), "int", "float");

                let r_val = eval_(env, r_expr)?;
                let r_val = r_val.borrow();

                match (l_val.deref(), r_val.deref()) {
                    (&Value::Int(ref l), &Value::Int(ref r)) => {
                        ok(Value::Int(*l + *r))
                    }
                    (&Value::Int(_), _) => {
                        type_error!(r_val, r_expr.span.clone(), "int")
                    }

                    (&Value::Float(ref l), &Value::Float(ref r)) => {
                        ok(Value::Float(*l + *r))
                    }
                    (&Value::Float(_), _) => {
                        type_error!(r_val, r_expr.span.clone(), "float")
                    }

                    _ => unreachable!("Bug in remake - plus"),
                }
            }
            &ast::BOp::Minus => {
                let l_val = eval_(env, l_expr)?;
                let l_val = l_val.borrow();
                type_guard!(l_val, l_expr.span.clone(), "int", "float");

                let r_val = eval_(env, r_expr)?;
                let r_val = r_val.borrow();

                match (l_val.deref(), r_val.deref()) {
                    (&Value::Int(ref l), &Value::Int(ref r)) => {
                        ok(Value::Int(*l - *r))
                    }
                    (&Value::Int(_), _) => {
                        type_error!(r_val, r_expr.span.clone(), "int")
                    }

                    (&Value::Float(ref l), &Value::Float(ref r)) => {
                        ok(Value::Float(*l - *r))
                    }
                    (&Value::Float(_), _) => {
                        type_error!(r_val, r_expr.span.clone(), "float")
                    }

                    _ => unreachable!("Bug in remake - minus"),
                }
            }
            &ast::BOp::Div => {
                let l_val = eval_(env, l_expr)?;
                let l_val = l_val.borrow();
                type_guard!(l_val, l_expr.span.clone(), "int", "float");

                let r_val = eval_(env, r_expr)?;
                let r_val = r_val.borrow();

                match (l_val.deref(), r_val.deref()) {
                    (&Value::Int(ref l), &Value::Int(ref r)) => {
                        if *r == 0 {
                            return Err(InternalError::new(
                                ErrorKind::ZeroDivisionError {
                                    neum: format!("{}", l),
                                },
                                expr.span.clone(),
                            ));
                        }
                        ok(Value::Int(*l / *r))
                    }
                    (&Value::Int(_), _) => {
                        type_error!(r_val, r_expr.span.clone(), "int")
                    }

                    (&Value::Float(ref l), &Value::Float(ref r)) => {
                        if *r == 0.0 {
                            return Err(InternalError::new(
                                ErrorKind::ZeroDivisionError {
                                    neum: format!("{}", l),
                                },
                                expr.span.clone(),
                            ));
                        }
                        ok(Value::Float(*l / *r))
                    }
                    (&Value::Float(_), _) => {
                        type_error!(r_val, r_expr.span.clone(), "float")
                    }

                    _ => unreachable!("Bug in remake - div"),
                }
            }
            &ast::BOp::Times => {
                let l_val = eval_(env, l_expr)?;
                let l_val = l_val.borrow();
                type_guard!(l_val, l_expr.span.clone(), "int", "float");

                let r_val = eval_(env, r_expr)?;
                let r_val = r_val.borrow();

                match (l_val.deref(), r_val.deref()) {
                    (&Value::Int(ref l), &Value::Int(ref r)) => {
                        ok(Value::Int((*l) * (*r)))
                    }
                    (&Value::Int(_), _) => {
                        type_error!(r_val, r_expr.span.clone(), "int")
                    }

                    (&Value::Float(ref l), &Value::Float(ref r)) => {
                        ok(Value::Float((*l) * (*r)))
                    }
                    (&Value::Float(_), _) => {
                        type_error!(r_val, r_expr.span.clone(), "float")
                    }

                    _ => unreachable!("Bug in remake - times"),
                }
            }
            &ast::BOp::Mod => {
                let l_val = eval_(env, l_expr)?;
                let l_val = l_val.borrow();
                type_guard!(l_val, l_expr.span.clone(), "int", "float");

                let r_val = eval_(env, r_expr)?;
                let r_val = r_val.borrow();

                match (l_val.deref(), r_val.deref()) {
                    (&Value::Int(ref l), &Value::Int(ref r)) => {
                        ok(Value::Int(*l % *r))
                    }
                    (&Value::Int(_), _) => {
                        type_error!(r_val, r_expr.span.clone(), "int")
                    }

                    (&Value::Float(ref l), &Value::Float(ref r)) => {
                        ok(Value::Float(*l % *r))
                    }
                    (&Value::Float(_), _) => {
                        type_error!(r_val, r_expr.span.clone(), "float")
                    }

                    _ => unreachable!("Bug in remake - mod"),
                }
            }
        },

        ExprKind::UnaryOp(ref op, ref e) => {
            let e_val = eval_(env, e)?;
            let e_val = e_val.borrow();

            match op {
                &ast::UOp::Not => match e_val.deref() {
                    &Value::Bool(ref b) => ok(Value::Bool(!*b)),
                    _ => type_error!(e_val, e.span.clone(), "bool"),
                },
                &ast::UOp::Neg => match e_val.deref() {
                    &Value::Int(ref i) => ok(Value::Int(-*i)),
                    &Value::Float(ref f) => ok(Value::Float(-*f)),
                    _ => type_error!(e_val, e.span.clone(), "int", "float"),
                },
                &ast::UOp::RepeatZeroOrMore(ref greedy) => {
                    match e_val.deref() {
                        &Value::Regex(ref re) => {
                            ok(rep_zero_or_more(re.clone(), *greedy))
                        }
                        _ => type_error!(e_val, e.span.clone(), "regex"),
                    }
                }
                &ast::UOp::RepeatOneOrMore(ref greedy) => match e_val.deref() {
                    &Value::Regex(ref re) => {
                        ok(rep_one_or_more(re.clone(), *greedy))
                    }
                    _ => type_error!(e_val, e.span.clone(), "regex"),
                },
                &ast::UOp::RepeatZeroOrOne(ref greedy) => match e_val.deref() {
                    &Value::Regex(ref re) => {
                        ok(rep_zero_or_one(re.clone(), *greedy))
                    }
                    _ => type_error!(e_val, e.span.clone(), "regex"),
                },
                &ast::UOp::RepeatRange(ref greedy, ref range) => {
                    match e_val.deref() {
                        &Value::Regex(ref re) => {
                            ok(rep_range(re.clone(), *greedy, range.clone()))
                        }
                        _ => type_error!(e_val, e.span.clone(), "regex"),
                    }
                }
            }
        }

        ExprKind::Capture(ref e, ref name) => {
            let e_val = eval_(env, e)?;
            let e_val = e_val.borrow();

            match e_val.deref() {
                &Value::Regex(ref re) => ok(capture(re.clone(), name.clone())),
                _ => type_error!(e_val, e.span.clone(), "regex"),
            }
        }

        ExprKind::Block(ref statements, ref value) => {
            env.push_block_env();
            for s in statements {
                exec(env, s)?;
            }
            let res = eval_(env, value)?;
            env.pop_block_env();

            Ok(res)
        }

        ExprKind::Var(ref var) => env.lookup(var)
            .map_err(|e| InternalError::new(e, expr.span.clone())),

        ExprKind::Index(ref collection, ref key) => {
            let c_val = eval_(env, collection)?;
            let c_val = c_val.borrow();

            match c_val.deref() {
                &Value::Dict(ref d) => {
                    let k_val = eval_(env, key)?;
                    println!("d={:?}", d);

                    // weird borrow games so we can move k_val later
                    {
                        let k_valb = k_val.borrow();
                        type_guard!(
                            k_valb,
                            key.span.clone(),
                            "str",
                            "int",
                            "float",
                            "bool"
                        );

                        match d.get(k_valb.deref()) {
                            Some(v) => return Ok(v.clone()),
                            None => {} // FALLTHROUGH
                        }
                    }

                    ok(Value::Tuple(vec![
                        Rc::new(RefCell::new(Value::Str("err".to_string()))),
                        Rc::new(RefCell::new(Value::Str(
                            "KeyError".to_string(),
                        ))),
                        k_val,
                    ]))
                }
                &Value::Tuple(ref t) => {
                    let k_val = eval_(env, key)?;

                    // weird borrow games so we can move k_val later
                    {
                        let k_valb = k_val.borrow();

                        match k_valb.deref() {
                            &Value::Int(ref i) => {
                                match t.get(*i as usize) {
                                    Some(v) => return Ok(v.clone()),
                                    None => {} // FALLTHROUGH
                                }
                            }
                            _ => {
                                return type_error!(
                                    k_valb,
                                    key.span.clone(),
                                    "int"
                                )
                            }
                        }
                    }

                    ok(Value::Tuple(vec![
                        Rc::new(RefCell::new(Value::Str("err".to_string()))),
                        Rc::new(RefCell::new(Value::Str(
                            "KeyError".to_string(),
                        ))),
                        k_val,
                    ]))
                }
                &Value::Vector(ref v) => {
                    let k_val = eval_(env, key)?;

                    // weird borrow games so we can move k_val later
                    {
                        let k_valb = k_val.borrow();

                        match k_valb.deref() {
                            &Value::Int(ref i) => {
                                match v.get(*i as usize) {
                                    Some(v) => return Ok(v.clone()),
                                    None => {} // FALLTHROUGH
                                }
                            }
                            _ => {
                                return type_error!(
                                    k_valb,
                                    key.span.clone(),
                                    "int"
                                )
                            }
                        }
                    }

                    ok(Value::Tuple(vec![
                        Rc::new(RefCell::new(Value::Str("err".to_string()))),
                        Rc::new(RefCell::new(Value::Str(
                            "KeyError".to_string(),
                        ))),
                        k_val,
                    ]))
                }
                _ => type_error!(
                    c_val,
                    collection.span.clone(),
                    "dict",
                    "tuple",
                    "vec"
                ),
            }
        }

        ExprKind::IndexSlice {
            ref collection,
            ref start,
            ref end,
        } => eval_slice(env, collection, start, end),

        ExprKind::If {
            ref condition,
            ref true_branch,
            ref false_branch,
        } => {
            let c_val = eval_(env, condition)?;
            let c_val = c_val.borrow();

            match c_val.deref() {
                &Value::Bool(ref b) => {
                    if *b {
                        eval_(env, true_branch)
                    } else {
                        eval_(env, false_branch)
                    }
                }
                _ => type_error!(c_val, condition.span.clone(), "bool"),
            }
        }

        ExprKind::ExprPoison => panic!("Bug in remake - poison expr"),
    }
}

fn exec(env: &mut EvalEnv, s: &Statement) -> Result<(), InternalError> {
    match s.kind {
        StatementKind::LetBinding(ref id, ref e) => {
            let v = eval_(env, &*e)?;
            env.bind(id.clone(), v);
            Ok(())
        }

        StatementKind::Assign(ref lvalue, ref e) => {
            match lvalue.kind {
                ExprKind::Var(ref var) => {
                    let span = e.span.clone();
                    let v = eval_(env, &e)?;
                    env.set(var.clone(), v)
                        .map_err(|e| InternalError::new(e, span))
                }
                ExprKind::Index(ref collection, ref key) => {
                    let c_val = eval_(env, &collection)?;
                    let mut c_val = c_val.borrow_mut();

                    let k_val = eval_(env, &key)?;

                    match c_val.deref_mut() {
                        &mut Value::Tuple(ref mut v)
                        | &mut Value::Vector(ref mut v) => {
                            let k_val = k_val.borrow();

                            return match k_val.deref() {
                                &Value::Int(ref i) => {
                                    if 0 <= *i && (*i as usize) < v.len() {
                                        v[*i as usize] = eval_(env, &e)?;
                                        Ok(())
                                    } else {
                                        key_error!(k_val, key.span.clone())
                                    }
                                }
                                _ => {
                                    type_error!(k_val, key.span.clone(), "int")
                                }
                            };
                        }

                        &mut Value::Dict(ref mut d) => {
                            let k_val = k_val.borrow();
                            type_guard!(
                                k_val,
                                key.span.clone(),
                                "str",
                                "int",
                                "float",
                                "bool"
                            );

                            d.insert(k_val.deref().clone(), eval_(env, &e)?);

                            return Ok(());
                        }

                        _ => {} // FALLTHROUGH: to please the borrow chk
                    };

                    type_error!(
                        c_val,
                        collection.span.clone(),
                        "dict",
                        "tuple",
                        "vec"
                    )
                }
                _ => unreachable!("Bug in remake - assign to non-lvalue"),
            }
        }

        StatementKind::IfElse {
            ref condition,
            ref true_branch,
            ref false_branch,
        } => {
            let c_val = eval_(env, condition)?;
            let c_val = c_val.borrow();

            match c_val.deref() {
                &Value::Bool(ref b) => {
                    if *b {
                        for s in true_branch {
                            exec(env, s)?;
                        }
                    } else {
                        for s in false_branch {
                            exec(env, s)?;
                        }
                    }

                    Ok(())
                }
                _ => type_error!(c_val, condition.span.clone(), "bool"),
            }
        }

        StatementKind::IfTrue {
            ref condition,
            ref true_branch,
        } => {
            let c_val = eval_(env, condition)?;
            let c_val = c_val.borrow();

            match c_val.deref() {
                &Value::Bool(ref b) => {
                    if *b {
                        for s in true_branch {
                            exec(env, s)?;
                        }
                    }

                    Ok(())
                }
                _ => type_error!(c_val, condition.span.clone(), "bool"),
            }
        }

        StatementKind::Expr(ref e) => {
            eval_(env, &e)?;
            Ok(())
        }
    }
}

//
// Utils
//

fn eval_slice(
    env: &mut EvalEnv,
    collection: &Expr,
    start: &Option<Box<Expr>>,
    end: &Option<Box<Expr>>,
) -> Result<Rc<RefCell<Value>>, InternalError> {
    let c_val = eval_(env, collection)?;
    let c_val = c_val.borrow();

    // given an endpoint, return an index that is garenteed to be in range.
    fn normalize_slice_index(v: &Vec<Rc<RefCell<Value>>>, i: i64) -> usize {
        let idx = if i < 0 { (v.len() as i64) + i } else { i };

        if idx < 0 {
            0
        } else {
            if idx as usize > v.len() {
                v.len()
            } else {
                idx as usize
            }
        }
    }

    match c_val.deref() {
        &Value::Vector(ref v) => match (start, end) {
            (&Some(ref start), &Some(ref end)) => {
                let s_val = eval_(env, start)?;
                let s_val = s_val.borrow();
                type_guard!(s_val, collection.span.clone(), "int");

                let e_val = eval_(env, end)?;
                let e_val = e_val.borrow();
                type_guard!(e_val, collection.span.clone(), "int");

                match (s_val.deref(), e_val.deref()) {
                    (&Value::Int(ref s), &Value::Int(ref e)) => {
                        let s = normalize_slice_index(v, *s);
                        let e = normalize_slice_index(v, *e);

                        let mut v_new = Vec::with_capacity((e - s) * 2);
                        for elem in v[s..e].iter() {
                            v_new.push(elem.clone());
                        }

                        ok(Value::Vector(v_new))
                    }

                    (&Value::Int(_), _) => {
                        type_error!(e_val, end.span.clone(), "int")
                    }
                    _ => type_error!(s_val, start.span.clone(), "int"),
                }
            }
            (&Some(ref start), &None) => {
                let s_val = eval_(env, start)?;
                let s_val = s_val.borrow();
                type_guard!(s_val, collection.span.clone(), "int");

                match s_val.deref() {
                    &Value::Int(ref s) => {
                        let s = normalize_slice_index(v, *s);
                        let e = v.len();

                        let mut v_new = Vec::with_capacity((e - s) * 2);
                        for elem in v[s..e].iter() {
                            v_new.push(elem.clone());
                        }

                        ok(Value::Vector(v_new))
                    }

                    _ => type_error!(s_val, start.span.clone(), "int"),
                }
            }
            (&None, &Some(ref end)) => {
                let e_val = eval_(env, end)?;
                let e_val = e_val.borrow();
                type_guard!(e_val, collection.span.clone(), "int");

                match e_val.deref() {
                    &Value::Int(ref e) => {
                        let e = normalize_slice_index(v, *e);

                        let mut v_new = Vec::with_capacity(e * 2);
                        for elem in v[0..e].iter() {
                            v_new.push(elem.clone());
                        }

                        ok(Value::Vector(v_new))
                    }

                    _ => type_error!(e_val, end.span.clone(), "int"),
                }
            }

            (&None, &None) => {
                let mut v_new = Vec::with_capacity(v.len() * 2);
                for elem in v.iter() {
                    v_new.push(elem.clone());
                }

                ok(Value::Vector(v_new))
            }
        },
        _ => type_error!(c_val, collection.span.clone(), "vec"),
    }
}

fn eval_equals(
    env: &mut EvalEnv,
    l_expr: &Expr,
    r_expr: &Expr,
) -> Result<bool, InternalError> {
    let l_val = eval_(env, l_expr)?;
    let l_val = l_val.borrow();

    let r_val = eval_(env, r_expr)?;
    let r_val = r_val.borrow();

    fn eq(
        lhs: &Value,
        rhs: &Value,
        r_expr: &Expr,
    ) -> Result<bool, InternalError> {
        match (lhs, rhs) {
            (&Value::Regex(ref l), &Value::Regex(ref r)) => Ok(*l == *r),
            (&Value::Regex(_), _) => {
                type_error!(rhs, r_expr.span.clone(), "regex")
            }

            (&Value::Str(ref l), &Value::Str(ref r)) => Ok(*l == *r),
            (&Value::Str(_), _) => type_error!(rhs, r_expr.span.clone(), "str"),

            (&Value::Int(ref l), &Value::Int(ref r)) => Ok(*l == *r),
            (&Value::Int(_), _) => type_error!(rhs, r_expr.span.clone(), "int"),

            (&Value::Float(ref l), &Value::Float(ref r)) => {
                Ok((*l - *r).abs() < FLOAT_EQ_EPSILON)
            }
            (&Value::Float(_), _) => {
                type_error!(rhs, r_expr.span.clone(), "float")
            }

            (&Value::Bool(ref l), &Value::Bool(ref r)) => Ok(*l == *r),
            (&Value::Bool(_), _) => {
                type_error!(rhs, r_expr.span.clone(), "bool")
            }

            (&Value::Dict(ref l), &Value::Dict(ref r)) => {
                if l.len() != r.len() {
                    return Ok(false);
                }

                for (k, v1) in l.iter() {
                    match r.get(k) {
                        None => return Ok(false),
                        Some(v2) => {
                            let v1b = v1.borrow();
                            let v2b = v2.borrow();
                            if !eq(v1b.deref(), v2b.deref(), r_expr)
                                .unwrap_or(false)
                            {
                                return Ok(false);
                            }
                        }
                    }
                }

                Ok(true)
            }
            (&Value::Dict(_), _) => {
                type_error!(rhs, r_expr.span.clone(), "dict")
            }

            (&Value::Tuple(ref l), &Value::Tuple(ref r)) => {
                if l.len() != r.len() {
                    return Ok(false);
                }

                for (lv, rv) in l.iter().zip(r.iter()) {
                    let lvb = lv.borrow();
                    let rvb = rv.borrow();

                    if !eq(lvb.deref(), rvb.deref(), r_expr).unwrap_or(false) {
                        return Ok(false);
                    }
                }

                Ok(true)
            }
            (&Value::Tuple(_), _) => {
                type_error!(rhs, r_expr.span.clone(), "tuple")
            }

            (&Value::Vector(ref l), &Value::Vector(ref r)) => {
                if l.len() != r.len() {
                    return Ok(false);
                }

                for (lv, rv) in l.iter().zip(r.iter()) {
                    let lvb = lv.borrow();
                    let rvb = rv.borrow();

                    if !eq(lvb.deref(), rvb.deref(), r_expr).unwrap_or(false) {
                        return Ok(false);
                    }
                }

                Ok(true)
            }
            (&Value::Vector(_), _) => {
                type_error!(rhs, r_expr.span.clone(), "vec")
            }
        }
    };

    eq(l_val.deref(), r_val.deref(), r_expr)
}

fn eval_lt(
    env: &mut EvalEnv,
    l_expr: &Expr,
    r_expr: &Expr,
) -> Result<bool, InternalError> {
    let l_val = eval_(env, l_expr)?;
    let l_val = l_val.borrow();
    type_guard!(l_val, l_expr.span.clone(), "int", "float", "str", "bool");

    let r_val = eval_(env, r_expr)?;
    let r_val = r_val.borrow();

    match (l_val.deref(), r_val.deref()) {
        (&Value::Str(ref l), &Value::Str(ref r)) => Ok(l < r),
        (&Value::Str(_), _) => type_error!(r_val, r_expr.span.clone(), "str"),

        (&Value::Int(ref l), &Value::Int(ref r)) => Ok(l < r),
        (&Value::Int(_), _) => type_error!(r_val, r_expr.span.clone(), "int"),

        (&Value::Float(ref l), &Value::Float(ref r)) => Ok(l < r),
        (&Value::Float(_), _) => {
            type_error!(r_val, r_expr.span.clone(), "float")
        }

        (&Value::Bool(ref l), &Value::Bool(ref r)) => Ok(l < r),
        (&Value::Bool(_), _) => type_error!(r_val, r_expr.span.clone(), "bool"),

        _ => unreachable!("Bug in remake - lt"),
    }
}

fn eval_gt(
    env: &mut EvalEnv,
    l_expr: &Expr,
    r_expr: &Expr,
) -> Result<bool, InternalError> {
    let l_val = eval_(env, l_expr)?;
    let l_val = l_val.borrow();
    type_guard!(l_val, l_expr.span.clone(), "int", "float", "str", "bool");

    let r_val = eval_(env, r_expr)?;
    let r_val = r_val.borrow();

    match (l_val.deref(), r_val.deref()) {
        (&Value::Str(ref l), &Value::Str(ref r)) => Ok(l > r),
        (&Value::Str(_), _) => type_error!(r_val, r_expr.span.clone(), "str"),

        (&Value::Int(ref l), &Value::Int(ref r)) => Ok(l > r),
        (&Value::Int(_), _) => type_error!(r_val, r_expr.span.clone(), "int"),

        (&Value::Float(ref l), &Value::Float(ref r)) => Ok(l > r),
        (&Value::Float(_), _) => {
            type_error!(r_val, r_expr.span.clone(), "float")
        }

        (&Value::Bool(ref l), &Value::Bool(ref r)) => Ok(l > r),
        (&Value::Bool(_), _) => type_error!(r_val, r_expr.span.clone(), "bool"),

        _ => unreachable!("Bug in remake - gt"),
    }
}

fn eval_le(
    env: &mut EvalEnv,
    l_expr: &Expr,
    r_expr: &Expr,
) -> Result<bool, InternalError> {
    let l_val = eval_(env, l_expr)?;
    let l_val = l_val.borrow();
    type_guard!(l_val, l_expr.span.clone(), "int", "float", "str", "bool");

    let r_val = eval_(env, r_expr)?;
    let r_val = r_val.borrow();

    match (l_val.deref(), r_val.deref()) {
        (&Value::Str(ref l), &Value::Str(ref r)) => Ok(l <= r),
        (&Value::Str(_), _) => type_error!(r_val, r_expr.span.clone(), "str"),

        (&Value::Int(ref l), &Value::Int(ref r)) => Ok(l <= r),
        (&Value::Int(_), _) => type_error!(r_val, r_expr.span.clone(), "int"),

        (&Value::Float(ref l), &Value::Float(ref r)) => Ok(l <= r),
        (&Value::Float(_), _) => {
            type_error!(r_val, r_expr.span.clone(), "float")
        }

        (&Value::Bool(ref l), &Value::Bool(ref r)) => Ok(l <= r),
        (&Value::Bool(_), _) => type_error!(r_val, r_expr.span.clone(), "bool"),

        _ => unreachable!("Bug in remake - le"),
    }
}

fn eval_ge(
    env: &mut EvalEnv,
    l_expr: &Expr,
    r_expr: &Expr,
) -> Result<bool, InternalError> {
    let l_val = eval_(env, l_expr)?;
    let l_val = l_val.borrow();
    type_guard!(l_val, l_expr.span.clone(), "int", "float", "str", "bool");

    let r_val = eval_(env, r_expr)?;
    let r_val = r_val.borrow();

    match (l_val.deref(), r_val.deref()) {
        (&Value::Str(ref l), &Value::Str(ref r)) => Ok(l >= r),
        (&Value::Str(_), _) => type_error!(r_val, r_expr.span.clone(), "str"),

        (&Value::Int(ref l), &Value::Int(ref r)) => Ok(l >= r),
        (&Value::Int(_), _) => type_error!(r_val, r_expr.span.clone(), "int"),

        (&Value::Float(ref l), &Value::Float(ref r)) => Ok(l >= r),
        (&Value::Float(_), _) => {
            type_error!(r_val, r_expr.span.clone(), "float")
        }

        (&Value::Bool(ref l), &Value::Bool(ref r)) => Ok(l >= r),
        (&Value::Bool(_), _) => type_error!(r_val, r_expr.span.clone(), "bool"),

        _ => unreachable!("Bug in remake - ge"),
    }
}

fn rep_zero_or_more(re: Box<regex_syntax::ast::Ast>, greedy: bool) -> Value {
    Value::Regex(Box::new(regex_syntax::ast::Ast::Repetition(
        regex_syntax::ast::Repetition {
            span: POISON_SPAN,
            op: regex_syntax::ast::RepetitionOp {
                span: POISON_SPAN,
                kind: RepetitionKind::ZeroOrMore,
            },
            greedy: greedy,
            ast: Box::new(noncapturing_group(re)),
        },
    )))
}

fn rep_one_or_more(re: Box<regex_syntax::ast::Ast>, greedy: bool) -> Value {
    Value::Regex(Box::new(regex_syntax::ast::Ast::Repetition(
        regex_syntax::ast::Repetition {
            span: POISON_SPAN,
            op: regex_syntax::ast::RepetitionOp {
                span: POISON_SPAN,
                kind: RepetitionKind::OneOrMore,
            },
            greedy: greedy,
            ast: Box::new(noncapturing_group(re)),
        },
    )))
}

fn rep_zero_or_one(re: Box<regex_syntax::ast::Ast>, greedy: bool) -> Value {
    Value::Regex(Box::new(regex_syntax::ast::Ast::Repetition(
        regex_syntax::ast::Repetition {
            span: POISON_SPAN,
            op: regex_syntax::ast::RepetitionOp {
                span: POISON_SPAN,
                kind: RepetitionKind::ZeroOrOne,
            },
            greedy: greedy,
            ast: Box::new(noncapturing_group(re)),
        },
    )))
}

fn rep_range(
    re: Box<regex_syntax::ast::Ast>,
    greedy: bool,
    range: regex_syntax::ast::RepetitionRange,
) -> Value {
    Value::Regex(Box::new(regex_syntax::ast::Ast::Repetition(
        regex_syntax::ast::Repetition {
            span: POISON_SPAN,
            op: regex_syntax::ast::RepetitionOp {
                span: POISON_SPAN,
                kind: RepetitionKind::Range(range),
            },
            greedy: greedy,
            ast: Box::new(noncapturing_group(re)),
        },
    )))
}

fn capture(re: Box<regex_syntax::ast::Ast>, name: Option<String>) -> Value {
    Value::Regex(Box::new(regex_syntax::ast::Ast::Group(
        regex_syntax::ast::Group {
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
            ast: re,
        },
    )))
}

fn ok(val: Value) -> Result<Rc<RefCell<Value>>, InternalError> {
    Ok(Rc::new(RefCell::new(val)))
}

//
// The evaluation environment
//

struct EvalEnv {
    block_envs: Vec<HashMap<String, Rc<RefCell<Value>>>>,
}
impl EvalEnv {
    fn new() -> Self {
        EvalEnv { block_envs: vec![] }
    }

    fn push_block_env(&mut self) {
        self.block_envs.push(HashMap::new());
    }

    fn pop_block_env(&mut self) {
        self.block_envs.pop();
    }

    fn bind(&mut self, var: String, v: Rc<RefCell<Value>>) {
        let idx = self.block_envs.len() - 1;
        self.block_envs[idx].insert(var, v);
    }

    fn lookup(&self, var: &String) -> Result<Rc<RefCell<Value>>, ErrorKind> {
        for env in self.block_envs.iter().rev() {
            match env.get(var) {
                None => {}
                Some(val) => return Ok(val.clone()),
            }
        }

        Err(ErrorKind::NameError { name: var.clone() })
    }

    fn set(
        &mut self,
        var: String,
        v: Rc<RefCell<Value>>,
    ) -> Result<(), ErrorKind> {
        for env in self.block_envs.iter_mut().rev() {
            if env.contains_key(&var) {
                env.insert(var, v);
                return Ok(());
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
                let expr = match eval(&parser.parse(lexer).unwrap()) {
                    Ok(e) => e,
                    Err(e) => panic!("{}", e.overlay($remake_src)),
                };

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
                let expr = eval(&parser.parse(lexer).unwrap());

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
                match parser.parse(lexer) {
                    Ok(expr) =>
                        match eval(&expr) {
                            Ok(_) => panic!(
                                "The expr '{}' should not evaulate to anything",
                                $remake_src
                            ),
                            Err(e) => {
                                let err = e.overlay($remake_src).to_string();
                                assert!(err.contains($error_frag),
                                    "The expr '{}' must have an error containing '{}' (actually '{}')",
                                    $remake_src,
                                    $error_frag,
                                    err
                                );
                            }
                        }

                    Err(e) => {
                        let err = e.to_string();
                        assert!(err.contains($error_frag),
                            "The expr '{}' must have an error containing '{}' (actually '{}')",
                            $remake_src,
                            $error_frag,
                            err,
                        )
                    }
                }

            }
        };
    }

    eval_to!(basic_int_1_, " 5", Value::Int(5));
    eval_to!(basic_int_2_, " 8  ", Value::Int(8));

    eval_to!(basic_float_1_, " 5.0   ", Value::Float(5.0));
    eval_to!(basic_float_2_, " 5.9", Value::Float(5.9));

    eval_fail!(basic_float_3_, " 5 .9");

    eval_to!(basic_str_1_, " \"hello\"", Value::Str("hello".to_string()));
    eval_to!(basic_str_2_, " \"\" ", Value::Str("".to_string()));

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

    eval_to!(prim_cmp_1_, " \"aaa\" < \"zzz\" ", Value::Bool(true));
    eval_to!(prim_cmp_2_, " \"aaa\" > \"zzz\" ", Value::Bool(false));
    eval_to!(prim_cmp_3_, " \"aaa\" <= \"zzz\" ", Value::Bool(true));
    eval_to!(prim_cmp_4_, " \"aaa\" >= \"zzz\" ", Value::Bool(false));
    eval_to!(prim_cmp_5_, " \"aaa\" == \"zzz\" ", Value::Bool(false));
    eval_to!(prim_cmp_6_, " \"aaa\" != \"zzz\" ", Value::Bool(true));

    eval_to!(prim_cmp_7_, " 9 < 15 ", Value::Bool(true));
    eval_to!(prim_cmp_8_, " 9 > 15 ", Value::Bool(false));
    eval_to!(prim_cmp_9_, " 9 <= 15 ", Value::Bool(true));
    eval_to!(prim_cmp_10_, " 9 >= 15 ", Value::Bool(false));
    eval_to!(prim_cmp_11_, " 9 == 15 ", Value::Bool(false));
    eval_to!(prim_cmp_12_, " 9 != 15 ", Value::Bool(true));

    eval_to!(prim_cmp_13_, " 9.0 < 15.0 ", Value::Bool(true));
    eval_to!(prim_cmp_14_, " 9.0 > 15.0 ", Value::Bool(false));
    eval_to!(prim_cmp_15_, " 9.0 <= 15.0 ", Value::Bool(true));
    eval_to!(prim_cmp_16_, " 9.0 >= 15.0 ", Value::Bool(false));
    eval_to!(prim_cmp_17_, " 9.0 == 15.0 ", Value::Bool(false));
    eval_to!(prim_cmp_18_, " 9.0 != 15.0 ", Value::Bool(true));

    eval_to!(prim_cmp_19_, " /test/ == 'data' ", Value::Bool(false));
    eval_to!(prim_cmp_20_, " /test/ != /data/ ", Value::Bool(true));

    eval_to!(prim_cmp_21_, " false < true ", Value::Bool(true));
    eval_to!(prim_cmp_22_, " false > true ", Value::Bool(false));
    eval_to!(prim_cmp_23_, " false <= true ", Value::Bool(true));
    eval_to!(prim_cmp_24_, " false >= true ", Value::Bool(false));
    eval_to!(prim_cmp_25_, " false == true ", Value::Bool(false));
    eval_to!(prim_cmp_26_, " false != true ", Value::Bool(true));

    eval_fail!(prim_cmp_27_, " false < 1 ", "TypeError");
    eval_fail!(prim_cmp_28_, " false > 1 ", "TypeError");
    eval_fail!(prim_cmp_29_, " false <= 1 ", "TypeError");
    eval_fail!(prim_cmp_30_, " false >= 1 ", "TypeError");
    eval_fail!(prim_cmp_31_, " false == 1 ", "TypeError");
    eval_fail!(prim_cmp_32_, " false != 1 ", "TypeError");

    eval_to!(prim_cmp_33_, " false || true ", Value::Bool(true));
    eval_to!(prim_cmp_34_, " true && false ", Value::Bool(false));

    eval_fail!(prim_cmp_35_, " /regex/ > /regex/ ", "TypeError");
    eval_fail!(prim_cmp_36_, " /regex/ < /regex/ ", "TypeError");
    eval_fail!(prim_cmp_37_, " /regex/ <= /regex/ ", "TypeError");
    eval_fail!(prim_cmp_48_, " /regex/ >= /regex/ ", "TypeError");

    eval_to!(prim_cmp_49_, " !true ", Value::Bool(false));

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

    eval_fail!(arith_25_, " 19 </> 0", "ZeroDivisionError");

    //
    // Assignment
    //

    eval_to!(
        assign_1_,
        r#"
    let x = 1;
    x = 2;
    x
    "#,
        Value::Int(2)
    );

    eval_to!(
        assign_2_,
        r#"
    let x = 1;
    let y = {
        x = 2;
        "thrown out"
    };
    x
    "#,
        Value::Int(2)
    );

    eval_to!(
        assign_3_,
        r#"
    let x = 1;
    {
        x = 2;
        x
    }
    "#,
        Value::Int(2)
    );

    eval_fail!(assign_4_, "x = 1; 2", "NameError");

    eval_fail!(
        assign_5_,
        r#"
    let y = {
        let x = 1;
        "thrown out"
    };
    x = 1;
    2
    "#,
        "NameError"
    );

    //
    // dicts
    //

    eval_to!(dict_1_, " { 1: 2 } == { 1: 2 } ", Value::Bool(true));
    eval_to!(dict_2_, " { 1: 2 } == { 1: 2, 6: 8 } ", Value::Bool(false));
    eval_to!(
        dict_3_,
        " { 1: 2, 6: 9 } == { 1: 2, 6: 8 } ",
        Value::Bool(false)
    );
    eval_to!(
        dict_4_,
        " { 1: 2, 6: 8 } == { 1: 2, 6: 8 } ",
        Value::Bool(true)
    );

    eval_to!(dict_5_, " { 1: 2, 6: 8 }[1] ", Value::Int(2));

    eval_to!(dict_6_, " { 1: 4 }[3][0] ", Value::Str("err".to_string()));

    eval_to!(
        dict_7_,
        r#"
    let x = {0: 1, "how": "exciting" };
    x["how"] = "exciting!";
    x["how"]
    "#,
        Value::Str("exciting!".to_string())
    );

    eval_to!(
        dict_8_,
        r#"
    let x = {0: 1, "hello": "world" };
    x[6] = 5;
    x[6]
    "#,
        Value::Int(5)
    );

    eval_to!(
        dict_9_,
        " { 6: 8, 1: 2 } == { 1: 2, 6: 8 } ",
        Value::Bool(true)
    );
    eval_to!(
        dict_10_,
        " { 6: 8, 1: 2 } == { 1: 2, 6: 8 } ",
        Value::Bool(true)
    );

    eval_to!(dict_11_, " { 6.9: 8, 1: 2 }[6.9] ", Value::Int(8));

    eval_to!(dict_12_, r#" { "hi": 8, 1: 2 }["hi"] "#, Value::Int(8));

    eval_to!(
        dict_13_,
        r#"
        let ht = {};
        ht["hi"] = 5;
        ht["hi"]
        "#,
        Value::Int(5)
    );

    eval_fail!(
        dict_14_,
        r#"
        let ht = {};
        ht[{3:4}] = 5;
        ht
        "#,
        "TypeError"
    );
    eval_fail!(
        dict_15_,
        r#"
        let ht = {};
        ht[ [5, 6, 7] ] = 5;
        ht
        "#,
        "TypeError"
    );

    // TODO(ethan): tuples as keys
    // TODO(ethan): extend dict with other dict (requires functions)
    // TODO(ethan): keys (requires functions)
    // TODO(ethan): items (requires functions)
    // TODO(ethan): builtin containment checks with the `in` keyword

    //
    // tuples
    //

    eval_to!(tuple_1_, " (1, 2)[0] ", Value::Int(1));
    eval_to!(tuple_2_, " (1, 2)[2][0] ", Value::Str("err".to_string()));

    eval_to!(
        tuple_3_,
        " (1, 2)[2][1] ",
        Value::Str("KeyError".to_string())
    );
    eval_to!(tuple_4_, " (1, 2)[2][2] ", Value::Int(2));

    eval_to!(
        tuple_5_,
        r#"
    let x = (1, 2);
    x[0] = 5;
    x[0]
    "#,
        Value::Int(5)
    );

    eval_fail!(
        tuple_6_,
        r#"
    let x = (1, 2);
    x[6] = 5;
    x[0]
    "#,
        "KeyError"
    );

    //
    // vectors
    //

    eval_to!(vec_1_, " [1, 2][0] ", Value::Int(1));
    eval_to!(vec_2_, " [1, 2][2][0] ", Value::Str("err".to_string()));
    eval_to!(vec_3_, " [1, 2][2][1] ", Value::Str("KeyError".to_string()));
    eval_to!(vec_4_, " [1, 2][2][2] ", Value::Int(2));

    eval_to!(vec_5_, " [1, 2, 3][0:1][0] ", Value::Int(1));
    eval_to!(
        vec_6_,
        " [1, 2, 3][0:1][1][1] ",
        Value::Str("KeyError".to_string())
    );

    eval_to!(vec_7_, " [1, 2, 3][0:][2] ", Value::Int(3));
    eval_to!(vec_8_, " [1, 2, 3][1:][0] ", Value::Int(2));
    eval_to!(
        vec_9_,
        " [1, 2, 3][0:-1][2][1] ",
        Value::Str("KeyError".to_string())
    );
    eval_to!(
        vec_10_,
        " [1, 2, 3][1:-1][1][1] ",
        Value::Str("KeyError".to_string())
    );
    eval_to!(
        vec_11_,
        " [1, 2, 3][1:-1][1][1] ",
        Value::Str("KeyError".to_string())
    );
    eval_to!(vec_12_, " [1, 2, 3][1:-1][0] ", Value::Int(2));

    eval_to!(vec_13_, " [1, 2, 3][:-1][0] ", Value::Int(1));
    eval_to!(
        vec_14_,
        " [1, 2, 3][:-1][2][1] ",
        Value::Str("KeyError".to_string())
    );
    eval_to!(vec_15_, " [1, 2, 3][:][2] ", Value::Int(3));
    eval_to!(vec_16_, " [1, 2, 3][:][0] ", Value::Int(1));

    eval_to!(
        vec_17_,
        r#"
    let x = [1, 2];
    x[0] = 5;
    x[0]
    "#,
        Value::Int(5)
    );

    eval_fail!(
        vec_18_,
        r#"
    let x = [1, 2];
    x[6] = 5;
    x[0]
    "#,
        "KeyError"
    );

    // TODO(ethan): append to vector (requires functions)
    // TODO(ethan): extend vector (requires functions)

    //
    // If expressions & statements
    //

    // if expressions
    eval_to!(if_1_, " if (true) { 1 } else { 0 } ", Value::Int(1));
    eval_to!(if_2_, " if (false) { 1 } else { 0 } ", Value::Int(0));
    eval_fail!(if_3_, " if (false) { 1 } ", "Unrecognized EOF"); // parse error
    eval_to!(
        if_4_,
        r#"
    if (false) {
        1
    } else if (4 == 7) {
        2
    } else if (9 < 6) {
        3
    } else if ("hello" == "world") {
        4
    } else if (1 <+> 1 == 2) {
        5
    } else {
        0
    }
    "#,
        Value::Int(5)
    );

    eval_to!(
        if_5_,
        r#"
    let x = 5;
    if (1 == 1) {
        let y = 7;
        y = "this is a dynamically typed language";
        x = 9;
    }
    x
    "#,
        Value::Int(9)
    );

    eval_to!(
        if_6_,
        r#"
    let x = 5;
    if (1 == 2) {
        x = 9;
    } else {
        x = 18;
    }
    x
    "#,
        Value::Int(18)
    );

    eval_to!(
        if_7_,
        r#"
    let x = 5;
    if (1 == 11) {
        x = 9;
    } else if (8 > 3) {
        x = 2;
        x = 4;
    } else {
        x = 10;
    }
    x
    "#,
        Value::Int(4)
    );

    eval_to!(
        if_8_,
        r#"
    let x = 5;
    if (1 == 11) {
        x = 9;
    } else if (8 > 3) {
        x = 2;
        x = 4;
    } else if (false) {
        x = "how";
    } else {
        x = 10;
    }
    x
    "#,
        Value::Int(4)
    );

    eval_to!(
        expr_statement_1_,
        r#"
        let x = 1;
        {
            x = 2;
            x
        };
        x
        "#,
        Value::Int(2)
    );

}
