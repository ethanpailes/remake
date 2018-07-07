// Copyright 2018 Ethan Pailes.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

use regex_syntax;
use regex_syntax::ast::{GroupKind, RepetitionKind};

use ast;
use ast::{Expr, ExprKind, Span, Statement, StatementKind};
use error::{ErrorKind, InternalError, LoopErrorKind};
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
    Closure {
        env: Env,
        lambda: Rc<ast::Lambda>,
    },
    BuiltinFunction(BuiltIn),

    /// A user can never get ahold of an 'Undefined' value, but
    /// we need something to place in the cell of a recursive definition.
    /// If an 'Undefined' is ever discovered during evaluation, we trigger
    /// an immediate 'NameError'
    Undefined,
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
            &Value::Closure { env: _, lambda: _ } => "closure",

            // User's should not have to special case based on the
            // fact that a function is defined as a builtin. If they
            // really care that much they can check the result of show()
            &Value::BuiltinFunction(_) => "closure",

            &Value::Undefined => {
                unreachable!("Bug in remake - undefined type_of")
            }
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
            &Value::Closure { env: _, ref lambda } => {
                write!(
                    f,
                    "<closure with args ({})>",
                    lambda.args.clone().join(", ")
                )?;
            }
            &Value::BuiltinFunction(ref b) => {
                write!(f, "{:?}", b)?;
            }

            &Value::Regex(_) => write!(f, "TODO show regex")?,
            &Value::Dict(ref d) => {
                write!(f, "{{ ")?;
                for (i, (k, v)) in d.iter().enumerate() {
                    let v = v.borrow();
                    write!(f, "{}: {}", k, v.deref())?;
                    if i < d.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, " }}")?;
            }
            &Value::Tuple(ref tup) => {
                write!(f, "(")?;
                for (i, elem) in tup.iter().enumerate() {
                    let elem = elem.borrow();
                    write!(f, "{}", elem.deref())?;
                    if i < tup.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, ")")?;
            }
            &Value::Vector(ref vec) => {
                write!(f, "[")?;
                for (i, elem) in vec.iter().enumerate() {
                    let elem = elem.borrow();
                    write!(f, "{}", elem.deref())?;
                    if i < vec.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "]")?;
            }
            &Value::Undefined => unreachable!("Bug in remake - undefined disp"),
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
    let res;
    {
        let mut env = EvalEnv::new();
        res = eval_(&mut env, expr);
        debug_assert!(env.call_stack.len() == 1);
    }
    res.map(|v| Rc::try_unwrap(v).unwrap().into_inner())
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
                key: $key.to_string(),
            },
            $span,
        ))
    };
}

macro_rules! loop_error {
    ($keyword:expr, $span:expr) => {
        Err(InternalError::new(
            ErrorKind::LoopError { keyword: $keyword },
            $span,
        ))
    };
}

macro_rules! arity_error {
    ($expected:expr, $actual:expr, $span:expr) => {
        Err(InternalError::new(
            ErrorKind::ArityError {
                expected: $expected,
                actual: $actual,
            },
            $span,
        ))
    };
}

macro_rules! try_cleanup {
    ($e:expr, $($cleanup:tt)*) => {
        match $e {
            Ok(inner) => inner,
            Err(err) => {
                {
                    $($cleanup)*
                }
                return Err(From::from(err));
            }
        }
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

            &ast::BOp::In => {
                let l_val = eval_(env, l_expr)?;
                let r_val = eval_(env, r_expr)?;

                let l_val = l_val.borrow();
                let r_val = r_val.borrow();

                match r_val.deref() {
                    &Value::Dict(ref d) => {
                        type_guard!(
                            l_val,
                            l_expr.span.clone(),
                            "str",
                            "int",
                            "float",
                            "bool"
                        );

                        return ok(Value::Bool(d.contains_key(l_val.deref())));
                    }
                    &Value::Vector(ref v) | &Value::Tuple(ref v) => {
                        for elem in v.iter() {
                            let e_val = elem.borrow();
                            if eval_equals_value(l_val.deref(), e_val.deref()) {
                                return ok(Value::Bool(true));
                            }
                        }

                        return ok(Value::Bool(false));
                    }
                    _ => {} // FALLTHROUGH
                }
                type_error!(r_val, r_expr.span.clone(), "dict", "vec", "tuple")
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
                try_cleanup!(exec(env, s), env.pop_block_env());
            }
            let res = try_cleanup!(eval_(env, value), env.pop_block_env());
            env.pop_block_env();

            Ok(res)
        }

        ExprKind::Var(ref var) => env.lookup(var)
            .map_err(|e| InternalError::new(e, expr.span.clone())),

        ExprKind::Index(ref collection, ref key) => {
            let c_val = eval_(env, collection)?;
            {
                let c_val = c_val.borrow();
                type_guard!(
                    c_val,
                    collection.span.clone(),
                    "dict",
                    "tuple",
                    "vec"
                );
            }
            let k_val = eval_(env, key)?;

            let c_val = c_val.borrow();
            match c_val.deref() {
                &Value::Dict(ref d) => {
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
                            Some(v) => return Ok(Rc::clone(v)),
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
                &Value::Tuple(ref v) |
                &Value::Vector(ref v) => {
                    // weird borrow games so we can move k_val later
                    {
                        let k_valb = k_val.borrow();

                        match k_valb.deref() {
                            &Value::Int(ref i) => {
                                match v.get(*i as usize) {
                                    Some(v) => return Ok(Rc::clone(v)),
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
                _ =>
                type_error!(
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

        ExprKind::Lambda {
            ref expr,
            ref free_vars,
        } => {
            let mut closure_env = Env::new();
            for v in free_vars.iter() {
                let e = env.lookup(v).map_err(|e| {
                    InternalError::new(e, expr.body.span.clone())
                })?;
                closure_env.insert(v.clone(), e);
            }

            ok(Value::Closure {
                env: closure_env,
                lambda: expr.clone(),
            })
        }

        ExprKind::Apply { ref func, ref args } => {
            let f_val = eval_(env, func)?;
            let f_val = f_val.borrow();

            match f_val.deref() {
                Value::Closure {
                    env: ref closure_env,
                    ref lambda,
                } => {
                    if lambda.args.len() != args.len() {
                        return arity_error!(
                            lambda.args.len(),
                            args.len(),
                            expr.span.clone()
                        );
                    }

                    let mut new_env = closure_env.clone();
                    let mut iter = args.iter().zip(lambda.args.iter());
                    for (arg_expr, arg_name) in iter {
                        let arg_val = eval_(env, &arg_expr)?;
                        new_env.insert(arg_name.to_string(), arg_val);
                    }

                    env.push_closure_frame(new_env);
                    let ret = eval_(env, &lambda.body);
                    env.pop_stack_frame();
                    ret
                }
                Value::BuiltinFunction(ref builtin) => {
                    let arg_spans =
                        args.iter().map(|a| a.span.clone()).collect::<Vec<_>>();
                    let args = args.iter()
                        .map(|a| eval_(env, a))
                        .collect::<Result<Vec<_>, _>>()?;
                    (builtin.func)(&args, &expr.span, &arg_spans)
                }
                _ => type_error!(f_val, func.span.clone(), "closure"),
            }
        }

        ExprKind::ExprPoison => unreachable!("Bug in remake - poison expr"),
    }
}

fn exec(env: &mut EvalEnv, s: &Statement) -> Result<(), InternalError> {
    match s.kind {
        StatementKind::LetBinding(ref id, ref e) => {
            env.bind(id.clone(), Rc::new(RefCell::new(Value::Undefined)));

            let v = match Rc::try_unwrap(eval_(env, &e)?) {
                Ok(inner) => inner.into_inner(),
                Err(rc) => Rc::try_unwrap(Rc::clone(&rc)).unwrap().into_inner(),
            };

            env.set(id.clone(), v)
                .expect("Bug in remake - set just bound");
            Ok(())
        }

        StatementKind::Assign(ref lvalue, ref e) => {
            match lvalue.kind {
                ExprKind::Var(ref var) => {
                    let span = e.span.clone();
                    let v = match Rc::try_unwrap(eval_(env, &e)?) {
                        Ok(inner) => inner.into_inner(),
                        Err(rc) => {
                            Rc::try_unwrap(Rc::clone(&rc)).unwrap().into_inner()
                        }
                    };
                    let res = env.set(var.clone(), v)
                        .map_err(|e| InternalError::new(e, span));
                    res
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

        StatementKind::For {
            ref variable,
            ref collection,
            ref body,
        } => {
            let v_val = eval_(env, &collection)?;
            let v_val = v_val.borrow();

            match v_val.deref() {
                &Value::Tuple(ref v) | &Value::Vector(ref v) => {
                    'FOR_LOOP: for elem in v.iter() {
                        env.bind(variable.clone(), elem.clone());
                        for s in body.iter() {
                            match exec(env, s) {
                                Err(err) => match err.kind {
                                    ErrorKind::LoopError {
                                        keyword: LoopErrorKind::Continue,
                                    } => continue 'FOR_LOOP,
                                    ErrorKind::LoopError {
                                        keyword: LoopErrorKind::Break,
                                    } => break 'FOR_LOOP,
                                    _ => return Err(err),
                                },
                                Ok(()) => {} // FALLTHROUGH
                            }
                        }
                    }

                    Ok(())
                }
                _ => {
                    type_error!(v_val, collection.span.clone(), "vec", "tuple")
                }
            }
        }

        StatementKind::While {
            ref condition,
            ref body,
        } => {
            'WHILE_LOOP: loop {
                let c_val = eval_(env, &condition)?;
                let c_val = c_val.borrow();

                match c_val.deref() {
                    &Value::Bool(false) => break 'WHILE_LOOP,
                    &Value::Bool(true) => {
                        for s in body.iter() {
                            match exec(env, s) {
                                Err(err) => match err.kind {
                                    ErrorKind::LoopError {
                                        keyword: LoopErrorKind::Continue,
                                    } => continue 'WHILE_LOOP,
                                    ErrorKind::LoopError {
                                        keyword: LoopErrorKind::Break,
                                    } => break 'WHILE_LOOP,
                                    _ => return Err(err),
                                },
                                Ok(()) => {} // FALLTHROUGH
                            }
                        }
                    }
                    _ => {
                        return type_error!(
                            c_val,
                            condition.span.clone(),
                            "bool"
                        )
                    }
                }
            }

            Ok(())
        }

        StatementKind::Continue => {
            loop_error!(LoopErrorKind::Continue, s.span.clone())
        }
        StatementKind::Break => {
            loop_error!(LoopErrorKind::Break, s.span.clone())
        }

        StatementKind::Block(ref statements) => {
            env.push_block_env();
            for s in statements.iter() {
                try_cleanup!(exec(env, s), env.pop_block_env());
            }
            env.pop_block_env();

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

    Ok(eval_equals_value(l_val.deref(), r_val.deref()))
}

fn eval_equals_value(lhs: &Value, rhs: &Value) -> bool {
    match (lhs, rhs) {
        (&Value::Regex(ref l), &Value::Regex(ref r)) => *l == *r,
        (&Value::Regex(_), _) => false,

        (&Value::Str(ref l), &Value::Str(ref r)) => *l == *r,
        (&Value::Str(_), _) => false,

        (&Value::Int(ref l), &Value::Int(ref r)) => *l == *r,
        (&Value::Int(_), _) => false,

        (&Value::Float(ref l), &Value::Float(ref r)) => {
            (*l - *r).abs() < FLOAT_EQ_EPSILON
        }
        (&Value::Float(_), _) => false,

        (&Value::Bool(ref l), &Value::Bool(ref r)) => *l == *r,
        (&Value::Bool(_), _) => false,

        (&Value::Dict(ref l), &Value::Dict(ref r)) => {
            if l.len() != r.len() {
                return false;
            }

            for (k, v1) in l.iter() {
                match r.get(k) {
                    None => return false,
                    Some(v2) => {
                        let v1b = v1.borrow();
                        let v2b = v2.borrow();
                        if !eval_equals_value(v1b.deref(), v2b.deref()) {
                            return false;
                        }
                    }
                }
            }

            true
        }
        (&Value::Dict(_), _) => false,

        (&Value::Vector(ref l), &Value::Vector(ref r))
        | (&Value::Tuple(ref l), &Value::Tuple(ref r)) => {
            if l.len() != r.len() {
                return false;
            }

            for (lv, rv) in l.iter().zip(r.iter()) {
                let lvb = lv.borrow();
                let rvb = rv.borrow();

                if !eval_equals_value(lvb.deref(), rvb.deref()) {
                    return false;
                }
            }

            true
        }
        (&Value::Tuple(_), _) => false,
        (&Value::Vector(_), _) => false,

        // closures are never equal to anything
        (&Value::Closure { env: _, lambda: _ }, _) => false,
        (&Value::BuiltinFunction(_), _) => false,
        (&Value::Undefined, _) => unreachable!("Bug in remake - eq undefined"),
    }
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
        (&Value::Str(_), _) => Ok(false),

        (&Value::Int(ref l), &Value::Int(ref r)) => Ok(l < r),
        (&Value::Int(_), _) => Ok(false),

        (&Value::Float(ref l), &Value::Float(ref r)) => Ok(l < r),
        (&Value::Float(_), _) => Ok(false),

        (&Value::Bool(ref l), &Value::Bool(ref r)) => Ok(l < r),
        (&Value::Bool(_), _) => Ok(false),

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
        (&Value::Str(_), _) => Ok(false),

        (&Value::Int(ref l), &Value::Int(ref r)) => Ok(l > r),
        (&Value::Int(_), _) => Ok(false),

        (&Value::Float(ref l), &Value::Float(ref r)) => Ok(l > r),
        (&Value::Float(_), _) => Ok(false),

        (&Value::Bool(ref l), &Value::Bool(ref r)) => Ok(l > r),
        (&Value::Bool(_), _) => Ok(false),

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
        (&Value::Str(_), _) => Ok(false),

        (&Value::Int(ref l), &Value::Int(ref r)) => Ok(l <= r),
        (&Value::Int(_), _) => Ok(false),

        (&Value::Float(ref l), &Value::Float(ref r)) => Ok(l <= r),
        (&Value::Float(_), _) => Ok(false),

        (&Value::Bool(ref l), &Value::Bool(ref r)) => Ok(l <= r),
        (&Value::Bool(_), _) => Ok(false),

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
        (&Value::Str(_), _) => Ok(false),

        (&Value::Int(ref l), &Value::Int(ref r)) => Ok(l >= r),
        (&Value::Int(_), _) => Ok(false),

        (&Value::Float(ref l), &Value::Float(ref r)) => Ok(l >= r),
        (&Value::Float(_), _) => Ok(false),

        (&Value::Bool(ref l), &Value::Bool(ref r)) => Ok(l >= r),
        (&Value::Bool(_), _) => Ok(false),

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

#[derive(Debug)]
struct EvalEnv {
    call_stack: Vec<CallFrame>,
}
impl EvalEnv {
    fn new() -> Self {
        let mut basis = CallFrame::new();
        basis.push_block_env();
        for builtin in BUILTINS.iter() {
            basis.bind(
                builtin.name.to_string(),
                Rc::new(RefCell::new(Value::BuiltinFunction(builtin.clone()))),
            );
        }
        EvalEnv {
            call_stack: vec![basis],
        }
    }

    fn push_closure_frame(&mut self, env: Env) {
        self.call_stack.push(CallFrame::from_env(env));
    }
    fn pop_stack_frame(&mut self) {
        self.call_stack.pop();
    }

    fn push_block_env(&mut self) {
        self.top_frame_mut().push_block_env()
    }
    fn pop_block_env(&mut self) {
        self.top_frame_mut().pop_block_env()
    }
    fn bind(&mut self, var: String, v: Rc<RefCell<Value>>) {
        self.top_frame_mut().bind(var, v)
    }
    fn lookup(&self, var: &String) -> Result<Rc<RefCell<Value>>, ErrorKind> {
        self.top_frame().lookup(var)
    }
    fn set(&mut self, var: String, v: Value) -> Result<(), ErrorKind> {
        self.top_frame_mut().set(var, v)
    }

    fn top_frame_mut(&mut self) -> &mut CallFrame {
        let len = self.call_stack.len();
        &mut self.call_stack[len - 1]
    }
    fn top_frame(&self) -> &CallFrame {
        let len = self.call_stack.len();
        &self.call_stack[len - 1]
    }
}

#[derive(Debug)]
struct CallFrame {
    block_envs: Vec<Env>,
}
type Env = HashMap<String, Rc<RefCell<Value>>>;
impl CallFrame {
    fn new() -> Self {
        CallFrame { block_envs: vec![] }
    }
    fn from_env(env: Env) -> Self {
        CallFrame {
            block_envs: vec![env],
        }
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
                Some(val) => return Ok(Rc::clone(val)),
            }
        }

        Err(ErrorKind::NameError { name: var.clone() })
    }

    fn set(&mut self, var: String, v: Value) -> Result<(), ErrorKind> {
        for env in self.block_envs.iter_mut().rev() {
            if env.contains_key(&var) {
                env.entry(var).and_modify(|location| {
                    location.replace(v);
                });
                return Ok(());
            }
        }

        Err(ErrorKind::NameError { name: var })
    }
}

/////////////////////////////////////////////////////////////////////////////
//
//                   Initial Basis & Standard Library
//
// TODO(ethan): once you have internet again, figure out how to reuse
//              the error macros, and move this to its own module.
//
/////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct BuiltIn {
    name: &'static str,
    func: fn(&[Rc<RefCell<Value>>], &Span, &[Span])
        -> Result<Rc<RefCell<Value>>, InternalError>,
}

impl fmt::Debug for BuiltIn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "<builtin '{}'>", self.name)?;
        Ok(())
    }
}

const BUILTINS: [BuiltIn; 7] = [
    // dict methods
    BuiltIn {
        name: "keys",
        func: remake_keys,
    },
    BuiltIn {
        name: "values",
        func: remake_values,
    },
    BuiltIn {
        name: "items",
        func: remake_items,
    },
    BuiltIn {
        name: "extend",
        func: remake_extend,
    },
    BuiltIn {
        name: "clear",
        func: remake_clear,
    },
    BuiltIn {
        name: "show",
        func: remake_show,
    },
    // vec methods
    BuiltIn {
        name: "push",
        func: remake_push,
    },
];

fn remake_keys(
    args: &[Rc<RefCell<Value>>],
    apply_span: &Span,
    arg_spans: &[Span],
) -> Result<Rc<RefCell<Value>>, InternalError> {
    debug_assert!(args.len() == arg_spans.len());

    if args.len() != 1 {
        return arity_error!(1, args.len(), apply_span.clone());
    }

    let dict_val = args[0].borrow();
    match dict_val.deref() {
        &Value::Dict(ref d) => {
            // Note that we make a new rc-refcell because keys are
            // immutable within a hashtable.
            let keys = d.keys()
                .map(|k| Rc::new(RefCell::new(k.clone())))
                .collect::<Vec<_>>();
            ok(Value::Vector(keys))
        }
        _ => type_error!(dict_val, arg_spans[0].clone(), "dict"),
    }
}

fn remake_values(
    args: &[Rc<RefCell<Value>>],
    apply_span: &Span,
    arg_spans: &[Span],
) -> Result<Rc<RefCell<Value>>, InternalError> {
    debug_assert!(args.len() == arg_spans.len());

    if args.len() != 1 {
        return arity_error!(1, args.len(), apply_span.clone());
    }

    let dict_val = args[0].borrow();
    match dict_val.deref() {
        &Value::Dict(ref d) => {
            let values = d.values().map(|v| Rc::clone(v)).collect::<Vec<_>>();
            ok(Value::Vector(values))
        }
        _ => type_error!(dict_val, arg_spans[0].clone(), "dict"),
    }
}

fn remake_items(
    args: &[Rc<RefCell<Value>>],
    apply_span: &Span,
    arg_spans: &[Span],
) -> Result<Rc<RefCell<Value>>, InternalError> {
    debug_assert!(args.len() == arg_spans.len());

    if args.len() != 1 {
        return arity_error!(1, args.len(), apply_span.clone());
    }

    let dict_val = args[0].borrow();
    match dict_val.deref() {
        &Value::Dict(ref d) => {
            let items = d.iter()
                .map(|(k, v)| {
                    Rc::new(RefCell::new(Value::Tuple(vec![
                        Rc::new(RefCell::new(k.clone())),
                        Rc::clone(v),
                    ])))
                })
                .collect::<Vec<_>>();
            ok(Value::Vector(items))
        }
        _ => type_error!(dict_val, arg_spans[0].clone(), "dict"),
    }
}

fn remake_clear(
    args: &[Rc<RefCell<Value>>],
    apply_span: &Span,
    arg_spans: &[Span],
) -> Result<Rc<RefCell<Value>>, InternalError> {
    debug_assert!(args.len() == arg_spans.len());

    if args.len() != 1 {
        return arity_error!(1, args.len(), apply_span.clone());
    }

    // This should always succeed because Remake is single
    // threaded.
    let mut dict_val = args[0].borrow_mut();
    {
        match dict_val.deref_mut() {
            &mut Value::Dict(ref mut d) => {
                d.clear();
                return ok(Value::Str("ok".to_string()));
            }
            _ => {} // FALLTHROUGH
        }
    }

    type_error!(dict_val, arg_spans[0].clone(), "dict")
}

fn remake_extend(
    args: &[Rc<RefCell<Value>>],
    apply_span: &Span,
    arg_spans: &[Span],
) -> Result<Rc<RefCell<Value>>, InternalError> {
    debug_assert!(args.len() == arg_spans.len());

    if args.len() != 2 {
        return arity_error!(2, args.len(), apply_span.clone());
    }

    // This should always succeed because Remake is single
    // threaded.
    let mut dict_val = args[0].borrow_mut();
    {
        match dict_val.deref_mut() {
            &mut Value::Dict(ref mut d) => {
                let ex_val = args[1].borrow();
                match ex_val.deref() {
                    &Value::Dict(ref ex) => {
                        for (k, v) in ex.iter() {
                            d.insert(k.clone(), Rc::clone(v));
                        }

                        return ok(Value::Str("ok".to_string()));
                    }
                    _ => {} // FALLTHROUGH
                }

                return type_error!(ex_val, arg_spans[1].clone(), "dict");
            }
            &mut Value::Vector(ref mut v) => {
                let ex_val = args[1].borrow();
                match ex_val.deref() {
                    &Value::Vector(ref ex) => {
                        for elem in ex.iter() {
                            v.push(Rc::clone(elem));
                        }

                        return ok(Value::Str("ok".to_string()));
                    }
                    _ => {} // FALLTHROUGH
                }

                return type_error!(ex_val, arg_spans[1].clone(), "vec");
            }
            _ => {} // FALLTHROUGH
        }
    }

    type_error!(dict_val, arg_spans[0].clone(), "dict", "vec")
}

fn remake_show(
    args: &[Rc<RefCell<Value>>],
    apply_span: &Span,
    arg_spans: &[Span],
) -> Result<Rc<RefCell<Value>>, InternalError> {
    debug_assert!(args.len() == arg_spans.len());

    if args.len() != 1 {
        return arity_error!(1, args.len(), apply_span.clone());
    }

    // This should always succeed because Remake is single
    // threaded.
    let val = args[0].borrow();
    ok(Value::Str(val.to_string()))
}

fn remake_push(
    args: &[Rc<RefCell<Value>>],
    apply_span: &Span,
    arg_spans: &[Span],
) -> Result<Rc<RefCell<Value>>, InternalError> {
    debug_assert!(args.len() == arg_spans.len());

    if args.len() != 2 {
        return arity_error!(2, args.len(), apply_span.clone());
    }

    // This should always succeed because Remake is single
    // threaded.
    let mut dict_val = args[0].borrow_mut();
    {
        match dict_val.deref_mut() {
            &mut Value::Vector(ref mut v) => {
                v.push(Rc::clone(&args[1]));
                return ok(Value::Str("ok".to_string()));
            }
            _ => {} // FALLTHROUGH
        }
    }

    type_error!(dict_val, arg_spans[0].clone(), "vec")
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

    eval_to!(prim_cmp_27_, " false < 1 ", Value::Bool(false));
    eval_to!(prim_cmp_28_, " false > 1 ", Value::Bool(false));
    eval_to!(prim_cmp_29_, " false <= 1 ", Value::Bool(false));
    eval_to!(prim_cmp_30_, " false >= 1 ", Value::Bool(false));
    eval_to!(prim_cmp_31_, " false == 1 ", Value::Bool(false));
    eval_to!(prim_cmp_32_, " false != 1 ", Value::Bool(true));

    eval_to!(prim_cmp_33_, " false || true ", Value::Bool(true));
    eval_to!(prim_cmp_34_, " true && false ", Value::Bool(false));

    eval_fail!(prim_cmp_35_, " /regex/ > /regex/ ", "TypeError");
    eval_fail!(prim_cmp_36_, " /regex/ < /regex/ ", "TypeError");
    eval_fail!(prim_cmp_37_, " /regex/ <= /regex/ ", "TypeError");
    eval_fail!(prim_cmp_48_, " /regex/ >= /regex/ ", "TypeError");

    eval_to!(prim_cmp_49_, " !true ", Value::Bool(false));

    eval_to!(prim_cmp_50_, r#" /re/ == 3 "#, Value::Bool(false));
    eval_to!(prim_cmp_51_, r#" "str" == 3 "#, Value::Bool(false));
    eval_to!(prim_cmp_52_, r#" {} == 3 "#, Value::Bool(false));
    eval_to!(prim_cmp_53_, r#" [] == 3 "#, Value::Bool(false));
    eval_to!(prim_cmp_54_, r#" (3,4) == 3 "#, Value::Bool(false));

    eval_to!(prim_cmp_55_, r#" {1:2} == {1:2} "#, Value::Bool(true));
    eval_to!(
        prim_cmp_56_,
        r#" (1,2,(3,4)) == (1,2,(3,4)) "#,
        Value::Bool(true)
    );
    eval_to!(
        prim_cmp_57_,
        r#" [1,2,[3,4]] == [1,2,[3,4]] "#,
        Value::Bool(true)
    );

    eval_to!(prim_cmp_58_, r#" "str" < 3 "#, Value::Bool(false));
    eval_to!(prim_cmp_59_, r#" 3.1 < 3 "#, Value::Bool(false));
    eval_to!(prim_cmp_60_, r#" 3 < 3.1 "#, Value::Bool(false));

    eval_to!(prim_cmp_61_, r#" "str" > 3 "#, Value::Bool(false));
    eval_to!(prim_cmp_62_, r#" 3.1 > 3 "#, Value::Bool(false));
    eval_to!(prim_cmp_63_, r#" 3 > 3.1 "#, Value::Bool(false));

    eval_to!(prim_cmp_64_, r#" "str" <= 3 "#, Value::Bool(false));
    eval_to!(prim_cmp_65_, r#" 3.1 <= 3 "#, Value::Bool(false));
    eval_to!(prim_cmp_66_, r#" 3 <= 3.1 "#, Value::Bool(false));

    eval_to!(prim_cmp_67_, r#" "str" >= 3 "#, Value::Bool(false));
    eval_to!(prim_cmp_68_, r#" 3.1 >= 3 "#, Value::Bool(false));
    eval_to!(prim_cmp_69_, r#" 3 >= 3.1 "#, Value::Bool(false));

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
    eval_fail!(arith_26_, " 19.0 </> 0.0 ", "ZeroDivisionError");

    eval_fail!(arith_27_, " 19 </> 0.0", "TypeError");
    eval_fail!(arith_28_, " 19.0 </> 0 ", "TypeError");

    eval_fail!(arith_29_, r#" - "hello" "#, "TypeError");
    eval_fail!(arith_30_, r#" ! "hello" "#, "TypeError");
    eval_fail!(arith_31_, r#" "hello"{1} "#, "TypeError");
    eval_fail!(arith_32_, r#" "hello"{1,} "#, "TypeError");
    eval_fail!(arith_33_, r#" "hello"{1,2} "#, "TypeError");
    eval_fail!(arith_34_, r#" "hello"? "#, "TypeError");
    eval_fail!(arith_35_, r#" "hello"+ "#, "TypeError");
    eval_fail!(arith_36_, r#" "hello"* "#, "TypeError");

    eval_fail!(arith_37_, r#" 5.8 % 2 "#, "TypeError");
    eval_fail!(arith_38_, r#" 5 <*> 2.0 "#, "TypeError");

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

    eval_to!(
        dict_16_,
        r#" "key" in { "key": "val" } "#,
        Value::Bool(true)
    );

    eval_to!(
        dict_17_,
        r#" "bad key" in { "key": "val" } "#,
        Value::Bool(false)
    );

    eval_to!(
        dict_18_,
        r#"
        let ks = keys({ "how": 1, "exciting": 2 });
        ("how" in ks) && ("exciting" in ks)
        "#,
        Value::Bool(true)
    );

    eval_to!(
        dict_19_,
        r#"
        let vs = values({ "how": 1, "exciting": 2 });
        (1 in vs) && (2 in vs)
        "#,
        Value::Bool(true)
    );

    eval_to!(
        dict_20_,
        r#"
        let is = items({ "how": 1, "exciting": 2 });
        (("how", 1) in is) && (("exciting", 2) in is)
        "#,
        Value::Bool(true)
    );

    eval_to!(
        dict_21_,
        r#"
        let d = { "how": 1 };
        extend(d, {"exciting": 2});
        d == { "how": 1, "exciting": 2 }
         "#,
        Value::Bool(true)
    );

    eval_fail!(dict_22_, r#" keys() "#, "ArityError");
    eval_fail!(dict_23_, r#" keys(1, 2) "#, "ArityError");

    eval_fail!(dict_24_, r#" items() "#, "ArityError");
    eval_fail!(dict_25_, r#" items(1, 2) "#, "ArityError");

    eval_fail!(dict_26_, r#" values() "#, "ArityError");
    eval_fail!(dict_27_, r#" values(1, 2) "#, "ArityError");

    eval_fail!(dict_28_, r#" extend(1) "#, "ArityError");
    eval_fail!(dict_29_, r#" extend(1, 3, 2) "#, "ArityError");

    eval_to!(
        dict_30_,
        r#"
        let d = { "how": 1 };
        clear(d);
        d == {}
         "#,
        Value::Bool(true)
    );
    eval_fail!(dict_31_, r#" clear() "#, "ArityError");
    eval_fail!(dict_32_, r#" clear(1, 2) "#, "ArityError");

    eval_to!(
        dict_33_,
        r#"
        let d = { "how": 1 };
        d[{
            clear(d);
            d[1] = "x";
            d["x"] = "y";
            d[1]
        }]
         "#,
        Value::Str("y".to_string())
    );

    eval_to!(
        dict_34_,
        r#"
        let d = { "how": 1 };
        show(d)
        "#,
        Value::Str(r#"{ "how": 1 }"#.to_string())
    );

    // TODO(ethan): tuples as keys

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

    eval_fail!(tuple_7_, r#" (1, 2)["bad key"] "#, "TypeError");

    eval_fail!(
        tuple_8_,
        r#"
    let x = (1, 2);
    x["bad key"] = 5;
    x[0]
    "#,
        "TypeError"
    );

    eval_to!(
        tuple_9_,
        r#" show((1, 2)) "#,
        Value::Str("(1, 2)".to_string())
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

    eval_fail!(
        vec_19_,
        r#"
    let x = [1, 2];
    x["bad key"] = 5;
    x[0]
    "#,
        "TypeError"
    );

    eval_to!(vec_20_, " [1, 2, 3][-2:][0] ", Value::Int(2));

    eval_fail!(vec_21_, r#" [1, 2]["bad key":"bad key"] "#, "TypeError");
    eval_fail!(vec_22_, r#" [1, 2]["bad key":] "#, "TypeError");
    eval_fail!(vec_23_, r#" [1, 2][:"bad key"] "#, "TypeError");
    eval_fail!(vec_24_, r#" [1, 2][1:"bad key"] "#, "TypeError");
    eval_fail!(vec_25_, r#" (3, 4)[1:2] "#, "TypeError");

    eval_fail!(vec_26_, r#" [1, 2]["bad key"] "#, "TypeError");

    eval_to!(
        vec_27_,
        r#" show([1, 2]) "#,
        Value::Str("[1, 2]".to_string())
    );

    eval_to!(
        vec_28_,
        r#"
    let v = [1, 2];
    push(v, "next");
    v == [1, 2, "next"]
    "#,
        Value::Bool(true)
    );

    eval_to!(
        vec_29_,
        r#"
    let v = [1, 2];
    extend(v, [3, 4]);
    v == [1, 2, 3, 4]
    "#,
        Value::Bool(true)
    );

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

    eval_fail!(if_9_, r#" if (1) { 9 } else { 10 } "#, "TypeError");
    eval_fail!(
        if_10_,
        r#"
        if (1) { let y = 1; }
        5
        "#,
        "TypeError"
    );

    /* TODO
    eval_to!(
        if_11_,
        r#"
        if (true) {
            let y = 1;
            y
        } else {
            17
        }
        "#,
        Value::Int(1)
    );
    */

    //
    // For loops
    //

    eval_fail!(for_1_, r#" continue; x "#, "LoopError");
    eval_fail!(for_2_, r#" break; x "#, "LoopError");

    eval_to!(
        for_3_,
        r#"
        let x = [1, 2, 3];
        let sum = 0;
        for (i in x) {
            sum = sum <+> i;
        }
        sum
        "#,
        Value::Int(6)
    );

    eval_to!(
        for_4_,
        r#"
        let x = [1, 2, 3];
        let sum = 0;
        for (i in x) {
            if (i == 2) {
                continue;
            }
            sum = sum <+> i;
        }
        sum
        "#,
        Value::Int(4)
    );

    eval_to!(
        for_5_,
        r#"
        let x = [1, 2, 3];
        let sum = 0;
        for (i in x) {
            if (i == 2) {
                break;
            }
            sum = sum <+> i;
        }
        sum
        "#,
        Value::Int(1)
    );

    eval_to!(
        for_6_,
        r#"
        let sum = 0;
        for (i in (1, 2, 3)) {
            sum = sum <+> i;
        }
        sum
        "#,
        Value::Int(6)
    );

    //
    // While Loops
    //

    eval_to!(
        while_1_,
        r#"
        let cnt = 1;
        let sum = 0;
        while (cnt < 4) {
            sum = sum <+> cnt;
            cnt = cnt <+> 1;
        }
        sum
        "#,
        Value::Int(6)
    );

    eval_to!(
        while_2_,
        r#"
        let cnt = 0;
        let sum = 0;
        while (cnt < 3) {
            cnt = cnt <+> 1;
            if (cnt == 2) {
                continue;
            }
            sum = sum <+> cnt;
        }
        sum
        "#,
        Value::Int(4)
    );

    eval_to!(
        while_3_,
        r#"
        let cnt = 1;
        let sum = 0;
        while (cnt < 4) {
            if (cnt == 2) {
                break;
            }
            sum = sum <+> cnt;
            cnt = cnt <+> 1;
        }
        sum
        "#,
        Value::Int(1)
    );

    eval_to!(
        while_4_,
        r#"
        let v = [1, 2, 3, "term"];
        let i = -1;
        while ({ i = i <+> 1; v[i] != "term" }) {}
        v[i]
        "#,
        Value::Str("term".to_string())
    );

    //
    // Loop Loops
    //

    eval_to!(
        loop_loop_1_,
        r#"
        let cnt = 1;
        let sum = 0;
        loop {
            if (!(cnt < 4)) {
                break;
            }
            sum = sum <+> cnt;
            cnt = cnt <+> 1;
        }
        sum
        "#,
        Value::Int(6)
    );

    eval_to!(
        loop_loop_2_,
        r#"
        let cnt = 0;
        let sum = 0;
        loop {
            if (!(cnt < 3)) {
                break;
            }
            cnt = cnt <+> 1;
            if (cnt == 2) {
                continue;
            }
            sum = sum <+> cnt;
        }
        sum
        "#,
        Value::Int(4)
    );

    eval_to!(
        loop_loop_3_,
        r#"
        let cnt = 1;
        let sum = 0;
        loop {
            if (cnt == 2) {
                break;
            }
            sum = sum <+> cnt;
            cnt = cnt <+> 1;
        }
        sum
        "#,
        Value::Int(1)
    );

    eval_to!(
        loop_loop_4_,
        r#"
        let v = [1, 2, 3, "term"];
        let i = -1;
        loop { i = i <+> 1; if (v[i] == "term") { break; } }
        v[i]
        "#,
        Value::Str("term".to_string())
    );

    //
    // Statement blocks
    //

    /* TODO
    eval_to!(
        stmt_block_1_,
        r#"
        let x = 1;
        {
            x = 2;
            let y = 5;
        }
        x
        "#,
        Value::Int(2)
    );

    eval_fail!(
        stmt_block_2_,
        r#"
        {
            let y = 5;
        }
        y
        "#,
        "NameError"
    );
    */

    //
    // Functions
    //

    eval_to!(
        fn_1_,
        r#"
        fn f(x) { x <+> 1 }
        f(0)
        "#,
        Value::Int(1)
    );

    eval_to!(
        fn_2_,
        r#"
        (fn(x) { x <+> 1 })(0)
        "#,
        Value::Int(1)
    );

    eval_to!(
        fn_3_,
        r#"
        fn f(x) { x <+> 1 }
        fn g(y) { f(y) <+> 1 }
        g(0)
        "#,
        Value::Int(2)
    );

    eval_to!(
        fn_4_,
        r#"
        let x = 0;
        fn f() {
            x = x <+> 1;
            x
        }
        f();
        f();
        f();
        x
        "#,
        Value::Int(3)
    );

    eval_fail!(fn_5_, "fn() { x }", "NameError");
    eval_fail!(fn_6_, "15(\"hi\")", "TypeError");

    eval_fail!(
        fn_7_,
        r#"
    fn f(x, y) {
        x <+> y
    }
    f(1)
    "#,
        "ArityError"
    );

    eval_to!(
        fn_recur_1_,
        r#"
    fn fib(x) {
        if ( (x == 1) || (x == 2) ) {
            1
        } else {
            fib(x - 1) <+> fib(x - 2)
        }
    }
    fib(5)
    "#,
        Value::Int(5)
    );

    //
    // Rc Borrow Tests (also see dict_33_)
    //

    eval_to!(
        rc_borrow_1_,
        r#"
        let d = [1];
        d[{
            push(d, "y");
            d[0]
        }]
         "#,
        Value::Str("y".to_string())
    );

    eval_to!(
        rc_borrow_2_,
        r#"
        let d = (1, "y");
        d[ d[0] ]
        "#,
        Value::Str("y".to_string())
    );

    /* TODO
    eval_fail!(
        rc_borrow_3_,
        r#"
        let d = 1;
        d <+> { d[0] = 2; d[0] }
        "#,
        "TypeError"
    );
    */

    //
    // Misc
    //

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

    // TODO
    // eval_to!(index_prec_1_, r#"let x = (1, 2); x == x[1]"#, Value::Bool(false));

    eval_fail!(cap_1_, r#" cap 3.5 as foo "#, "TypeError");

    eval_fail!(bad_index_1_, r#" 3[9] "#, "TypeError");

    eval_to!(
        vec_cmp_1_,
        r#"
    let v = [1, "term"];
    let i = 1;
    v[i] == "term"
    "#,
        Value::Bool(true)
    );
    eval_fail!(vec_cmp_2_, r#" v[i] == "term" "#, "NameError");

    eval_to!(
        break_cleanup_1_,
        r#"
    let x = 1;
    loop {
        let _ = { let x = 2; break; x };
    }
    x
    "#,
        Value::Int(1)
    );

}
