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
use std::ops::{Deref};
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
    Tuple(Vec<Rc<RefCell<Value>>>),
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
            &Value::Tuple(_) => "tuple",
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
        {
            let val = $val.borrow();
            let ty = val.type_of();
            if $((ty != $expected) &&)* true {
                return type_error!(val, $span, $($expected),*);
            }
        }
    }
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

/// Evaluate an expression in a given evaluation context
///
/// The public facing version is `eval`, but this guy is the
/// workhorse.
///
/// A key invariant in the `eval_` function is that we never keep
/// a dynamic borrow of a Remake value (`Rc.borrow`) across a
/// recursive call to the evaluator. The reason is that evaluation
/// might require taking a mutable or immutable borrow of an
/// arbitrary value, which can cause a dynamic borrow panic.
/// This is usually considered A Bad Thing.
fn eval_(
    env: &mut EvalEnv,
    expr: &Expr,
) -> Result<Rc<RefCell<Value>>, InternalError> {
    match expr.kind {
        ExprKind::RegexLiteral(ref r) => ok(Value::Regex(r.clone())),
        ExprKind::IntLiteral(ref i) => ok(Value::Int(i.clone())),
        ExprKind::FloatLiteral(ref f) => ok(Value::Float(f.clone())),
        ExprKind::StringLiteral(ref s) => ok(Value::Str(s.clone())),
        ExprKind::TupleLiteral(ref es) => {
            debug_assert!(es.len() != 0);
            let mut vs = Vec::with_capacity(es.len());

            for v_expr in es.iter() {
                vs.push(eval_(env, &v_expr)?);
            }

            ok(Value::Tuple(vs))
        }

        ExprKind::BinOp(ref l_expr, ref op, ref r_expr) => match op {
            &ast::BOp::Concat => {
                let l_val = eval_(env, l_expr)?;
                type_guard!(l_val, l_expr.span.clone(), "regex", "str");
                let r_val = eval_(env, r_expr)?;

                let l_val = l_val.borrow();
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
                type_guard!(l_val, l_expr.span.clone(), "regex");
                let r_val = eval_(env, r_expr)?;

                let r_val = r_val.borrow();
                let l_val = l_val.borrow();

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

            &ast::BOp::Plus => {
                let l_val = eval_(env, l_expr)?;
                type_guard!(l_val, l_expr.span.clone(), "int", "float");
                let r_val = eval_(env, r_expr)?;

                let r_val = r_val.borrow();
                let l_val = l_val.borrow();

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
                type_guard!(l_val, l_expr.span.clone(), "int", "float");
                let r_val = eval_(env, r_expr)?;

                let r_val = r_val.borrow();
                let l_val = l_val.borrow();

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
                type_guard!(l_val, l_expr.span.clone(), "int", "float");
                let r_val = eval_(env, r_expr)?;

                let r_val = r_val.borrow();
                let l_val = l_val.borrow();

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
                type_guard!(l_val, l_expr.span.clone(), "int", "float");
                let r_val = eval_(env, r_expr)?;

                let r_val = r_val.borrow();
                let l_val = l_val.borrow();

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
                type_guard!(l_val, l_expr.span.clone(), "int", "float");
                let r_val = eval_(env, r_expr)?;

                let l_val = l_val.borrow();
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
                    // unwrap the smart pointer to the result of evaluation so
                    // that we can move it into the cell that the variable
                    // points to.
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
                _ => unreachable!("Bug in remake - assign to non-lvalue"),
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
                &Value::Tuple(ref v) => {
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

const BUILTINS: [BuiltIn; 1] = [
    BuiltIn {
        name: "show",
        func: remake_show,
    },
];

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

/// We don't have to spend any effort assigning indicies to groups because
/// we are going to pretty-print the AST and have regex just parse it.
/// If we passed the AST to the regex crate directly, we would need some
/// way to thread the group index through its parser. This way we can
/// just ignore the whole problem.
const BOGUS_GROUP_INDEX: u32 = 0;

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

    eval_to!(
        str_1_,
        " \"hello \" . \"world\"",
        Value::Str("hello world".to_string())
    );

    eval_fail!(str_2_, " \"hello \" . 'sup' ", "TypeError");
    eval_fail!(str_3_, " 'regex' . \"str\" ", "TypeError");

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
    // tuples
    //

    eval_to!(
        tuple_9_,
        r#" show((1, 2)) "#,
        Value::Str("(1, 2)".to_string())
    );

    //
    // For loops
    //

    eval_fail!(for_1_, r#" continue; x "#, "LoopError");
    eval_fail!(for_2_, r#" break; x "#, "LoopError");

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

    eval_fail!(cap_1_, r#" cap 3.5 as foo "#, "TypeError");
}
