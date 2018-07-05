use std::rc::Rc;

use regex_syntax::ast::{Position, Span};
use ast;
use std::collections::HashSet;

pub const POISON_SPAN: Span = Span {
    start: Position {
        offset: 0,
        line: 1,
        column: 1,
    },
    end: Position {
        offset: 0,
        line: 1,
        column: 1,
    },
};

/// Compute the variables free in an expression given an initial set
/// of bindings.
pub fn free_vars(expr: &ast::Expr) -> HashSet<&str> {
    ast::visit_expr(FreeVarVisitor {
        free_set: HashSet::new(),
        block_bindings: vec![],
    }, expr).unwrap()
}
struct FreeVarVisitor<'expr> {
    /// The free variables we have found so far. Some of them
    /// might actually be the result of local bindings and get
    /// removed later.
    free_set: HashSet<&'expr str>,
    /// The bindings introduced in the current block
    block_bindings: Vec<&'expr str>,
}
impl<'expr> ast::Visitor<'expr> for FreeVarVisitor<'expr> {
    type Output = HashSet<&'expr str>;
    type Err = ();

    fn finish(self) -> Result<HashSet<&'expr str>, ()> {
        Ok(self.free_set)
    }

    fn visit_expr_post(&mut self, expr: &'expr ast::Expr) -> Result<(), ()> {
        match &expr.kind {
            // Any variable gets added to the set of free variables
            &ast::ExprKind::Var(ref v) => {
                self.free_set.insert(v);
            }
            // Function definitions also introduce bindings.
            //
            // TODO: technically we could probably avoid a little bit
            // of (parse time) work here by taking advantage of the
            // cached free variables. This would just mean adding the
            // ability to control visitor behavior with a return code.
            &ast::ExprKind::Lambda {
                ref expr,
                ref free_vars,
            } => {
                for a in expr.args.iter() {
                    self.free_set.remove(a.as_str());
                }
                debug_assert!(
                    free_vars.iter().all(|fv| self.free_set.contains(fv.as_str())),
                    "Cached free variable list disagrees with computed list."
                );
            }

            &ast::ExprKind::Block(_, _) => {
                let mut bindings = vec![];
                ::std::mem::swap(&mut self.block_bindings, &mut bindings);

                for b in bindings {
                    self.free_set.remove(b);
                }
            }

            _ => {} // FALLTHROUGH
        }

        Ok(())
    }

    fn visit_stmt_post(&mut self, stmt: &'expr ast::Statement) -> Result<(), ()> {
        match &stmt.kind {
            // When we see a let binding, we remove that variable from
            // the set.
            &ast::StatementKind::LetBinding(ref v, _) => {
                self.block_bindings.push(v.as_str());
            }
            &ast::StatementKind::Block(_) => {
                let mut bindings = vec![];
                ::std::mem::swap(&mut self.block_bindings, &mut bindings);

                for b in bindings {
                    self.free_set.remove(b);
                }
            }
            _ => {} // FALLTHROUGH
        }

        Ok(())
    }
}

pub fn construct_lambda(args: Vec<&str>, body: ast::Expr) -> ast::ExprKind {
    let free_vs;
    {
        let mut fvs = free_vars(&body);
        for a in args.iter() {
            fvs.remove(a);
        }
        free_vs = fvs
            .into_iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>();
    }

    ast::ExprKind::Lambda {
        expr: Rc::new(ast::Lambda {
            args: args.iter().map(|a| a.to_string()).collect::<Vec<_>>(),
            body: Box::new(body),
        }),
        free_vars: free_vs,
    }
}

#[cfg(test)]
mod tests {
    use parse::BlockBodyParser;
    use lex;
    use super::free_vars;

    macro_rules! free_vars_are {
        ($func_name:ident, $remake_src:expr, $($fvs:expr),*) => {
            #[test]
            fn $func_name() {
                let parser = BlockBodyParser::new();
                let lexer = lex::Lexer::new($remake_src);
                let expr = parser.parse(lexer).expect("The expr to parse");
                let fvs = free_vars(&expr);
                let expected = vec![$($fvs),*];
                debug_assert!(expected.len() == fvs.len());
                debug_assert!(expected.iter().all(|v| fvs.contains(v)));
            }
        }
    }

    free_vars_are!(fv_1_, "x", "x");

    free_vars_are!(fv_2_, "x - y", "x", "y");

    free_vars_are!(fv_3_, r#"
    fn(x) {
        x - y
    }
    "#, "y");

    free_vars_are!(fv_expr_block_1_, r#"
    let x = 1;
    x <+> y
    "#, "y");

    /* TODO
    free_vars_are!(fv_stmt_blk_1_, r#"
    {
        let x = 1;
        x <+> y
    }
    1
    "#, "y");
    */
}
