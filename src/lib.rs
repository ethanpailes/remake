extern crate regex_syntax;
extern crate regex;
extern crate lalrpop_util;

mod ast;
mod parse;

//
// I want the eventual interface to look something like:
//
// ```
// let r = Remake::new(r"/foo/").unwrap();
// let wrap_parens = Remake::new(r"(re) => '(' + re ')'").unwrap();
// let re: Regex = wrap_parens.apply(r).unwrap().eval().unwrap();
// ```
//
// The idea is that remake expressions can be parsed and then passed
// around within rust as opaque expressions. They can then be combined
// through function application.
//

use regex::Regex;

pub struct Remake {
    #[doc(hidden)]
    expr: ast::Expr,
}

impl Remake {
    pub fn new(src: &str) -> Result<Self, Error> {
        match parse::ExprParser::new().parse(src) {
            Ok(expr) => Ok(Remake { expr: *expr }),
            Err(err) => Err(Error::ParseError(format!("{}", err))),
        }
    }

    pub fn eval(&self) -> Result<Regex, Error> {
        match self.expr {
            ast::Expr::RegexLiteral(ref regex_ast) =>
                Ok(Regex::new(&format!("{}", regex_ast)).unwrap()),
        }
    }

    pub fn eval_str(src: &str) -> Result<Regex, Error> {
        Self::new(src)?.eval()
    }
}

// TODO(ethan): use failure?
#[derive(Debug)]
pub enum Error {
    /// A parse error is just parameterized by a string because
    /// zero code is going to be both smart enough to correct the
    /// issue and dumb that it can't parse the human readable error.
    ParseError(String),
}

// TODO(ethan): impl Display for Error
// TODO(ethan): Write all the basic combinators so that we can get an
//              expression language going here.
//              - It is probably worth making a POISON_SPAN the result
//                of regex ASTs that are built up with remake concatination
//                or alternation or whatever.
// TODO(ethan): Write let expressions.
// TODO(ethan): Cleanup and going over error messages to make sure that they
//              are useful.
// TODO(ethan): Docs.
// TODO(ethan): Release!!! (don't talk about it until we have lambdas though).

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! mat {
        ($test_name:ident, $remake_src:expr, $input:expr) => {
            #[test]
            fn $test_name() {
                let re = Remake::eval_str($remake_src).unwrap();
                assert!(re.is_match($input),
                    format!("/{:?}/ does not match {:?}.", re, $input));
            }
        }
    }

    macro_rules! noparse {
        ($test_name:ident, $remake_src:expr) => {
            #[test]
            fn $test_name() {
                assert!(!Remake::eval_str($remake_src).ok().is_some(),
                        format!("{:?} parses.", $remake_src));
            }
        }
    }

    mat!(lit_1, r"/a/", "a");
    mat!(lit_2, r"'a'", "a");
    mat!(lit_3, r"/\p{Currency_Symbol}/", r"$");
    mat!(lit_4, r"'\p{Currency_Symbol}'", r"\p{Currency_Symbol}");
    mat!(lit_5, r"'\u{Currency_Symbol}'", r"\u{Currency_Symbol}");

    noparse!(noparse_1, "a");
    noparse!(noparse_2, "'a");
    noparse!(noparse_3, "a/");
    noparse!(noparse_4, "'a/");
}
