extern crate regex_syntax;
extern crate regex;
extern crate lalrpop_util;

mod ast;
mod parse;
mod error;

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

use std::fmt;
use regex::Regex;

pub struct Remake {
    /// The parsed remake expression.
    expr: ast::Expr,
    /// The source used to construct this remake expression.
    ///
    /// Required to interpret spans.
    src: String,
}

impl Remake {
    /// Construct a Remake expression which can be evaluated
    /// at a later time.
    pub fn new(src: String) -> Result<Self, Error> {
        let mut remake = Remake {
            expr: ast::Expr::new(ast::ExprKind::ExprPoison,
                                 ast::Span { start: 0, end: 0 }),
            src: src,
        };

        remake.expr = match parse::ExprParser::new().parse(&remake.src) {
            Ok(expr) => *expr,
            Err(err) => {
                return Err(Error::ParseError(match err {
                    lalrpop_util::ParseError::User { error } =>
                        format!("{}", error.overlay(&remake.src)),
                    err => format!("{}", err),
                }));
            }
        };

        Ok(remake)
    }

    /// Evaluate a Remake expression.
    pub fn eval(&self) -> Result<Regex, Error> {
        match self.expr.kind() {
            &ast::ExprKind::RegexLiteral(ref regex_ast) =>
                Ok(Regex::new(&format!("{}", regex_ast)).unwrap()),

            &ast::ExprKind::ExprPoison =>
                panic!("Bug in remake."),
        }
    }

    /// Evaluate some Remake source to produce a regular expression.
    pub fn compile(src: &str) -> Result<Regex, Error> {
        Self::new(String::from(src))?.eval()
    }
}

// TODO(ethan): use failure?
// TODO(ethan): impl std::error::Error
#[derive(Clone)]
pub enum Error {
    /// A parse error is just parameterized by a string because
    /// zero code is going to be both smart enough to correct the
    /// issue and dumb that it can't parse the human readable error.
    ParseError(String),

    #[doc(hidden)]
    __NonExaustive,
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Error::*;

        match self {
            &ParseError(ref err) => {
                writeln!(f, "\nremake parse error:")?;
                writeln!(f, "{}", err)?;
            }

            &__NonExaustive => unreachable!(),
        }

        Ok(())
    }
}

// TODO(ethan): Write all the basic combinators so that we can get an
//              expression language going here.
//              - It is probably worth making a POISON_SPAN the result
//                of regex ASTs that are built up with remake concatination
//                or alternation or whatever.
// TODO(ethan): Write let expressions.
// TODO(ethan): Cleanup and going over error messages to make sure that they
//              are useful.
// TODO(ethan): Docs.
// TODO(ethan): Run rustfmt
// TODO(ethan): Release!!! (don't talk about it until we have lambdas though).

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! mat {
        ($test_name:ident, $remake_src:expr, $input:expr) => {
            #[test]
            fn $test_name() {
                let re = Remake::compile($remake_src).unwrap();
                assert!(re.is_match($input),
                    format!("/{:?}/ does not match {:?}.", re, $input));
            }
        }
    }

    macro_rules! parse_error {
        ($test_name:ident, $remake_src:expr, $expected_err_str:expr) => {
            #[test]
            fn $test_name() {
                let result = Remake::compile($remake_src);
                match &result {
                    &Ok(_) => panic!("Should not eval to anything."),
                    &Err(Error::ParseError(ref reason)) => {
                        // When copying the right output into the test
                        // case uncommenting this can help debug.
                        //
                        // assert_eq!($expected_err_str, reason);
                        
                        if $expected_err_str != reason {
                            // We unwrap rather than asserting or something
                            // a little more reasonable so that we can see
                            // the error output as the user sees it.
                            result.clone().unwrap();
                        }
                    }
                    _ => panic!("Should not parse."),
                }
            }
        }
    }

    mat!(lit_1, r"/a/", "a");
    mat!(lit_2, r"'a'", "a");
    mat!(lit_3, r"/\p{Currency_Symbol}/", r"$");
    mat!(lit_4, r"'\p{Currency_Symbol}'", r"\p{Currency_Symbol}");
    mat!(lit_5, r"'\u{Currency_Symbol}'", r"\u{Currency_Symbol}");

    // remake parse errors
    parse_error!(unmatched_tick_1_, r"'a", "FIXME");
    parse_error!(unmatched_tick_2_, r"a'", "FIXME");
    parse_error!(unmatched_slash_1_, r"/a", "FIXME");
    parse_error!(unmatched_slash_2_, r"a/", "FIXME");
    parse_error!(unmatched_tick_slash_1_, r"'a/", "FIXME");
    parse_error!(unmatched_tick_slash_2_, r"/a'", "FIXME");

    //
    // parse errors that bubble up from the regex crate
    //
    
    parse_error!(re_parse_err_1_, r"/a[/", 
        r#"    at line 1, col 1:
    > /a[/
      ^^^^
Error parsing the regex literal: /a[/
    regex parse error:
        a[
         ^
    error: unclosed character class
"#);

    parse_error!(re_parse_err_2_, r"/a[]/",
        r#"    at line 1, col 1:
    > /a[]/
      ^^^^^
Error parsing the regex literal: /a[]/
    regex parse error:
        a[]
         ^^
    error: unclosed character class
"#);

    parse_error!(re_multiline_parse_err_1_,
        r#"

            /a[/

        "#,
        r#"    at line 3, col 13:
    > 
    >             /a[/
                  ^^^^
    > 
Error parsing the regex literal: /a[/
    regex parse error:
        a[
         ^
    error: unclosed character class
"#);
}
