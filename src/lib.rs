extern crate regex_syntax;
extern crate regex;
extern crate lalrpop_util;

mod ast;
mod parse;
mod error;
mod operators;
mod util;

//
// I want the eventual interface to look something like:
//
// ```
// let r = Remake::new(r"/foo/")?;
// let wrap_parens = Remake::new(r"(re) => '(' + re ')'")?;
// let re: Regex = wrap_parens.apply(r)?.eval()?;
// ```
//
// The idea is that remake expressions can be parsed and then passed
// around within rust as opaque expressions. They can then be combined
// through function application.
//

use std::fmt;
use regex::Regex;
use error::InternalError;

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

                    lalrpop_util::ParseError::InvalidToken { location } =>
                        format!("{}",
                            InternalError::new(
                                error::ErrorKind::InvalidToken,
                                ast::Span { start: location, end: location + 1 }
                                ).overlay(&remake.src)),

                    lalrpop_util::ParseError::UnrecognizedToken {
                        token: Some((l, tok, r)), expected
                    } => format!(
                        "{}",
                        InternalError::new(
                            error::ErrorKind::UnrecognizedToken {
                                token: tok.to_string(),
                                expected: expected,
                            },
                            ast::Span { start: l, end: r }
                            ).overlay(&remake.src)),

                    err => format!("{}", err),
                }));
            }
        };

        Ok(remake)
    }

    /// Evaluate a Remake expression.
    pub fn eval(self) -> Result<Regex, Error> {
        match self.expr.eval() {
            Ok(ast) => Ok(Regex::new(&ast.to_string()).unwrap()),
            Err(err) => Err(Error::RuntimeError(
                            format!("{}", err.overlay(&self.src)))),
        }
        // Ok(Regex::new(&format!("{}", self.expr.eval()?)).unwrap())
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
    /// A parse error occurred.
    ///
    /// A parse error is just parameterized by a string because
    /// zero code is going to be both smart enough to correct the
    /// issue and dumb that it can't parse the human readable error.
    ParseError(String),

    /// A runtime error occurred.
    RuntimeError(String),
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Error::*;

        match self {
            &ParseError(ref err) => {
                writeln!(f, "\nremake parse error:")?;
                writeln!(f, "{}", err)?;
            }

            &RuntimeError(ref err) => {
                writeln!(f, "\nremake evaluation error:")?;
                writeln!(f, "{}", err)?;
            }
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

    macro_rules! captures {
        ($test_name:ident, $remake_src:expr, $input:expr, $( $group:expr ),*) => {
            #[test]
            fn $test_name() {
                let re = Remake::compile($remake_src)
                            .expect("The regex to compile.");
                let expected = vec![$($group),*];
                let actual = re.captures($input)
                               .expect("The regex to match.")
                               .iter()
                               .map(|mat| mat.map(|g| g.as_str()))
                               .collect::<Vec<_>>();

                assert_eq!(&expected[..], &actual[1..],
                    "Captures did not match. re={:?}", re);
            }
        }
    }

    macro_rules! captures_named {
        ($test_name:ident, $remake_src:expr, $input:expr, $( $group:expr ),*) => {
            #[test]
            fn $test_name() {
                use std::str::FromStr;

                let re = Remake::compile($remake_src)
                            .expect("The regex to compile.");
                let expected_caps = vec![$($group),*];
                let num_name_re =
                    Remake::compile("'_' . cap /[0-9]+/").unwrap();

                let caps = re.captures($input).expect("The regex to match.");

                for &(name, result) in expected_caps.iter() {
                    let cap = match num_name_re.captures(name) {
                        Some(c) =>
                            caps.get(usize::from_str(&c[1]).unwrap())
                                .map(|m| m.as_str()),
                        None => caps.name(name).map(|m| m.as_str()),
                    };

                    assert_eq!(cap, result, "Group '{}' did not match", name);
                }
            }
        }
    }

    macro_rules! no_mat {
        ($test_name:ident, $remake_src:expr, $input:expr) => {
            #[test]
            fn $test_name() {
                let re = Remake::compile($remake_src).unwrap();
                assert!(!re.is_match($input),
                    format!("/{:?}/ matches {:?}.", re, $input));
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

    macro_rules! parse_error_pre {
        ($test_name:ident, $remake_src:expr, $expected_err_str:expr) => {
            #[test]
            fn $test_name() {
                let result = Remake::compile($remake_src);
                match &result {
                    &Ok(_) => panic!("Should not eval to anything."),
                    &Err(Error::ParseError(ref reason)) => {
                        let reason = &reason[0..$expected_err_str.len()];
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

    //
    // remake literals
    //

    mat!(lit_1, r"/a/", "a");
    mat!(lit_2, r"'a'", "a");
    mat!(lit_3, r"/\p{Currency_Symbol}/", r"$");
    mat!(lit_4, r"'\p{Currency_Symbol}'", r"\p{Currency_Symbol}");
    mat!(lit_5, r"'\u{Currency_Symbol}'", r"\u{Currency_Symbol}");

    //
    // remake parse errors
    //
    
    parse_error_pre!(unmatched_tick_1_, r"'a",
        r#"    at line 1, col 1:
    0001 > 'a
           ^
Invalid token."#);

    parse_error_pre!(unmatched_tick_2_, r"a'",
        r#"    at line 1, col 1:
    0001 > a'
           ^
Unexpected token 'a'."#);

    parse_error!(unmatched_slash_1_, r"/a",
        r#"    at line 1, col 1:
    0001 > /a
           ^
Invalid token.
"#);

    parse_error_pre!(unmatched_slash_2_, r"a/",
        r#"    at line 1, col 1:
    0001 > a/
           ^
Unexpected token 'a'."#);

    parse_error!(unmatched_tick_slash_1_, r"'a/",
        r#"    at line 1, col 1:
    0001 > 'a/
           ^
Invalid token.
"#);

    parse_error!(unmatched_tick_slash_2_, r"/a'",
        r#"    at line 1, col 1:
    0001 > /a'
           ^
Invalid token.
"#);

    //
    // parse errors that bubble up from the regex crate
    //
    
    parse_error!(re_parse_err_1_, r"/a[/", 
        r#"    at line 1, col 1:
    0001 > /a[/
           ^^^^
Error parsing the regex literal: /a[/
    regex parse error:
        a[
         ^
    error: unclosed character class
"#);

    parse_error!(re_parse_err_2_, r"/a[]/",
        r#"    at line 1, col 1:
    0001 > /a[]/
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
    0002 > 
    0003 >             /a[/
                       ^^^^
    0004 > 
Error parsing the regex literal: /a[/
    regex parse error:
        a[
         ^
    error: unclosed character class
"#);

    parse_error_pre!(unrecognized_token_1_, r"/foo/ /foo/",
        r#"    at line 1, col 7:
    0001 > /foo/ /foo/
                 ^^^^^
Unexpected token '/foo/'. Expected one of:"#);

    parse_error_pre!(binary_plus_1_, r"/foo/ + /bar/",
        r#"    at line 1, col 9:
    0001 > /foo/ + /bar/
                   ^^^^^
Unexpected token '/bar/'. Expected one of:"#);

    parse_error_pre!(binary_plus_multiline_1_, r#"/foo/ +
        /bar
        /"#, r#"    starting at line 2, col 9 and ending at line 3, col 10:
    0001  > /foo/ +
    0002  >         /bar
    start >         ^
    0003  >         /
    end   >         ^
Unexpected token '/bar
        /'. Expected one of:"#);

    parse_error_pre!(binary_plus_multiline_2_, r#"
        /foo/ + /
        
        bar



        /"#, r#"    starting at line 2, col 17 and ending at line 8, col 10:
    0001  > 
    0002  >         /foo/ + /
    start >                 ^
    0003  >         
    ...
    0007  > 
    0008  >         /
    end   >         ^
Unexpected token"#);

    //
    // Remake expressions.
    //

    mat!(concat_1_, r"/foo/ . 'bar'", "foobar");
    mat!(concat_2_, r"/foo/ . 'b[r'", "foob[r");
    mat!(alt_1_, r"/foo/ | /bar/", "foo");

    mat!(greedy_star_1_, r"/a/ *", "aaaaaaa");
    mat!(greedy_star_2_, r"/a/ * . 'b'", "aaaaaaab");
    mat!(greedy_star_3_, r"/a/ * . 'b'", "b");

    mat!(lazy_star_1_, r"/a/ *?", "aaaaaaa");
    mat!(lazy_star_2_, r"/a/ *? . 'b'", "aaaaaaab");
    mat!(lazy_star_3_, r"/a/ *? . 'b'", "b");

    mat!(greedy_plus_1_, r"/a/ +", "aaaaaaa");
    mat!(greedy_plus_2_, r"/a/ + . 'b'", "aaaaaaab");
    no_mat!(greedy_plus_3_, r"/a/ + . 'b'", "b");

    mat!(lazy_plus_1_, r"/a/ +?", "aaaaaaa");
    mat!(lazy_plus_2_, r"/a/ +? . 'b'", "aaaaaaab");
    no_mat!(lazy_plus_3_, r"/a/ +? . 'b'", "b");

    mat!(greedy_question_1_, r"/a/ ?", "a");
    mat!(greedy_question_2_, r"/a/ ? . 'b'", "ab");
    mat!(greedy_question_3_, r"/a/ ? . 'b'", "b");

    mat!(lazy_question_1_, r"/a/ ??", "a");
    mat!(lazy_question_2_, r"/a/ ?? . 'b'", "ab");
    mat!(lazy_question_3_, r"/a/ ?? . 'b'", "b");

    mat!(greedy_exact_1_, r"/a/ {3}", "aaa");
    mat!(greedy_exact_2_, r"/a/ { 3} . 'b'", "aaab");

    // lazyness does not matter for an exact repetition
    no_mat!(lazy_exact_1_, r"/a/ {3}?", "a");
    no_mat!(lazy_exact_2_, r"/a/ {3 }? . 'b'", "ab");
    mat!(lazy_exact_3_, r"/a/ {3}?", "aaa");
    mat!(lazy_exact_4_, r"/a/ {3 }? . 'b'", "aaab");


    mat!(greedy_atleast_1_, r"/a/ {3,}", "aaaaaaaa");
    mat!(greedy_atleast_2_, r"/a/ { 3 , } . 'b'", "aaaaaab");
    no_mat!(greedy_atleast_3_, r"/a/ { 3 , } . 'b'", "ab");

    mat!(lazy_atleast_1_, r"/a/ {3,}?", "aaaaaaaa");
    mat!(lazy_atleast_2_, r"/a/ { 3 , }? . 'b'", "aaaaaab");
    no_mat!(lazy_atleast_3_, r"/a/ { 3 , }? . 'b'", "ab");

    captures!(cap_basic_1_, r"/a(b)a/", "aba",
        Some("b"));

    captures!(cap_remake_1_, r"/a/ . cap /b/ . /a/", "aba",
        Some("b"));
    captures!(cap_remake_2_, r"cap (/a/ . cap /b/ . /a/)", "aba",
        Some("aba"), Some("b"));

    captures!(cap_remake_mixed_1_, r"cap (/a/ . /(b)/ . /a/)", "aba",
        Some("aba"), Some("b"));
    captures!(cap_remake_mixed_2_, r"/(a)/ . cap (/a/ . /(b)/ . /a/)", "aaba",
        Some("a"), Some("aba"), Some("b"));
    captures!(cap_remake_mixed_3_, r"cap /blah/ . (/(a)/ | (/b/ . cap /c/))", 
        "blahbc",
        Some("blah"), None, Some("c"));
    captures!(cap_remake_mixed_4_, r"/(a)(b)(c)/ . cap /blah/", "abcblah",
        Some("a"), Some("b"), Some("c"), Some("blah"));

    captures_named!(cap_remake_named_1_, r"/a/ . cap /b/ as foo . /a/", "aba",
        ("foo", Some("b")));
    captures_named!(cap_remake_named_2_, r"cap cap /foo/ as blah", "foo",
                    ("_1", Some("foo")), ("blah", Some("foo")));
}
