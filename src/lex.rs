// Copyright 2018 Ethan Pailes.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::convert::From;
use std::fmt;
use std::num::{ParseFloatError, ParseIntError};
use std::str::{CharIndices, FromStr};

use regex::Regex;

use ast::Span;
use error;
use error::InternalError;

pub type Spanned<Tok, Loc, Error> = Result<(Loc, Tok, Loc), Error>;

#[derive(Debug, Clone)]
pub enum Token<'input> {
    RegexLit(String),
    RawRegexLit(&'input str),
    IntLit(i64),
    FloatLit(f64),
    StringLit(String),
    Id(&'input str),

    // Operators
    OpenParen,
    CloseParen,
    OpenBracket,
    CloseBracket,
    OpenCurly,
    LazyCloseCurly,
    CloseCurly,
    LazyStar,
    Star,
    LazyPlus,
    Plus,
    Comma,
    Pipe,
    Dot,
    Equals,
    Question,
    LazyQuestion,
    Semi,
    Colon,

    And,
    Or,
    DoubleEq,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
    Bang,

    Minus,
    Add,
    Times,
    Div,
    Percent,

    // Keywords.
    As,
    Cap,
    Let,
    True,
    False,
    If,
    Else,
    In,
    For,
    Continue,
    Break,
    While,
    Loop,
}

impl<'input> fmt::Display for Token<'input> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &Token::RegexLit(ref re_src) => write!(f, "/{}/", re_src),
            &Token::RawRegexLit(ref re_src) => write!(f, "'{}'", re_src),
            &Token::IntLit(ref num) => write!(f, "{}", num),
            &Token::FloatLit(ref num) => {
                // HACK: I couldn't figure out how to ask rust for
                // a floating point format with at least one decimal place.
                let mut s = format!("{}", num);
                if !s.contains('.') {
                    s.push_str(".0");
                }
                write!(f, "{}", s)
            }
            &Token::StringLit(ref s) => write!(f, "\"{}\"", s),

            &Token::Id(ref id) => write!(f, "identifier: {}", id),

            // Operators
            &Token::OpenParen => write!(f, "("),
            &Token::CloseParen => write!(f, ")"),
            &Token::OpenBracket => write!(f, "["),
            &Token::CloseBracket => write!(f, "]"),
            &Token::OpenCurly => write!(f, "{{"),
            &Token::LazyCloseCurly => write!(f, "}}?"),
            &Token::CloseCurly => write!(f, "}}"),
            &Token::LazyStar => write!(f, "*?"),
            &Token::Star => write!(f, "*"),
            &Token::LazyPlus => write!(f, "+?"),
            &Token::Plus => write!(f, "+"),
            &Token::Comma => write!(f, ","),
            &Token::Pipe => write!(f, "|"),
            &Token::Dot => write!(f, "."),
            &Token::Equals => write!(f, "="),
            &Token::Question => write!(f, "?"),
            &Token::LazyQuestion => write!(f, "??"),
            &Token::Semi => write!(f, ";"),
            &Token::And => write!(f, "&&"),
            &Token::Or => write!(f, "||"),
            &Token::DoubleEq => write!(f, "=="),
            &Token::Ne => write!(f, "!="),
            &Token::Le => write!(f, "<="),
            &Token::Ge => write!(f, ">="),
            &Token::Lt => write!(f, "<"),
            &Token::Gt => write!(f, ">"),
            &Token::Bang => write!(f, "!"),
            &Token::Colon => write!(f, ":"),

            &Token::Minus => write!(f, "-"),
            &Token::Percent => write!(f, "%"),
            &Token::Div => write!(f, "</>"),
            &Token::Add => write!(f, "<+>"),
            &Token::Times => write!(f, "<*>"),

            // Keywords.
            &Token::As => write!(f, "as"),
            &Token::Cap => write!(f, "cap"),
            &Token::Let => write!(f, "let"),
            &Token::True => write!(f, "true"),
            &Token::False => write!(f, "false"),
            &Token::If => write!(f, "if"),
            &Token::Else => write!(f, "else"),
            &Token::In => write!(f, "in"),
            &Token::For => write!(f, "for"),
            &Token::Continue => write!(f, "continue"),
            &Token::Break => write!(f, "break"),
            &Token::While => write!(f, "while"),
            &Token::Loop => write!(f, "loop"),
        }
    }
}

#[derive(Debug)]
pub enum LexicalErrorKind {
    UnclosedRegexLiteral,
    UnclosedRawRegexLiteral,
    EmptyRawRegexLiteral,
    UnclosedStringLiteral,
    UnknownEscapeSequence,

    BadIdentifier,
    BadOperator,

    IntParseError(String, ParseIntError),
    FloatParseError(String, Option<ParseFloatError>),
    ReservedButNotUsedOperator { op: String, end: usize },
    ReservedButNotUsedKeyword { word: String, end: usize },
    UnclosedBlockComment { nest_level: usize },
    UnexpectedChar(char),
}

impl<'input> fmt::Display for LexicalErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &LexicalErrorKind::UnclosedRegexLiteral => {
                write!(f, "Unclosed regex literal.")
            }
            &LexicalErrorKind::UnclosedRawRegexLiteral => {
                write!(f, "Unclosed raw regex literal.")
            }
            &LexicalErrorKind::EmptyRawRegexLiteral => {
                write!(f, "Empty raw regex literal.")
            }
            &LexicalErrorKind::UnclosedStringLiteral => {
                write!(f, "Unclosed string literal.")
            }
            &LexicalErrorKind::UnknownEscapeSequence => {
                write!(f, "Unknown escape sequence.")
            }
            &LexicalErrorKind::BadIdentifier => write!(f, "Bad identifier."),
            &LexicalErrorKind::BadOperator => write!(f, "Bad operator."),

            &LexicalErrorKind::IntParseError(ref num_str, ref err) => {
                write!(f, "Error parsing '{}' as a number: {}.", num_str, err)
            }

            &LexicalErrorKind::FloatParseError(ref num_str, ref err) => write!(
                f,
                "Error parsing '{}' as a number{}.",
                num_str,
                match err {
                    &Some(ref e) => format!(": {}", e),
                    &None => "".to_string(),
                }
            ),

            &LexicalErrorKind::ReservedButNotUsedOperator {
                ref op,
                end: _,
            } => write!(f, "Reserved operator: '{}'.", op),

            &LexicalErrorKind::ReservedButNotUsedKeyword {
                ref word,
                end: _,
            } => write!(f, "Reserved keyword: '{}'.", word),

            &LexicalErrorKind::UnclosedBlockComment { ref nest_level } => {
                write!(
                    f,
                    "Unclosed block comment (nested {} levels deep).",
                    nest_level
                )
            }

            &LexicalErrorKind::UnexpectedChar(ref c) => {
                write!(f, "Unexpected char '{}'.", c)
            }
        }
    }
}

pub struct Lexer<'input> {
    input: &'input str,
    at: usize,
    current_token_start: usize,

    char_indices: CharIndices<'input>,

    /// Cache the most recently returned char to support one character
    /// of lookahead.
    lookahead: Option<(usize, char)>,

    word_re: Regex,
    operator_re: Regex,
}

impl<'input> Lexer<'input> {
    pub fn new(input: &'input str) -> Self {
        let mut ci = input.char_indices();
        Lexer {
            input: input,
            current_token_start: 0,
            // The first .bump() returns the first char, so we don't
            // start off with a coherent `at` value.
            at: 0xdeadbeef,
            lookahead: ci.next(),
            char_indices: ci,

            word_re: Regex::new(r"^[a-zA-Z_][a-zA-Z0-9_]*").unwrap(),
            operator_re: Regex::new(
                r"^(:?\(|\)|\{|\}\?|\}|,|;|<\*>|\*\?|\*|\+\?|\+|\?\?|\?|\.\.|\.|==|\|\||\||&&|=>|<=|>=|<\+>|</>|<|>|!=|!|=|-|%|:|\[|\])").unwrap(),
        }
    }

    fn bump(&mut self) -> Option<(usize, char)> {
        match self.lookahead {
            Some(look) => {
                self.lookahead = self.char_indices.next();
                self.at = look.0;
                Some(look)
            }
            None => None,
        }
    }

    fn drop_n(&mut self, n: usize) {
        for _ in 0..n {
            self.bump();
        }
    }

    fn error(&self, e: LexicalErrorKind) -> InternalError {
        InternalError::new(
            error::ErrorKind::LexicalError(e),
            Span {
                start: self.current_token_start,
                end: self.at + 1,
            },
        )
    }

    fn look_check<F>(&self, pred: F) -> bool
    where
        F: Fn(char) -> bool,
    {
        self.lookahead.map(|(_, c)| pred(c)).unwrap_or(false)
    }

    fn spanned(
        start: usize,
        tail: Result<(Token<'input>, usize), InternalError>,
    ) -> Option<Spanned<Token<'input>, usize, InternalError>> {
        Some(tail.map(|(t, end)| (start, t, end)))
    }

    fn is_start_word_char(&self, c: char) -> bool {
        match c {
            '_' | 'a'...'z' | 'A'...'Z' => true,
            _ => false,
        }
    }

    fn is_num_char(&self, c: char) -> bool {
        match c {
            '0'...'9' => true,
            _ => false,
        }
    }

    fn is_start_operator_char(&self, c: char) -> bool {
        match c {
            '{' | '}' | '*' | '+' | '?' | '.' | '|' | ',' | ';' | '=' | '('
            | ')' | '!' | '&' | '<' | '>' | '-' | '/' | '%' | ':' | '['
            | ']' => true,

            // reserved but not used
            '^' => true,
            _ => false,
        }
    }

    //
    // Custom Token Tail Consumption Functions
    //
    // We only enter a token tail function when we have already checked
    // that the first char matches. The tail function is responsible for
    // returning the token and the terminal index of the token.
    //

    fn block_comment(&mut self) -> Result<(), InternalError> {
        // First drop the leading '*' that the lexeme dispatch loop already
        // checked for.
        self.bump();

        let mut open_blocks = 1;

        while let Some((_, c)) = self.bump() {
            match c {
                '/' if self.look_check(|c| c == '*') => {
                    self.bump();
                    open_blocks += 1;
                }
                '*' if self.look_check(|c| c == '/') => {
                    open_blocks -= 1;
                    if open_blocks <= 0 {
                        self.bump();
                        return Ok(());
                    }
                }
                _ => continue,
            }
        }

        Err(self.error(LexicalErrorKind::UnclosedBlockComment {
            nest_level: open_blocks,
        }))
    }

    fn line_comment(&mut self) {
        while let Some((_, c)) = self.bump() {
            if c == '\n' {
                return;
            }
        }
    }

    fn regex_lit(&mut self) -> Result<(Token<'input>, usize), InternalError> {
        let mut s = String::new();

        while let Some((idx, c)) = self.bump() {
            match c {
                '\\' if self.look_check(|c| c == '/') => {
                    s.push('/');
                    self.bump();
                }
                '/' => return Ok((Token::RegexLit(s), idx + 1)),
                c => s.push(c),
            }
        }

        Err(self.error(LexicalErrorKind::UnclosedRegexLiteral))
    }

    fn string_lit(&mut self) -> Result<(Token<'input>, usize), InternalError> {
        let mut s = String::new();

        while let Some((idx, c)) = self.bump() {
            match c {
                '\\' => {
                    if let Some((_, c_next)) = self.bump() {
                        if c_next == '"' || c_next == '\\' {
                            s.push(c_next);
                        } else {
                            return Err(self.error(
                                LexicalErrorKind::UnknownEscapeSequence,
                            ));
                        }
                    }
                }
                '"' => return Ok((Token::StringLit(s), idx + 1)),
                c => s.push(c),
            }
        }

        Err(self.error(LexicalErrorKind::UnclosedStringLiteral))
    }

    fn raw_regex_lit(
        &mut self,
    ) -> Result<(Token<'input>, usize), InternalError> {
        let start = match self.bump() {
            Some((_, '\'')) => {
                return Err(self.error(LexicalErrorKind::EmptyRawRegexLiteral))
            }
            Some((idx, _)) => idx,
            None => {
                return Err(self.error(
                    LexicalErrorKind::UnclosedRawRegexLiteral,
                ))
            }
        };

        while let Some((idx, c)) = self.bump() {
            match c {
                '\'' => {
                    return Ok((
                        Token::RawRegexLit(&self.input[start..idx]),
                        idx + 1,
                    ))
                }
                _ => continue,
            }
        }

        Err(self.error(LexicalErrorKind::UnclosedRawRegexLiteral))
    }

    // either a keyword or an identifier
    fn word(&mut self) -> Result<(Token<'input>, usize), InternalError> {
        let start = self.at;

        match self.word_re.find(&self.input[start..]) {
            Some(m) => {
                // the regex does not know where we really are in the string.
                let end = start + m.end();

                self.drop_n(end - start - 1);

                match m.as_str() {
                    // actual keywords
                    "cap" => Ok((Token::Cap, end)),
                    "as" => Ok((Token::As, end)),
                    "let" => Ok((Token::Let, end)),
                    "true" => Ok((Token::True, end)),
                    "false" => Ok((Token::False, end)),
                    "if" => Ok((Token::If, end)),
                    "else" => Ok((Token::Else, end)),
                    "in" => Ok((Token::In, end)),
                    "for" => Ok((Token::For, end)),
                    "continue" => Ok((Token::Continue, end)),
                    "break" => Ok((Token::Break, end)),
                    "while" => Ok((Token::While, end)),
                    "loop" => Ok((Token::Loop, end)),

                    // Reserved Keywords
                    //
                    // "structured" is for a "structured typeof <expr>"
                    // expression which returns a more complicated
                    // description of types than the simple string from typeof.
                    "fn" | "match" | "enum" | "return" | "typeof"
                    | "structured" | "struct" => Err(self.error(
                        LexicalErrorKind::ReservedButNotUsedKeyword {
                            word: String::from(m.as_str()),
                            end: end,
                        },
                    )),

                    id => Ok((Token::Id(id), end)),
                }
            }
            None => Err(self.error(LexicalErrorKind::BadIdentifier)),
        }
    }

    fn num(&mut self) -> Result<(Token<'input>, usize), InternalError> {
        let start = self.at;
        let mut saw_dot = false;
        let mut just_saw_dot = false;

        let mut end = start + 1;
        while self.look_check(|c| self.is_num_char(c) || c == '.') {
            match self.bump() {
                Some((idx, c)) => {
                    if c == '.' {
                        saw_dot = true;
                        just_saw_dot = true;
                    } else {
                        just_saw_dot = false;
                    }
                    end = idx + 1
                }
                None => {
                    end = self.input.len();
                    break;
                }
            }
        }

        // we can't end with a dot
        if just_saw_dot {
            return Err(self.error(LexicalErrorKind::FloatParseError(
                String::from(&self.input[start..end]),
                None,
            )));
        }

        Ok(if saw_dot {
            (
                Token::FloatLit(f64::from_str(&self.input[start..end])
                    .map_err(|e| {
                        self.error(LexicalErrorKind::FloatParseError(
                            String::from(&self.input[start..end]),
                            Some(e),
                        ))
                    })?),
                end,
            )
        } else {
            (
                Token::IntLit(i64::from_str(&self.input[start..end])
                    .map_err(|e| {
                        self.error(LexicalErrorKind::IntParseError(
                            String::from(&self.input[start..end]),
                            e,
                        ))
                    })?),
                end,
            )
        })
    }

    fn operator(&mut self) -> Result<(Token<'input>, usize), InternalError> {
        let start = self.at;

        match self.operator_re.find(&self.input[start..]) {
            Some(m) => {
                // the regex does not know where we really are in the string.
                let end = start + m.end();

                self.drop_n(end - start - 1);

                match m.as_str() {
                    "(" => Ok((Token::OpenParen, end)),
                    ")" => Ok((Token::CloseParen, end)),
                    "[" => Ok((Token::OpenBracket, end)),
                    "]" => Ok((Token::CloseBracket, end)),

                    "{" => Ok((Token::OpenCurly, end)),
                    "}?" => Ok((Token::LazyCloseCurly, end)),
                    "}" => Ok((Token::CloseCurly, end)),

                    "," => Ok((Token::Comma, end)),
                    ";" => Ok((Token::Semi, end)),

                    "*?" => Ok((Token::LazyStar, end)),
                    "*" => Ok((Token::Star, end)),

                    "+?" => Ok((Token::LazyPlus, end)),
                    "+" => Ok((Token::Plus, end)),

                    "|" => Ok((Token::Pipe, end)),
                    "." => Ok((Token::Dot, end)),
                    "=" => Ok((Token::Equals, end)),
                    "!" => Ok((Token::Bang, end)),

                    "?" => Ok((Token::Question, end)),
                    "??" => Ok((Token::LazyQuestion, end)),

                    "==" => Ok((Token::DoubleEq, end)),
                    "&&" => Ok((Token::And, end)),
                    "||" => Ok((Token::Or, end)),
                    "<" => Ok((Token::Lt, end)),
                    ">" => Ok((Token::Gt, end)),
                    "<=" => Ok((Token::Le, end)),
                    ">=" => Ok((Token::Ge, end)),
                    "!=" => Ok((Token::Ne, end)),

                    "-" => Ok((Token::Minus, end)),
                    "%" => Ok((Token::Percent, end)),
                    "</>" => Ok((Token::Div, end)),
                    "<*>" => Ok((Token::Times, end)),
                    "<+>" => Ok((Token::Add, end)),
                    ":" => Ok((Token::Colon, end)),

                    "=>" => Err(self.error(
                        LexicalErrorKind::ReservedButNotUsedOperator {
                            op: String::from(&self.input[start..end]),
                            end: end,
                        },
                    )),

                    // unreachable
                    _ => Err(self.error(LexicalErrorKind::BadOperator)),
                }
            }
            None => Err(self.error(LexicalErrorKind::BadOperator)),
        }
    }
}

impl<'input> Iterator for Lexer<'input> {
    type Item = Spanned<Token<'input>, usize, InternalError>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((start, c)) = self.bump() {
            self.current_token_start = start;
            return match c {
                '/' if self.look_check(|c| c == '*') => {
                    match self.block_comment() {
                        Err(e) => return Some(Err(e)),
                        _ => {} // FALLTHROUGH
                    }
                    continue;
                }
                '/' if self.look_check(|c| c == '/') => {
                    self.line_comment();
                    continue;
                }
                '/' => Self::spanned(start, self.regex_lit()),

                '\'' => Self::spanned(start, self.raw_regex_lit()),

                '"' => Self::spanned(start, self.string_lit()),

                c if self.is_start_word_char(c) => {
                    Self::spanned(start, self.word())
                }

                c if self.is_num_char(c) => Self::spanned(start, self.num()),

                c if self.is_start_operator_char(c) => {
                    Self::spanned(start, self.operator())
                }

                c if c.is_whitespace() => continue,

                c => Some(Err(self.error(LexicalErrorKind::UnexpectedChar(c)))),
            };
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An equality function for remake runtime values.
    ///
    /// We give this guys an awkward name rather than just adding
    /// an Eq impl to avoid taking a position about the right way to
    /// compare floating point values in all cases. For testing this
    /// function is good enough.
    fn test_eq(lhs: &Token, rhs: &Token) -> bool {
        match (lhs, rhs) {
            (&Token::RegexLit(ref l), &Token::RegexLit(ref r)) => *l == *r,
            (&Token::RawRegexLit(ref l), &Token::RawRegexLit(ref r)) => {
                *l == *r
            }
            (&Token::IntLit(ref l), &Token::IntLit(ref r)) => *l == *r,
            (&Token::StringLit(ref l), &Token::StringLit(ref r)) => *l == *r,
            (&Token::Id(ref l), &Token::Id(ref r)) => *l == *r,

            (&Token::OpenParen, &Token::OpenParen) => true,
            (&Token::CloseParen, &Token::CloseParen) => true,
            (&Token::OpenCurly, &Token::OpenCurly) => true,
            (&Token::LazyCloseCurly, &Token::LazyCloseCurly) => true,
            (&Token::CloseCurly, &Token::CloseCurly) => true,
            (&Token::LazyStar, &Token::LazyStar) => true,
            (&Token::Star, &Token::Star) => true,
            (&Token::LazyPlus, &Token::LazyPlus) => true,
            (&Token::Plus, &Token::Plus) => true,
            (&Token::Comma, &Token::Comma) => true,
            (&Token::Pipe, &Token::Pipe) => true,
            (&Token::Dot, &Token::Dot) => true,
            (&Token::Equals, &Token::Equals) => true,
            (&Token::Question, &Token::Question) => true,
            (&Token::LazyQuestion, &Token::LazyQuestion) => true,
            (&Token::Semi, &Token::Semi) => true,
            (&Token::As, &Token::As) => true,
            (&Token::Cap, &Token::Cap) => true,
            (&Token::Let, &Token::Let) => true,
            (&Token::True, &Token::True) => true,
            (&Token::False, &Token::False) => true,

            (&Token::And, &Token::And) => true,
            (&Token::Or, &Token::Or) => true,
            (&Token::DoubleEq, &Token::DoubleEq) => true,
            (&Token::Ne, &Token::Ne) => true,
            (&Token::Le, &Token::Le) => true,
            (&Token::Ge, &Token::Ge) => true,
            (&Token::Lt, &Token::Lt) => true,
            (&Token::Gt, &Token::Gt) => true,
            (&Token::Bang, &Token::Bang) => true,

            // stupid fixed-epsilon test
            (&Token::FloatLit(ref l), &Token::FloatLit(ref r)) => {
                (*l - *r).abs() < 0.0000001
            }

            (_, _) => false,
        }
    }

    macro_rules! tokens {
        ($fn_name:ident, $remake_source:expr, $( $token:expr ),*) => {
            #[test]
            fn $fn_name() {
                let tokens = Lexer::new($remake_source)
                    .map(|tok| tok.map(|(_, t, _)| t))
                    .collect::<Result<Vec<_>, _>>();

                let expected_tokens = vec![$($token),*];

                assert!(
                    expected_tokens.iter().zip(
                        tokens.expect("the source to lex.").iter())
                        .all(|(r, l)| test_eq(r, l)));
            }
        }
    }

    macro_rules! lex_error_has {
        ($fn_name:ident, $remake_source:expr, $lex_err:expr) => {
            #[test]
            fn $fn_name() {
                let tokens = Lexer::new($remake_source)
                    .map(|tok| tok.map(|(_, t, _)| t))
                    .collect::<Result<Vec<_>, _>>();

                match tokens {
                    Err(ref err) => {
                        let err_str =
                            format!("{}", err.overlay($remake_source));
                        // assert_eq!($lex_err, err_str);

                        assert!(err_str.contains($lex_err), err_str);
                    }
                    Ok(ts) => panic!("Should not lex. ts={:?}", ts),
                }
            }
        };
    }

    macro_rules! spanned {
        ($fn_name:ident, $remake_source:expr, $span_spec:expr) => {
            #[test]
            fn $fn_name() {
                let token_spans = Lexer::new($remake_source)
                    .map(|tok| tok.map(|(l, _, r)| (l, r)))
                    .collect::<Result<Vec<_>, _>>()
                    .expect("the source to lex");

                let s_spec = $span_spec;
                let mut spans = String::with_capacity(s_spec.len());

                let mut i = 0;
                for &(l, r) in token_spans.iter() {
                    while i < l {
                        spans.push(' ');
                        i += 1;
                    }
                    while i < r {
                        spans.push('~');
                        i += 1;
                    }
                }
                while i < s_spec.len() {
                    spans.push(' ');
                    i += 1;
                }

                assert_eq!(s_spec, spans);
            }
        };
    }

    macro_rules! tok_round_trip {
        ($fn_name:ident, $remake_source:expr) => {
            #[test]
            fn $fn_name() {
                let tok = Lexer::new($remake_source)
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap()[0]
                    .1
                    .clone();
                assert_eq!($remake_source, &format!("{}", tok));
            }
        };
        ($fn_name:ident, $remake_source:expr, $expected:expr) => {
            #[test]
            fn $fn_name() {
                let tok = Lexer::new($remake_source)
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap()[0]
                    .1
                    .clone();
                assert_eq!($expected, &format!("{}", tok));
            }
        };
    }

    //
    // Operators
    //

    tokens!(open_curly_1_, "{", Token::OpenCurly);
    tokens!(open_curly_2_, " {", Token::OpenCurly);
    tokens!(open_curly_3_, "{ ", Token::OpenCurly);

    tokens!(close_curly_1_, "}", Token::CloseCurly);
    tokens!(close_curly_2_, " } ", Token::CloseCurly);
    tokens!(close_curly_3_, "}?", Token::LazyCloseCurly);
    tokens!(close_curly_4_, " }? ", Token::LazyCloseCurly);

    tokens!(open_paren_1_, "(", Token::OpenParen);
    tokens!(open_paren_2_, " ( ", Token::OpenParen);

    tokens!(question_1_, "?", Token::Question);
    tokens!(question_2_, "??", Token::LazyQuestion);
    tokens!(question_3_, "???", Token::LazyQuestion, Token::Question);
    tokens!(
        question_4_,
        "????",
        Token::LazyQuestion,
        Token::LazyQuestion
    );
    tokens!(question_5_, ")?", Token::CloseParen, Token::Question);

    tokens!(comma_1_, ",", Token::Comma);
    tokens!(comma_2_, ",,", Token::Comma, Token::Comma);

    tokens!(star_1_, " *", Token::Star);

    tokens!(lazy_star_1_, "    *?  *  ", Token::LazyStar, Token::Star);

    tokens!(pipe_1_, " | ", Token::Pipe);

    tokens!(dot_1_, " . ", Token::Dot);

    tokens!(equals_1_, "=", Token::Equals);

    //
    // keywords
    //

    tokens!(keywords_1_, "cap", Token::Cap);
    tokens!(keywords_2_, "cap as", Token::Cap, Token::As);
    tokens!(
        keywords_3_,
        "as cap foo as bar",
        Token::As,
        Token::Cap,
        Token::Id("foo"),
        Token::As,
        Token::Id("bar")
    );

    //
    // identifiers
    //

    tokens!(ident_1_, "a,", Token::Id("a"), Token::Comma);

    //
    // Regex literals
    //

    tokens!(regex_lit_1_, "/foo/", Token::RegexLit("foo".to_string()));
    tokens!(
        regex_lit_2_,
        r" /fo\/o/ ",
        Token::RegexLit("fo/o".to_string())
    );

    lex_error_has!(regex_lit_3_, r" /fo\/ ", "Unclosed regex literal");
    lex_error_has!(regex_lit_4_, r" /fo ", "Unclosed regex literal");
    lex_error_has!(regex_lit_5_, r" '' ", "Empty raw regex literal");
    lex_error_has!(regex_lit_6_, " \"this isnt closed ", "Unclosed string");
    lex_error_has!(regex_lit_7_, " \"\\x\" ", "Unknown escape");
    lex_error_has!(regex_lit_8_, " \\ ", "Unexpected char");

    tokens!(raw_regex_lit_1_, "'foo'", Token::RawRegexLit("foo"));

    // raw regex don't have escape codes
    tokens!(raw_regex_lit_2_, r" '\' ", Token::RawRegexLit(r"\"));

    lex_error_has!(raw_regex_lit_4_, r" 'blah  ", "Unclosed raw regex literal");

    lex_error_has!(raw_regex_lit_5_, r"a'", "Unclosed raw");
    lex_error_has!(raw_regex_lit_6_, r"'", "Unclosed raw");

    //
    // Comments
    //

    tokens!(
        comment_1_,
        r#"
        // this is a comment
        // as is this
        this isnt
        // but this is
    "#,
        Token::Id(r"this"),
        Token::Id("isnt")
    );

    tokens!(
        comment_2_,
        r#"
        /* this is a comment */
        // as is this
        this isnt
        // but this is
    "#,
        Token::Id(r"this"),
        Token::Id("isnt")
    );

    tokens!(
        comment_3_,
        r#"
        /* this is a comment
        // as is this
        this isnt
        // but this is */ cap
    "#,
        Token::Cap
    );

    tokens!(
        comment_4_,
        r#"
        /**//**/ cap
    "#,
        Token::Cap
    );

    tokens!(
        comment_5_,
        r#"
        /* comments /* can
         *
         * /*
         *
         * be */ nested */ deeply
        */ cap
    "#,
        Token::Cap
    );

    lex_error_has!(comments_6_, "/* unclosed", "Unclosed block comment");

    //
    // Numbers
    //

    tokens!(num_1_, r" 56 98 ", Token::IntLit(56), Token::IntLit(98));
    lex_error_has!(num_2_, r" 56999999999999999999999 ", "number too large");
    tokens!(num_3_, r" 56,", Token::IntLit(56), Token::Comma);
    tokens!(num_4_, r" 5", Token::IntLit(5));
    tokens!(num_5_, r" 5 ", Token::IntLit(5));

    //
    // Strings
    //

    tokens!(str_1_, " \"hello\" ", Token::StringLit("hello".to_string()));

    tokens!(
        str_2_,
        " \"hello world\" ",
        Token::StringLit("hello world".to_string())
    );

    tokens!(
        str_3_,
        " \"hello \\\\ world\" ",
        Token::StringLit("hello \\ world".to_string())
    );

    tokens!(
        str_4_,
        " \"hello \\\" world\" ",
        Token::StringLit("hello \" world".to_string())
    );

    //
    // Spans
    //

    spanned!(span_regex_lit_1_, "   /foo/    ", "   ~~~~~    ");

    spanned!(
        span_regex_lit_2_,
        "  /foo/ + /bar/    ",
        "  ~~~~~ ~ ~~~~~    "
    );

    spanned!(span_raw_regex_lit_1_, "   'foo'    ", "   ~~~~~    ");

    spanned!(span_plus_1_, "   +  ", "   ~  ");

    spanned!(span_id_1_, " blah  ", " ~~~~  ");

    spanned!(span_comma_1_, " , ", " ~ ");

    spanned!(span_comma_2_, " 56, ", " ~~~ ");

    spanned!(
        span_comment_1_,
        " (blah . ){56, /* 32, */ 9} ",
        " ~~~~~ ~ ~~~~~           ~~ "
    );

    spanned!(
        span_comment_2_,
        " ( blah . ) { 56 , /* 32, */ 9 } ",
        " ~ ~~~~ ~ ~ ~ ~~ ~           ~ ~ "
    );

    spanned!(
        span_comment_3_,
        " ( blah . ) { 56 , // 32, */ 9 } ",
        " ~ ~~~~ ~ ~ ~ ~~ ~               "
    );

    //
    // Ensure that non-parameterized tokens are represented
    // the same way they look.
    //

    tok_round_trip!(trt_1_, "{");
    tok_round_trip!(trt_2_, "}");
    tok_round_trip!(trt_3_, "}?");
    tok_round_trip!(trt_4_, "=");
    tok_round_trip!(trt_5_, "(");
    tok_round_trip!(trt_6_, ")");
    tok_round_trip!(trt_7_, "foo", "identifier: foo");
    tok_round_trip!(trt_8_, "cap");
    tok_round_trip!(trt_9_, "as");
    tok_round_trip!(trt_10_, "let");
    tok_round_trip!(trt_11_, ".");
    tok_round_trip!(trt_12_, ",");
    tok_round_trip!(trt_13_, "9");
    tok_round_trip!(trt_14_, "+");
    tok_round_trip!(trt_15_, "+?");
    tok_round_trip!(trt_16_, "?");
    tok_round_trip!(trt_17_, "??");
    tok_round_trip!(trt_18_, "*");
    tok_round_trip!(trt_19_, "*?");
    tok_round_trip!(trt_20_, "|");
    tok_round_trip!(trt_21_, ";");
    tok_round_trip!(trt_22_, "true");
    tok_round_trip!(trt_23_, "false");
    tok_round_trip!(trt_24_, "&&");
    tok_round_trip!(trt_25_, "||");
    tok_round_trip!(trt_26_, "==");
    tok_round_trip!(trt_27_, "!=");
    tok_round_trip!(trt_28_, "<=");
    tok_round_trip!(trt_29_, ">=");
    tok_round_trip!(trt_30_, "<");
    tok_round_trip!(trt_31_, ">");
    tok_round_trip!(trt_32_, "!");
    tok_round_trip!(trt_33_, "-");
    tok_round_trip!(trt_35_, "%");
    tok_round_trip!(trt_36_, "</>");
    tok_round_trip!(trt_37_, "<+>");
    tok_round_trip!(trt_38_, "<*>");
    tok_round_trip!(trt_39_, "3.0");
    tok_round_trip!(trt_40_, "\"str\"");
    tok_round_trip!(trt_41_, "'re'");
    tok_round_trip!(trt_42_, ":");
    tok_round_trip!(trt_43_, "if");
    tok_round_trip!(trt_44_, "else");
    tok_round_trip!(trt_45_, "in");
    tok_round_trip!(trt_46_, "for");

    //
    // Specific lexical errors
    //

    lex_error_has!(bad_float_1_, " 5. 0", "as a number");
    tokens!(bad_float_2_, " .5", Token::Dot, Token::IntLit(5));
    spanned!(bad_float_3_, " 5 . 0", " ~ ~ ~");
}
