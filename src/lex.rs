use std::str::{FromStr, CharIndices};
use std::num::ParseIntError;
use std::convert::From;
use std::fmt;

use regex::Regex;

use ast::Span;
use error;
use error::InternalError;

pub type Spanned<Tok, Loc, Error> = Result<(Loc, Tok, Loc), Error>;

#[derive(Debug, PartialEq, Eq)]
pub enum Token<'input> {
    RegexLit(String),
    RawRegexLit(&'input str),
    U32(u32),
    Id(&'input str),

    // Operators
    OpenParen,
    CloseParen,
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

    // Keywords.
    As,
    Cap,
    Let,
}

impl<'input> fmt::Display for Token<'input> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &Token::RegexLit(ref re_src) => write!(f, "/{}/", re_src),
            &Token::RawRegexLit(ref re_src) => write!(f, "'{}'", re_src),
            &Token::U32(ref num) => write!(f, "{}", num),
            &Token::Id(ref id) => write!(f, "identifier {}", id),

            // Operators
            &Token::OpenParen => write!(f, "("),
            &Token::CloseParen => write!(f, ")"),
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

            // Keywords.
            &Token::As => write!(f, "as"),
            &Token::Cap => write!(f, "cap"),
            &Token::Let => write!(f, "let"),
        }
    }
}

#[derive(Debug)]
pub enum LexicalErrorKind {
    UnclosedRegexLiteral,
    UnclosedRawRegexLiteral,
    EmptyRawRegexLiteral,

    BadIdentifier,
    BadOperator,

    NumParseError(String, ParseIntError),
    ReservedButNotUsedOperator { op: String, end: usize },
    ReservedButNotUsedKeyword { word: String, end: usize },
    UnclosedBlockComment { nest_level: usize },
    UnexpectedChar(char),
}

impl<'input> fmt::Display for LexicalErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &LexicalErrorKind::UnclosedRegexLiteral =>
                write!(f, "Unclosed regex literal."),
            &LexicalErrorKind::UnclosedRawRegexLiteral =>
                write!(f, "Unclosed raw regex literal."),
            &LexicalErrorKind::EmptyRawRegexLiteral =>
                write!(f, "Empty raw regex literal."),
            &LexicalErrorKind::BadIdentifier =>
                write!(f, "Bad identifier."),
            &LexicalErrorKind::BadOperator=>
                write!(f, "Bad operator."),

            &LexicalErrorKind::NumParseError( ref num_str, ref err) =>
                write!(f, "Error parsing '{}' as a number: {}.", num_str, err),

            &LexicalErrorKind::ReservedButNotUsedOperator { ref op, end: _ } =>
                write!(f, "Reserved operator: '{}'.", op),

            &LexicalErrorKind::ReservedButNotUsedKeyword{ ref word, end: _ } =>
                write!(f, "Reserved keyword: '{}'.", word),

            &LexicalErrorKind::UnclosedBlockComment { ref nest_level } =>
                write!(f, "Unclosed block comment (nested {} levels deep).",
                            nest_level),

            &LexicalErrorKind::UnexpectedChar(ref c) =>
                write!(f, "Unexpected char '{}'.", c),
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
                r"^(:?\(|\)|\{|\}\?|\}|,|;|\*\?|\*|\+\?|\+|\?\?|\?|\.\.|\.|==|\|\||\||&&|=>|<=|>=|<|>|!=|=)").unwrap(),
        }
    }

    fn bump(&mut self) -> Option<(usize, char)> {
        match self.lookahead {
            Some(look) => {
                self.lookahead = self.char_indices.next();
                self.at = look.0;
                Some(look)
            }
            None => None
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
            })
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
            '{' | '}' | '*' | '+' | '?' | '.' | '|' | ',' | ';'
            | '=' | '(' | ')' => true,

            // reserved but not used
            '&' | '!' | '[' | ']' | '-' | '<' | '>' | '^' => true,
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
            nest_level: open_blocks
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
                },
                '/' => return Ok((Token::RegexLit(s), idx + 1)),
                c => s.push(c)
            }
        }

        Err(self.error(LexicalErrorKind::UnclosedRegexLiteral))
    }

    fn raw_regex_lit(&mut self) -> Result<(Token<'input>, usize), InternalError> {
        let start = match self.bump() {
            Some((_, '\'')) =>
                return Err(self.error(LexicalErrorKind::EmptyRawRegexLiteral)),
            Some((idx, _)) => idx,
            None =>
                return Err(self.error(LexicalErrorKind::UnclosedRawRegexLiteral)),
        };

        while let Some((idx, c)) = self.bump() {
            match c {
                '\'' =>
                    return Ok((Token::RawRegexLit(
                                &self.input[start..idx]), idx + 1)),
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

                    // Reserved Keywords
                    //
                    // "structured" is for a "structured typeof <expr>"
                    // expression which returns a more complicated
                    // description of types than the simple string from typeof.
                    "if" | "while" | "for" | "fn" | "else" | "match"
                    | "enum" | "true" | "false" | "return" | "in"
                    | "typeof" | "structured" | "continue" | "loop"
                    | "break" | "struct" =>
                        Err(self.error(LexicalErrorKind::ReservedButNotUsedKeyword {
                            word: String::from(m.as_str()),
                            end: end
                        })),

                    id => Ok((Token::Id(id), end)),
                }
            }
            None => Err(self.error(LexicalErrorKind::BadIdentifier)),
        }

    }

    fn num(&mut self) -> Result<(Token<'input>, usize), InternalError> {
        let start = self.at;

        let mut end = start + 1;
        while self.look_check(|c| self.is_num_char(c)) {
            match self.bump() {
                Some((idx, _)) => end = idx + 1,
                None => {
                    return Ok((Token::U32(
                        u32::from_str(&self.input[start..])
                            .map_err(|e|
                                self.error(LexicalErrorKind::NumParseError(
                                    String::from(&self.input[start..]), e)))?),
                        self.input.len()))
                }
            }
        }

        
        Ok((Token::U32(
                u32::from_str(&self.input[start..end])
                    .map_err(|e|
                        self.error(LexicalErrorKind::NumParseError(
                            String::from(&self.input[start..end]), e)))?),
            end))
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

                    "?" => Ok((Token::Question, end)),
                    "??" => Ok((Token::LazyQuestion, end)),

                    ".." | "==" | "||" | "&&" | "=>"
                    | "<" | ">" | ">=" | "<=" | "!=" =>
                        Err(self.error(LexicalErrorKind::ReservedButNotUsedOperator {
                            op: String::from(&self.input[start..end]),
                            end: end,
                        })),

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
                        _ => {}, // FALLTHROUGH
                    }
                    continue;
                }
                '/' if self.look_check(|c| c == '/') => {
                    self.line_comment();
                    continue;
                }
                '/' => Self::spanned(start, self.regex_lit()),

                '\'' => Self::spanned(start, self.raw_regex_lit()),

                c if self.is_start_word_char(c) =>
                    Self::spanned(start, self.word()),

                c if self.is_num_char(c) =>
                    Self::spanned(start, self.num()),

                c if self.is_start_operator_char(c) =>
                    Self::spanned(start, self.operator()),

                c if c.is_whitespace() => continue,

                c => Some(Err(self.error(LexicalErrorKind::UnexpectedChar(c)))),
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! tokens {
        ($fn_name:ident, $remake_source:expr, $( $token:expr ),*) => {
            #[test]
            fn $fn_name() {
                let tokens = Lexer::new($remake_source)
                    .map(|tok| tok.map(|(_, t, _)| t))
                    .collect::<Result<Vec<_>, _>>();

                let expected_tokens = vec![$($token),*];

                assert_eq!(
                    expected_tokens,
                    tokens.expect("the source to lex."));
            }
        }
    }

    macro_rules! bad_token {
        ($fn_name:ident, $remake_source:expr) => {
            #[test]
            fn $fn_name() {
                let tokens = Lexer::new($remake_source)
                    .map(|tok| tok.map(|(_, t, _)| t))
                    .collect::<Result<Vec<_>, _>>();

                assert!(!tokens.is_ok(), "tokens={:?}", tokens);
            }
        }
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
        }
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
    tokens!(question_4_, "????", Token::LazyQuestion, Token::LazyQuestion);
    tokens!(question_5_, ")?", Token::CloseParen, Token::Question);

    tokens!(comma_1_, ",", Token::Comma);
    tokens!(comma_2_, ",,", Token::Comma, Token::Comma);

    tokens!(star_1_, " *", Token::Star);

    tokens!(lazy_star_1_, "    *?  *  ", Token::LazyStar, Token::Star);

    tokens!(pipe_1_, " | ", Token::Pipe);

    tokens!(dot_1_, " . ", Token::Dot);

    tokens!(equals_1_ , "=", Token::Equals);

    bad_token!(unknown_op_1_, " || ");
    bad_token!(unknown_op_2_, "   && ");
    bad_token!(unknown_op_3_, "   == ");
    bad_token!(unknown_op_4_, "  .. ");

    //
    // keywords
    //

    tokens!(keywords_1_, "cap", Token::Cap);
    tokens!(keywords_2_, "cap as", Token::Cap, Token::As);
    tokens!(keywords_3_, "as cap foo as bar",
            Token::As, Token::Cap, Token::Id("foo"),
            Token::As, Token::Id("bar"));

    //
    // identifiers
    //

    tokens!(ident_1_, "a,", Token::Id("a"), Token::Comma);

    //
    // Regex literals
    //

    tokens!(regex_lit_1_, "/foo/", Token::RegexLit("foo".to_string()));
    tokens!(regex_lit_2_, r" /fo\/o/ ",
        Token::RegexLit("fo/o".to_string()));

    bad_token!(regex_lit_3_, r" /fo\/ ");
    bad_token!(regex_lit_4_, r" /fo ");

    tokens!(raw_regex_lit_1_, "'foo'", Token::RawRegexLit("foo"));

    // raw regex don't have escape codes
    tokens!(raw_regex_lit_2_, r" '\' ", Token::RawRegexLit(r"\"));

    bad_token!(raw_regex_lit_3_, r" '' "); // no empty allowed
    bad_token!(raw_regex_lit_4_, r" 'blah  ");

    bad_token!(raw_regex_lit_5_, r"a'");
    bad_token!(raw_regex_lit_6_, r"'");


    //
    // Comments
    //

    tokens!(comment_1_, r#"
        // this is a comment
        // as is this
        this isnt
        // but this is
    "#, Token::Id(r"this"), Token::Id("isnt"));

    tokens!(comment_2_, r#"
        /* this is a comment */
        // as is this
        this isnt
        // but this is
    "#, Token::Id(r"this"), Token::Id("isnt"));

    tokens!(comment_3_, r#"
        /* this is a comment
        // as is this
        this isnt
        // but this is */ cap
    "#, Token::Cap);

    tokens!(comment_4_, r#"
        /**//**/ cap
    "#, Token::Cap);

    tokens!(comment_5_, r#"
        /* comments /* can
         *
         * /*
         *
         * be */ nested */ deeply
        
        */ cap
    "#, Token::Cap);

    //
    // Numbers
    //
    
    tokens!(num_1_, r" 56 98 ", Token::U32(56), Token::U32(98));
    bad_token!(num_2_, r" 56999999999999999999999 ");
    tokens!(num_3_, r" 56,", Token::U32(56), Token::Comma);
    tokens!(num_4_, r" 5", Token::U32(5));
    tokens!(num_5_, r" 5 ", Token::U32(5));

    //
    // Spans
    //

    spanned!(span_regex_lit_1_,
        "   /foo/    ",
        "   ~~~~~    ");

    spanned!(span_regex_lit_2_,
        "  /foo/ + /bar/    ",
        "  ~~~~~ ~ ~~~~~    ");

    spanned!(span_raw_regex_lit_1_,
        "   'foo'    ",
        "   ~~~~~    ");

    spanned!(span_plus_1_,
        "   +  ",
        "   ~  ");

    spanned!(span_id_1_,
        " blah  ",
        " ~~~~  ");

    spanned!(span_comma_1_,
        " , ",
        " ~ ");

    spanned!(span_comma_2_,
        " 56, ",
        " ~~~ ");

    spanned!(span_comment_1_,
        " (blah . ){56, /* 32, */ 9} ",
        " ~~~~~ ~ ~~~~~           ~~ ");

    spanned!(span_comment_2_,
        " ( blah . ) { 56 , /* 32, */ 9 } ",
        " ~ ~~~~ ~ ~ ~ ~~ ~           ~ ~ ");

    spanned!(span_comment_3_,
        " ( blah . ) { 56 , // 32, */ 9 } ",
        " ~ ~~~~ ~ ~ ~ ~~ ~               ");

}
