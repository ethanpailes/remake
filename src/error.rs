// Copyright 2018 Ethan Pailes.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;

use ast::Span;
use lex;

/// A structured remake error. Can be a parse error OR a runtime error.
///
/// This error type is never exposed to the user. We go to all this
/// trouble anyway in order to be able to provide nice formatted error
/// messages in the user facing error type.
///
/// We choose to use this type to represent both parse errors and runtime
/// errors so that we can re-use the formatting code. If this type
/// was exposed to the user, we would probably want to split that up, but
/// we don't care about reporting.
///
/// Also worth noting is the fact that we don't impl std::error::Error.
/// Again, this isn't user facing, so there is no need.
#[derive(Debug)]
pub struct InternalError {
    pub kind: ErrorKind,
    pub span: Span,
}

impl InternalError {
    pub fn new(kind: ErrorKind, span: Span) -> Self {
        Self {
            kind: kind,
            span: span,
        }
    }

    pub fn overlay<'a, 'e>(&'e self, src: &'a str) -> ErrorSrcOverlay<'a, 'e> {
        ErrorSrcOverlay {
            src: src,
            err: &self,
        }
    }
}

// We implement Display for Error so that we can just display an
// lalrpop error directly, but we never do that without first special
// casing our errors.
impl fmt::Display for InternalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f,
            r#"Bug in remake. Internal errors should never be directly formatted.
               Please report this to https://github.com/ethanpailes/remake.
            "#)?;
        Ok(())
    }
}

pub struct ErrorSrcOverlay<'a, 'e> {
    src: &'a str,
    err: &'e InternalError,
}

impl<'a, 'e> fmt::Display for ErrorSrcOverlay<'a, 'e> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use error::ErrorKind::*;

        write!(
            f,
            "{}",
            block_leftpad(
                &PosSpan::new(&self.src, &self.err.span).highlight(&self.src),
                4
            )
        )?;

        match &self.err.kind {
            &RegexError { ref re, ref err } => {
                writeln!(f, "Error parsing the regex literal: /{}/", re)?;
                write!(f, "{}", block_leftpad(err, 4))?;
            }
            &LexicalError(ref kind) => {
                writeln!(f, "remake lexical error:")?;
                write!(f, "{}", kind)?;
            }
            &InvalidToken => {
                writeln!(f, "Invalid token.")?;
            }
            &UnrecognizedToken {
                ref token,
                ref expected,
            } => {
                write!(f, "Unexpected token '{}'.", token)?;
                if expected.len() > 0 {
                    write!(f, " Expected one of:")?;
                    for e in expected {
                        write!(f, " {}", e)?;
                    }
                }
                write!(f, "\n")?;
            }

            &NameError { ref name } => {
                writeln!(f, "NameError: unknown variable '{}'.", name)?;
            }
            &KeyError { ref key } => {
                writeln!(f, "KeyError: {}.", key)?;
            }
            &TypeError {
                ref expected,
                ref actual,
            } => match expected.len() {
                0 => writeln!(f, "TypeError: unexpected type {}", actual)?,
                1 => writeln!(
                    f,
                    "TypeError: must be {}, not {}",
                    expected[0], actual
                )?,
                _ => writeln!(
                    f,
                    "TypeError: must be one of {}, not {}",
                    expected.join(", "),
                    actual
                )?,
            },
            &ZeroDivisionError { ref neum } => {
                writeln!(
                    f,
                    "ZeroDivisionError: tried to divide {} by zero",
                    neum
                )?;
            }
            &LoopError { ref keyword } => {
                writeln!(
                    f,
                    "LoopError: '{}' not properly within loop",
                    match keyword {
                        &LoopErrorKind::Break => "break",
                        &LoopErrorKind::Continue => "continue",
                    }
                )?;
            }
            &FinalValueNotRegex { ref actual } => {
                writeln!(
                    f,
                    "remake expressions must evaluate to a regex not a {}",
                    actual
                )?;
            }
        }

        Ok(())
    }
}

impl<'a, 'e> fmt::Debug for ErrorSrcOverlay<'a, 'e> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{}", self)?;
        Ok(())
    }
}

#[derive(Debug)]
pub enum ErrorKind {
    //
    // Custom lexical error
    //
    LexicalError(lex::LexicalErrorKind),

    //
    // Repackaged parse errors so that we can highlight them
    // in the source nicely.
    //
    InvalidToken,
    UnrecognizedToken {
        token: String,
        expected: Vec<String>,
    },
    RegexError {
        re: String,
        err: String,
    },

    //
    // runtime errors
    //
    NameError {
        name: String,
    },
    TypeError {
        actual: String,
        expected: Vec<String>,
    },
    ZeroDivisionError {
        neum: String,
    },
    KeyError {
        key: String,
    },
    FinalValueNotRegex {
        actual: String,
    },
    LoopError {
        keyword: LoopErrorKind,
    },
}

#[derive(Debug)]
pub enum LoopErrorKind {
    Continue,
    Break,
}

#[derive(Debug)]
struct PosSpan {
    /// 1-indexed starting line number
    start_line: usize,
    /// 1-indexed starting col number
    start_col: usize,
    /// 1-indexed ending line number
    end_line: usize,
    /// 1-indexed ending col number
    end_col: usize,
}

impl PosSpan {
    /// Construct a positional span from a span and the source.
    ///
    /// We take a complexity and speed hit here in order to make thinking
    /// about spans easier for anyone who wants to construct an error.
    /// Remake sources are small, so the perf hit is not bad, and it is
    /// much easier to think about spans in terms of char offsets for the
    /// parser and interpreter.
    fn new(src: &str, span: &Span) -> Self {
        debug_assert!(span.start < src.len());
        debug_assert!(span.end <= src.len());
        debug_assert!(span.start != span.end);

        let mut line = 1;
        let mut col = 1;
        let mut idx = 0;

        let mut ps = PosSpan {
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
        };

        for c in src.chars() {
            if idx == span.start {
                ps.start_line = line;
                ps.start_col = col;
            } else if idx == span.end {
                ps.end_line = line;
                ps.end_col = col;
            }

            if c == '\n' {
                line += 1;
                col = 0;
            }
            col += 1;
            idx += 1;
        }
        if ps.end_line == 0 {
            ps.end_line = line;
            ps.end_col = col;
        }

        debug_assert!(ps.start_line != 0);
        debug_assert!(ps.start_col != 0);
        debug_assert!(ps.end_line != 0);
        debug_assert!(ps.end_col != 0);
        ps
    }

    /// Return a string highlighting the given positional span.
    fn highlight(&self, src: &str) -> String {
        if self.start_line != self.end_line {
            self.highlight_multiline(src)
        } else {
            let mut s = format!(
                "at line {}, col {}:\n",
                self.start_line, self.start_col
            );

            for (i, line) in src.split("\n").enumerate() {
                let line_no = i + 1;

                // Print two lines of context
                if line_no > self.start_line.saturating_sub(Self::CONTEXT_LINES)
                    && line_no
                        < self.start_line.saturating_add(Self::CONTEXT_LINES)
                {
                    s.push_str(&format!("{:04} > ", line_no));
                    s.push_str(line);
                    s.push('\n');
                }

                // Print uppercut chars indicating the section that
                // caused the error.
                if line_no == self.start_line {
                    s.push_str("       "); // to match the start line indicator
                    for i in 0..(self.end_col - 1) {
                        let col_no = i + 1;

                        if col_no < self.start_col {
                            s.push(' ');
                        } else {
                            s.push('^');
                        }
                    }
                    s.push('\n');
                }
            }

            s
        }
    }

    fn highlight_multiline(&self, src: &str) -> String {
        let mut s = format!(
            "starting at line {}, col {} and ending at line {}, col {}:\n",
            self.start_line, self.start_col, self.end_line, self.end_col
        );

        let mut printed_dots = false;
        for (i, line) in src.split("\n").enumerate() {
            let line_no = i + 1;

            // Print two lines of context around the starting line
            if line_no > self.start_line.saturating_sub(Self::CONTEXT_LINES)
                && line_no < self.start_line.saturating_add(Self::CONTEXT_LINES)
            {
                s.push_str(&format!("{:04}  > ", line_no));
                s.push_str(line);
                s.push('\n');
            }

            if line_no > self.start_line.saturating_add(Self::CONTEXT_LINES)
                && line_no < self.end_line.saturating_sub(Self::CONTEXT_LINES)
                && !printed_dots
            {
                s.push_str("...\n");
                printed_dots = true;
            }

            if line_no > self.end_line.saturating_sub(Self::CONTEXT_LINES)
                && line_no < self.end_line.saturating_add(Self::CONTEXT_LINES)
                && !(line_no
                    > self.start_line.saturating_sub(Self::CONTEXT_LINES)
                    && line_no
                        < self.start_line.saturating_add(Self::CONTEXT_LINES))
            {
                s.push_str(&format!("{:04}  > ", line_no));
                s.push_str(line);
                s.push('\n');
            }

            // Print uppercut chars indicating the section that
            // caused the error.
            if line_no == self.start_line {
                s.push_str("start >"); // to match the start line indicator
                for _ in 0..self.start_col {
                    s.push(' ');
                }
                s.push_str("^\n");
            }

            if line_no == self.end_line {
                s.push_str("end   >"); // to match the start line indicator
                for _ in 0..(self.end_col - 1) {
                    s.push(' ');
                }
                s.push_str("^\n");
            }
        }

        s
    }

    const CONTEXT_LINES: usize = 2;
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                             String Utils                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

fn block_leftpad(block: &str, pad: usize) -> String {
    // Guess that most blocks are going to have columns of 100
    // chars or less.
    let mut s =
        String::with_capacity(block.len() + (pad * (block.len() / 100)));

    let mut pad_str = String::with_capacity(pad);
    for _ in 0..pad {
        pad_str.push(' ');
    }

    for line in block.split("\n") {
        if line == "" {
            continue;
        }

        s.push_str(&pad_str);
        s.push_str(line);
        s.push('\n');
    }

    s
}
