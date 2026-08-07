use core::fmt;

// NOTE (vendor divergence): upstream implements `std::error::Error` for these.
// There is no `core` equivalent, and nothing in the crate or in the kernel's
// adapter uses `dyn Error` -- both errors are matched on directly. Dropping
// the impls is therefore invisible to every caller here.

/// An error that occurred during parsing
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ParseError {
    /// The input string length was too large to fit in a `u32`
    InvalidLength,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            ParseError::InvalidLength => {
                write!(f, "The input string length is too large to fit in a `u32`")
            }
        }
    }
}


/// An error that occurred during a call to `Bytes::set`
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SetBytesError {
    /// The length of the given data would overflow a `u32`
    LengthOverflow,
}

impl fmt::Display for SetBytesError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            SetBytesError::LengthOverflow => {
                write!(f, "The string length is too large to fit in a `u32`")
            }
        }
    }
}

