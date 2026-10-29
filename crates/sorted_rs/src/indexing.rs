pub(super) enum Nb {
    GTOne,
    One,
    Zero,
    MinusOne,
    LTMinusOne,
}
impl From<isize> for Nb {
    fn from(value: isize) -> Self {
        match value {
            0 => Self::Zero,
            1 => Self::One,
            -1 => Self::MinusOne,
            _ if value > 1 => Self::GTOne,
            _ => Self::LTMinusOne,
        }
    }
}
