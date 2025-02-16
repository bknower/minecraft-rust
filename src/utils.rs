use std::fmt::{Display, Formatter, Result};

#[macro_export]
macro_rules! printy {
    ($($val:expr),*) => {
        println!("{}", vec![$(format!("{:?}", $val)),*].join(", "));
    };
}

#[macro_export]
macro_rules! stats {
    ($($label:expr => $expr:expr),* $(,)?) => {{
        let data: Vec<(&str, Box<dyn Fn() -> StatValue>)> = vec![
            $(
                ($label, Box::new(|| $expr.into())) // Wrap each value in a closure
            ),*
        ];
        data
    }};
}

#[derive(Debug)]
pub enum StatValue {
    Float(f64),
    Int(i64),
    Uint(u64),
    Str(String),
}

impl Display for StatValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            StatValue::Float(value) => write!(f, "{:.2}", value), // or your float format
            StatValue::Int(value) => write!(f, "{:}", value),     // or however you want ints
            StatValue::Str(value) => write!(f, "{:}", value),
            StatValue::Uint(value) => write!(f, "{:}", value),
        }
    }
}
macro_rules! impl_stat_from_float {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl From<$ty> for StatValue {
                fn from(value: $ty) -> Self {
                    StatValue::Float(value as f64)
                }
            }
            impl From<&$ty> for StatValue {
                fn from(value: &$ty) -> Self {
                    StatValue::Float(*value as f64)
                }
            }
        )+
    }
}

macro_rules! impl_stat_from_int {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl From<$ty> for StatValue {
                fn from(value: $ty) -> Self {
                    StatValue::Int(value as i64)
                }
            }
            impl From<&$ty> for StatValue {
                fn from(value: &$ty) -> Self {
                    StatValue::Int(*value as i64)
                }
            }
        )+
    }
}

macro_rules! impl_stat_from_uint {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl From<$ty> for StatValue {
                fn from(value: $ty) -> Self {
                    StatValue::Uint(value as u64)
                }
            }
            impl From<&$ty> for StatValue {
                fn from(value: &$ty) -> Self {
                    StatValue::Uint(*value as u64)
                }
            }
        )+
    }
}

// Now actually invoke them:

impl_stat_from_float!(f32, f64);
impl_stat_from_int!(i8, i16, i32, i64, isize);
impl_stat_from_uint!(u8, u16, u32, u64, usize);

// For strings:
impl From<String> for StatValue {
    fn from(value: String) -> Self {
        StatValue::Str(value)
    }
}

impl From<&str> for StatValue {
    fn from(value: &str) -> Self {
        StatValue::Str(value.to_owned())
    }
}

pub struct StatItem {
    pub label: &'static str,
    pub value: StatValue,
}
