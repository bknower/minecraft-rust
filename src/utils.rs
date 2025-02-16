#[macro_export]
macro_rules! printy {
    ($($val:expr),*) => {
        println!("{}", vec![$(format!("{:?}", $val)),*].join(", "));
    };
}
