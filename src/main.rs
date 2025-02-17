#![allow(unused_imports, unused)]
use std::env;

use minecraft_rust::run;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    pollster::block_on(run());
}
