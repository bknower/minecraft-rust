#![allow(unused_imports, unused)]
use minecraft_rust::run;

fn main() {
    pollster::block_on(run());
}
