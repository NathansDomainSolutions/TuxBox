// Local debugging aid (not for commit): print root signature, PSV resources and instructions.
use std::{env, fs};
use tuxbox_xshader::{extract, rdna2::instructions};
fn main() {
    let a: Vec<String> = env::args().collect();
    let ex = extract(&fs::read(&a[1]).unwrap()).unwrap();
    println!("stage {:?} entry {:?}", ex.stage, ex.entry);
    println!("root {:#?}", ex.root_sig);
    println!("resources {:?}", ex.resources);
    for i in instructions(&ex.isa) {
        println!("{:#06x} {:<24} dst={:?} src={:?} imm={:?}", i.offset, i.mnemonic, i.dst, i.src, i.aux.imm);
    }
}
