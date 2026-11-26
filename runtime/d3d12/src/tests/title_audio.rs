use super::*;

fn image() -> Vec<u8> {
    let mut image = vec![0u8; EPILOGUE + 64];
    image[PADDING] = 0xcc;
    image[CHECK..CHECK + CHECK_BYTES.len()].copy_from_slice(&CHECK_BYTES);
    image[EPILOGUE..EPILOGUE + EPILOGUE_BYTES.len()].copy_from_slice(&EPILOGUE_BYTES);
    image[UNLOCK..UNLOCK + UNLOCK_BYTES.len()].copy_from_slice(&UNLOCK_BYTES);
    image
}

#[test]
fn only_the_matching_build_is_patched() {
    assert!(matches(&image()));
    for rva in [PADDING, CHECK, EPILOGUE, UNLOCK] {
        let mut other = image();
        other[rva] ^= 1;
        assert!(!matches(&other), "a change at {rva:#x} must prevent the patch");
    }
    assert!(!matches(&image()[..PADDING]));
}

#[test]
fn the_branch_moves_from_the_epilogue_to_the_padding() {
    let next = CHECK + DISPLACEMENT + 4;
    let original = i32::from_le_bytes(CHECK_BYTES[DISPLACEMENT..].try_into().unwrap());
    assert_eq!(next.wrapping_add_signed(original as isize), EPILOGUE);
    assert_eq!(next.wrapping_add_signed(i32::from_le_bytes(redirected()) as isize), PADDING);
}

#[test]
fn the_handler_calls_unlock_and_returns_to_the_epilogue() {
    let mut stack = [0u64; 8];
    let start = 0x1_4000_0000usize;
    let mut context = CONTEXT { Rdi: 0x1234, ..Default::default() };
    context.Rsp = (&raw mut stack[4]) as u64;
    assert!(!handle(&mut context, start, start + PADDING + 1));
    assert!(handle(&mut context, start, start + PADDING));
    assert_eq!(context.Rcx, 0x1234);
    assert_eq!(context.Rip, (start + UNLOCK) as u64);
    assert_eq!(context.Rsp, (&raw mut stack[3]) as u64);
    assert_eq!(stack[3], (start + EPILOGUE) as u64);
}
