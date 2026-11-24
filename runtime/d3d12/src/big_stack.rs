//! Host pipeline creation on a thread with a large stack.
//!
//! Proton's vkd3d-proton compiles DXIL to SPIR-V synchronously inside
//! Create*PipelineState, on the calling thread. Its control-flow analysis
//! recurses deeply on large translated shaders. Xbox titles create pipelines
//! on worker threads with 1 MiB stacks, so those threads overflowed their stack
//! inside the compiler and died (ProjectScarlett stayed on a black screen).

/// Reserved, not committed: untouched pages cost only address space.
const STACK_BYTES: usize = 64 << 20;

struct AssertSend<T>(T);
// SAFETY: D3D12 devices and the descriptors they read are free-threaded, and
// the caller blocks until the helper thread has finished with them.
unsafe impl<T> Send for AssertSend<T> {}

/// Run `job` on a scoped helper thread with a 64 MiB stack and return its
/// result. Falls back to the current thread if the helper cannot be created.
pub(crate) fn run<R, F: FnOnce() -> R>(job: F) -> R {
    let mut slot = Some(job);
    let pointer = AssertSend(&mut slot as *mut Option<F>);
    let result = std::thread::scope(|scope| {
        let spawned = std::thread::Builder::new()
            .name("tuxbox-pipeline".into())
            .stack_size(STACK_BYTES)
            .spawn_scoped(scope, move || {
                let pointer = pointer;
                // SAFETY: the spawning thread waits for this scope to end
                // before reading `slot` again.
                let job = unsafe { (*pointer.0).take() }.expect("pipeline job already taken");
                AssertSend(job())
            });
        match spawned {
            Ok(handle) => Some(handle.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)).0),
            Err(_) => None,
        }
    });
    match result {
        Some(value) => value,
        None => (slot.take().expect("pipeline job already taken"))(),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn deep_recursion_fits_on_the_helper_stack() {
        fn depth(n: u32) -> u32 {
            let pad = std::hint::black_box([0u8; 1024]);
            if n == 0 { pad[0] as u32 } else { depth(n - 1) + std::hint::black_box(1) }
        }
        // About 8 MiB of frames: more than a 1 MiB thread stack holds.
        assert_eq!(super::run(|| depth(8000)), 8000);
    }

    #[test]
    fn returns_borrowed_results() {
        let text = String::from("pso");
        assert_eq!(super::run(|| text.len()), 3);
    }
}
