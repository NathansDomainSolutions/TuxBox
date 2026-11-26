use super::*;

#[test]
fn a_directory_is_listed_once_and_extended() {
    let dir = std::env::temp_dir().join(format!("tuxbox-names-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("a.sha256"), b"").unwrap();
    let names = Names::new();
    assert!(names.of(&dir).contains("a.sha256"));
    fs::write(dir.join("b.sha256"), b"").unwrap();
    assert!(!names.of(&dir).contains("b.sha256"));
    names.of(&dir).insert("b.sha256".into());
    assert!(names.of(&dir).contains("b.sha256"));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_quick_reply_is_not_held_for_a_whole_pause() {
    let mut pauses = Vec::new();
    let mut looks = 0;
    let reply = await_reply(Duration::from_secs(90), |pause| pauses.push(pause), || {
        looks += 1;
        (looks == 3).then_some("reply")
    });
    assert_eq!(reply, Some("reply"));
    assert_eq!(pauses, [1, 2].map(Duration::from_millis));
}

#[test]
fn a_slow_reply_settles_at_the_slowest_pause() {
    let mut pauses = Vec::new();
    let mut looks = 0;
    await_reply(Duration::from_secs(90), |pause| pauses.push(pause), || {
        looks += 1;
        (looks == 9).then_some(())
    });
    assert_eq!(pauses, [1, 2, 4, 8, 16, 25, 25, 25].map(Duration::from_millis));
}

#[test]
fn waiting_ends_when_the_limit_passes() {
    let mut looks = 0;
    let reply = await_reply(Duration::ZERO, |_| panic!("paused past the limit"), || {
        looks += 1;
        None::<()>
    });
    assert_eq!((reply, looks), (None, 1));
}
