// A compressed RAF whose last strip is cut short by zeros must be refused with an error. Its
// bitstream loses sync and runs off the end of the strip; the decoder used to count the zeros
// after it forever.
#[test]
fn raf_with_a_zeroed_strip_tail_is_refused() {
  let Ok(path) = std::env::var("RAF_COMPRESSED_SAMPLE") else { eprintln!("RAF_COMPRESSED_SAMPLE unset; skipping"); return };
  let mut bytes = std::fs::read(&path).expect("read sample");
  let tail = bytes.len() - 65536;
  bytes[tail..].fill(0);
  let err = rawloader::decode(&mut std::io::Cursor::new(&bytes[..])).err().expect("a zeroed strip tail must be refused");
  assert!(err.to_string().contains("strip data ended early"), "unexpected error: {}", err);
}
