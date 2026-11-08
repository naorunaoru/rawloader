// Panasonic RW2 raw formats 6–8 (S1 series, GH5S, GH6, S5 II, G9 II) use compression schemes this
// decoder doesn't implement. They must be refused with an error, not run through the classic
// decoder, which reads past the end of the data and panics (a trap under wasm32).
#[test]
fn rw2_raw_format_8_is_refused_cleanly() {
  let Ok(path) = std::env::var("RW2_FORMAT8_SAMPLE") else { eprintln!("RW2_FORMAT8_SAMPLE unset; skipping"); return };
  let err = rawloader::decode_file(&path).expect_err("raw format 8 must be refused");
  let msg = err.to_string();
  assert!(msg.contains("raw format 8"), "unexpected error: {}", msg);
  assert!(!msg.contains("panic"), "refused by a caught panic, not a check: {}", msg);
}
