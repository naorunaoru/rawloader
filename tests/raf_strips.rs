// A compressed RAF decoded strip by strip, each strip on its own, must give the image the whole-file
// decode gives: the strips are independent, and everything after the decompression is shared.
use rawloader::{Buffer, RawImageData};

fn sample() -> Option<Vec<u8>> {
  let Ok(path) = std::env::var("RAF_COMPRESSED_SAMPLE") else { eprintln!("RAF_COMPRESSED_SAMPLE unset; skipping"); return None };
  Some(std::fs::read(&path).expect("read sample"))
}

/// Decodes every strip into its block and copies the blocks into their columns, last strip first.
fn pixels_from_strips(bytes: &[u8], strips: &rawloader::FujiStrips) -> Result<Vec<u16>, String> {
  let mut pixels = vec![0u16; strips.width * strips.height];
  for strip in strips.strips.iter().rev() {
    let block = rawloader::decode_fuji_strip(&bytes[strips.header..strips.header + 16], &bytes[strip.data.clone()],
      &bytes[strip.q_bases.clone()], strip.strip_width, strip.cols, strips.height).map_err(|e| e.to_string())?;
    assert_eq!(block.len(), strip.cols * strips.height);
    if strip.cols == 0 { continue }
    for (row, src) in pixels.chunks_mut(strips.width).zip(block.chunks(strip.cols)) {
      row[strip.x..strip.x + strip.cols].copy_from_slice(src);
    }
  }
  Ok(pixels)
}

#[test]
fn raf_decoded_strip_by_strip_matches_the_whole_decode() {
  let Some(bytes) = sample() else { return };
  let whole = rawloader::decode(&mut std::io::Cursor::new(&bytes[..])).expect("decode");

  let buffer = Buffer::new(&mut std::io::Cursor::new(&bytes[..])).unwrap();
  let strips = rawloader::fuji_strips(&buffer).expect("layout").expect("a compressed RAF has strips");
  assert_eq!((strips.width, strips.height), (whole.width, whole.height));
  assert!(strips.strips.len() > 1);
  let pixels = pixels_from_strips(&bytes, &strips).expect("strips");
  let image = rawloader::decode_with_fuji_pixels(&buffer, pixels).expect("finish");

  let (RawImageData::Integer(a), RawImageData::Integer(b)) = (&whole.data, &image.data) else { panic!("integer data") };
  assert!(a == b, "pixels differ");
  assert_eq!(whole.crops, image.crops);
  assert_eq!(whole.cfa.name, image.cfa.name);
  assert_eq!(whole.whitelevels, image.whitelevels);
  assert_eq!(whole.calibrated_whitelevels(), image.calibrated_whitelevels());
  assert_eq!(whole.blacklevels, image.blacklevels);
  assert_eq!(whole.fuji_lens.map(|l| l.distortion), image.fuji_lens.map(|l| l.distortion));
}

#[test]
fn raf_strip_cut_short_by_zeros_is_refused_on_its_own() {
  let Some(mut bytes) = sample() else { return };
  let tail = bytes.len() - 65536;
  bytes[tail..].fill(0);
  let buffer = Buffer::new(&mut std::io::Cursor::new(&bytes[..])).unwrap();
  let strips = rawloader::fuji_strips(&buffer).unwrap().unwrap();
  let err = pixels_from_strips(&bytes, &strips).expect_err("the last strip must be refused");
  assert!(err.contains("strip data ended early"), "unexpected error: {}", err);
  // The other strips still decode
  let first = &strips.strips[0];
  rawloader::decode_fuji_strip(&bytes[strips.header..strips.header + 16], &bytes[first.data.clone()],
    &bytes[first.q_bases.clone()], first.strip_width, first.cols, strips.height).expect("the first strip is intact");
}

#[test]
fn pixels_of_the_wrong_size_and_files_without_strips_are_refused() {
  // Not a raw file at all
  let buffer = Buffer::new(&mut std::io::Cursor::new(&[0u8; 64][..])).unwrap();
  assert!(rawloader::fuji_strips(&buffer).is_err());
  assert!(rawloader::decode_with_fuji_pixels(&buffer, vec![0; 16]).is_err());

  let Some(bytes) = sample() else { return };
  let buffer = Buffer::new(&mut std::io::Cursor::new(&bytes[..])).unwrap();
  let err = rawloader::decode_with_fuji_pixels(&buffer, vec![0; 16]).err().expect("too few pixels");
  assert!(err.to_string().contains("don't match the image size"), "unexpected error: {}", err);
}
