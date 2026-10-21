#[test]
fn raf_exposes_fuji_lens_tables() {
  let Ok(path) = std::env::var("RAF_SAMPLE") else { eprintln!("RAF_SAMPLE unset; skipping"); return };
  let image = rawloader::decode_file(&path).expect("decode");
  let tables = image.fuji_lens.expect("tables present on a Fuji RAF");
  assert_eq!((tables.distortion.len(), tables.ca.len(), tables.vignetting.len()), (19, 29, 19));
  assert!((tables.distortion[1] - 0.3535211268).abs() < 1e-6);
  assert!((tables.distortion[18] - -8.895065308).abs() < 1e-4);
  assert_eq!(tables.frame[4], 4160); // cropped height of the X-T3 sample
  assert_eq!(tables.frame[5], 6240);
}
