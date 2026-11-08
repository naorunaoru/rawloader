use crate::decoders::cfa::CFA;

/// White-point calibration: the effective white level of each CFA colour, measured from the data.
///
/// Camera databases (this crate's TOML files included) often give the theoretical ADC maximum
/// (16383 for 14 bits) rather than the level where photosites actually saturate, which can be
/// lower. For each colour, over the visible area (`crops`: top, right, bottom, left), this finds
/// the data maximum and how many photosites sit within 1 DN of it. If more than 0.01 % of that
/// colour's photosites do (real clipping) and the maximum is below the metadata level, the
/// maximum becomes the white level. Levels are in RGBE order, like `whitelevels`; E keeps its
/// metadata value.
pub fn calibrate_whitelevels(data: &[u16], width: usize, crops: [usize;4], cfa: &CFA, whitelevels: [u16;4]) -> [u16;4] {
  let (pw, ph) = (cfa.width, cfa.height);
  if pw == 0 || ph == 0 || width == 0 { return whitelevels }
  let height = data.len() / width;
  let (top, right, bottom, left) = (crops[0], crops[1], crops[2], crops[3]);
  if top + bottom >= height || left + right >= width { return whitelevels }
  let (vw, vh) = (width - left - right, height - top - bottom);

  // One pass, per phase of the pattern so each inner loop sees a single colour: the maximum, how
  // many photosites sit at it, and how many sit 1 DN below it.
  let mut phases = vec![Phase::default(); pw * ph];
  for y in 0..vh {
    let row = &data[(y + top) * width + left..][..vw];
    let base = (y % ph) * pw;
    for px in 0..pw.min(vw) {
      let s = &mut phases[base + px];
      let (mut max, mut at, mut below) = (s.max as i32, s.at, s.below);
      for &v in row[px..].iter().step_by(pw) {
        let v = v as i32;
        if v < max - 1 { continue }
        if v == max { at += 1 }
        else if v > max { below = if v == max + 1 { at } else { 0 }; at = 1; max = v }
        else { below += 1 }
      }
      *s = Phase { max: max as u16, at, below };
    }
  }

  // Merge the phases of each colour.
  let (mut max, mut cnt, mut near) = ([0u16; 3], [0usize; 3], [0usize; 3]);
  for py in 0..ph.min(vh) {
    for px in 0..pw.min(vw) {
      let c = cfa.color_at(top + py, left + px);
      if c > 2 { continue }
      cnt[c] += ((vh - py + ph - 1) / ph) * ((vw - px + pw - 1) / pw);
      max[c] = max[c].max(phases[py * pw + px].max);
    }
  }
  for (i, s) in phases.iter().enumerate() {
    let (py, px) = (i / pw, i % pw);
    if py >= vh || px >= vw { continue }
    let c = cfa.color_at(top + py, left + px);
    if c > 2 { continue }
    if s.max == max[c] { near[c] += s.at + s.below }
    else if s.max as i32 == max[c] as i32 - 1 { near[c] += s.at }
  }

  let mut out = whitelevels;
  for c in 0..3 {
    if near[c] as f64 > cnt[c] as f64 * 1e-4 && max[c] < whitelevels[c] { out[c] = max[c] }
  }
  out
}

#[derive(Clone, Copy, Default)]
struct Phase { max: u16, at: usize, below: usize }

#[cfg(test)]
mod tests {
  use super::*;

  const WHITE: [u16;4] = [16383; 4];
  const XTRANS: &str = "RBGBRGGGRGGBGGBGGRBRGRBGGGBGGRGGRGGB";

  /// The rule applied per photosite in two passes, as the app's JS did before.
  fn reference(data: &[u16], width: usize, height: usize, crops: [usize;4], cfa: &CFA, whites: [u16;4]) -> [u16;4] {
    let (top, right, bottom, left) = (crops[0], crops[1], crops[2], crops[3]);
    let (mut max, mut cnt, mut near) = ([0u16; 4], [0usize; 4], [0usize; 4]);
    for row in top..height - bottom {
      for col in left..width - right {
        let c = cfa.color_at(row, col);
        max[c] = max[c].max(data[row * width + col]);
        cnt[c] += 1;
      }
    }
    for row in top..height - bottom {
      for col in left..width - right {
        let c = cfa.color_at(row, col);
        if data[row * width + col] as i32 >= max[c] as i32 - 1 { near[c] += 1; }
      }
    }
    let mut out = whites;
    for c in 0..3 {
      if near[c] as f64 > cnt[c] as f64 * 1e-4 && max[c] < whites[c] { out[c] = max[c]; }
    }
    out
  }

  /// Deterministic pseudo-random data clustered near `top`, so maxima and near-maxima repeat.
  fn noisy(n: usize, top: u16, seed: u32) -> Vec<u16> {
    let mut x = seed;
    (0..n).map(|_| {
      x = x.wrapping_mul(1103515245).wrapping_add(12345);
      (top as i32 - (((x >> 16) % 5) * ((x >> 8) % 3)) as i32).max(0) as u16
    }).collect()
  }

  #[test]
  fn adopts_a_measured_saturation_below_the_metadata_level() {
    // 200×200 RGGB: red saturates at 4000 on 10% of its photosites; green and blue each have a
    // single bright outlier (under the 0.01% threshold), so they keep the metadata level.
    let w = 200;
    let mut data = vec![100u16; w * w];
    data[1] = 500;
    data[w + 1] = 500;
    for y in (0..w).step_by(2) { for x in (0..w).step_by(20) { data[y * w + x] = 4000; } }
    assert_eq!(calibrate_whitelevels(&data, w, [0;4], &CFA::new("RGGB"), WHITE), [4000, 16383, 16383, 16383]);
  }

  #[test]
  fn keeps_the_metadata_level_when_the_maximum_reaches_it() {
    let mut data = vec![1000u16; 16];
    data[0] = 1200;
    assert_eq!(calibrate_whitelevels(&data, 4, [0;4], &CFA::new("RGGB"), [1000; 4]), [1000; 4]);
  }

  #[test]
  fn measures_only_the_visible_area() {
    // 8×6 readout, visible 4×4 at (top 2, left 2); the outlier lies outside it.
    let mut data = vec![100u16; 8 * 6];
    data[0] = 60000;
    assert_eq!(calibrate_whitelevels(&data, 8, [2, 2, 0, 2], &CFA::new("RGGB"), WHITE), [100, 100, 100, 16383]);
  }

  #[test]
  fn matches_the_per_photosite_reference_on_xtrans_with_crops() {
    // Crops that are not multiples of the period, so phases start mid-pattern and some phases
    // have one row or column fewer than others. Maxima of 0, 1 and 2 cover the 1-DN edge cases.
    let (w, h) = (60, 48);
    let cfa = CFA::new(XTRANS);
    for crops in [[0, 0, 0, 0], [2, 1, 3, 4], [5, 3, 1, 1]] {
      for (top, seed) in [(4000, 1), (4000, 7), (1, 3), (0, 5), (2, 9)] {
        let data = noisy(w * h, top, seed);
        assert_eq!(calibrate_whitelevels(&data, w, crops, &cfa, WHITE), reference(&data, w, h, crops, &cfa, WHITE),
                   "crops {:?} top {} seed {}", crops, top, seed);
        // A single photosite 1 DN above the rest must not count as clipping.
        let mut lifted = data.clone();
        lifted[crops[0] * w + crops[3]] = top + 1;
        assert_eq!(calibrate_whitelevels(&lifted, w, crops, &cfa, WHITE), reference(&lifted, w, h, crops, &cfa, WHITE),
                   "lifted, crops {:?} top {} seed {}", crops, top, seed);
      }
    }
  }

  #[test]
  fn raw_image_reports_calibrated_levels_for_integer_mosaics() {
    use crate::decoders::{Camera, Encoding, RawImage};
    let mut camera = Camera::new();
    camera.cfa = CFA::new("RGGB");
    camera.whitelevels = [16383, 16383, 16383, 0];
    let image = RawImage::new(camera, 4, 4, [1.0; 4], 0, 14, Encoding::NotImplemented, vec![4000u16; 16], false);
    assert_eq!(image.calibrated_whitelevels(), [4000, 4000, 4000, 0]);
  }

  #[test]
  fn leaves_levels_alone_without_a_cfa() {
    assert_eq!(calibrate_whitelevels(&[5u16; 16], 4, [0;4], &CFA::new(""), [1000; 4]), [1000; 4]);
  }
}
