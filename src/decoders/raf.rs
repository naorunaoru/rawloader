use std::f32::NAN;

use crate::decoders::*;
use crate::decoders::tiff::*;
use crate::decoders::basics::*;

#[derive(Debug, Clone)]
pub struct RafDecoder<'a> {
  buffer: &'a [u8],
  rawloader: &'a RawLoader,
  tiff: TiffIFD<'a>,
}

impl<'a> RafDecoder<'a> {
  pub fn new(buf: &'a [u8], tiff: TiffIFD<'a>, rawloader: &'a RawLoader) -> RafDecoder<'a> {
    RafDecoder {
      buffer: buf,
      tiff: tiff,
      rawloader: rawloader,
    }
  }
}

/// Where a RAF's raw data lies and how it is laid out
struct RafRaw<'a> {
  camera: Camera,
  width: usize,
  height: usize,
  offset: usize,
  bps: usize,
  src: &'a [u8],
}

impl<'a> RafRaw<'a> {
  /// The offset in `src` of the Fuji compressed data, or None when the data is not compressed
  fn compressed_offset(&self) -> Result<Option<usize>, String> {
    let src = self.src;
    if self.camera.find_hint("double_width") || self.camera.find_hint("jpeg32") || src.len() >= self.bps*self.width*self.height/8 {
      return Ok(None);
    }
    // Compressed RAF — find the Fuji compressed header (0x4953 signature)
    for i in 0..src.len().saturating_sub(16).min(100000) {
      if src[i] == 0x49 && src[i+1] == 0x53 && (src[i+2] == 0 || src[i+2] == 1) && (src[i+3] == 0 || src[i+3] == 16) {
        return Ok(Some(i));
      }
    }
    Err("RAF: Compressed but could not find Fuji compressed header".to_string())
  }
}

impl<'a> RafDecoder<'a> {
  fn raw(&self) -> Result<RafRaw<'a>, String> {
    let camera = self.rawloader.check_supported(&self.tiff)?;
    let raw = fetch_ifd!(&self.tiff, Tag::RafOffsets);
    let (width,height) = if raw.has_entry(Tag::RafImageWidth) {
      (fetch_tag!(raw, Tag::RafImageWidth).get_usize(0),
       fetch_tag!(raw, Tag::RafImageLength).get_usize(0))
    } else {
      let sizes = fetch_tag!(raw, Tag::ImageWidth);
      (sizes.get_usize(1), sizes.get_usize(0))
    };
    let offset = fetch_tag!(raw, Tag::RafOffsets).get_usize(0) + raw.start_offset();
    let bps = match raw.find_entry(Tag::RafBitsPerSample) {
      Some(val) => val.get_u32(0) as usize,
      None      => 16,
    };
    Ok(RafRaw { camera, width, height, offset, bps, src: &self.buffer[offset..] })
  }

  /// Everything after the Fuji decompression; `image` is the decompressor's output
  fn compressed_image(&self, raw: RafRaw, image: Vec<u16>, dummy: bool) -> Result<RawImage, String> {
    let RafRaw { mut camera, width, height, offset, bps, .. } = raw;
    // The Fuji compressed decompressor always outputs pixels at a fixed
    // CFA phase (matching rawpy/libraw), which may differ from the phase
    // in rawloader's camera TOML. Override the CFA to match the
    // decompressor's output so downstream WB/demosaic use correct channels.
    camera.cfa = cfa::CFA::new("GGRGGBGGBGGRBRGRBGGGBGGRGGRGGBRBGBRG");
    // The decompressed output includes the full sensor with zero-filled
    // borders (optical black padding) that the TOML crops may not account
    // for. Detect and skip zero rows/columns so they don't produce noise.
    if !dummy {
      let mut top_skip = 0;
      for y in 0..height.min(24) {
        if image[y * width..(y + 1) * width].iter().all(|&v| v == 0) {
          top_skip = y + 1;
        } else {
          break;
        }
      }
      let mut bot_skip = 0;
      for y in (height.saturating_sub(24)..height).rev() {
        if image[y * width..(y + 1) * width].iter().all(|&v| v == 0) {
          bot_skip = height - y;
        } else {
          break;
        }
      }
      // Sample every 6th row for column checks (aligned to CFA period).
      // Zero columns are fully zero from decompressor, so sparse sampling
      // is sufficient and avoids scanning all ~4000 rows per column.
      let sample_rows: Vec<usize> = (0..height).step_by(6).collect();
      let mut right_skip = 0;
      for x in (width.saturating_sub(256)..width).rev() {
        if sample_rows.iter().all(|&y| image[y * width + x] == 0) {
          right_skip = width - x;
        } else {
          break;
        }
      }
      let mut left_skip = 0;
      for x in 0..width.min(256) {
        if sample_rows.iter().all(|&y| image[y * width + x] == 0) {
          left_skip = x + 1;
        } else {
          break;
        }
      }
      // Use detected zero boundaries directly — the decompressor
      // produces clean zero borders, so these are always correct
      // and override potentially wrong TOML values.
      camera.crops[0] = top_skip;
      camera.crops[1] = right_skip;
      camera.crops[2] = bot_skip;
      camera.crops[3] = left_skip;
    }
    ok_image(camera, width, height, self.get_wb()?, offset, bps, Encoding::Fuji, image)
      .map(|mut img| { img.fuji_lens = self.lens_tables(); img })
  }
}

impl<'a> Decoder for RafDecoder<'a> {
  fn image(&self, dummy: bool) -> Result<RawImage,String> {
    let raw = self.raw()?;
    if let Some(off) = raw.compressed_offset()? {
      let image = fuji_compressed::decode_fuji_compressed(&raw.src[off..], raw.width, raw.height, dummy)?;
      return self.compressed_image(raw, image, dummy);
    }
    let RafRaw { camera, width, height, offset, bps, src } = raw;

    let image = if camera.find_hint("double_width") {
      // Some fuji SuperCCD cameras include a second raw image next to the first one
      // that is identical but darker to the first. The two combined can produce
      // a higher dynamic range image. Right now we're ignoring it.
      decode_16le_skiplines(src, width, height, dummy)
    } else if camera.find_hint("jpeg32") {
      decode_12be_msb32(src, width, height, dummy)
    } else {
      match bps {
        12 => decode_12le(src, width, height, dummy),
        14 => decode_14le_unpacked(src, width, height, dummy),
        16 => {
          if self.tiff.little_endian() {
            decode_16le(src, width, height, dummy)
          } else {
            decode_16be(src, width, height, dummy)
          }
        },
        _ => {return Err(format!("RAF: Don't know how to decode bps {}", bps).to_string());},
      }
    };

    if camera.find_hint("fuji_rotation") || camera.find_hint("fuji_rotation_alt") {
      let (width, height, image) = RafDecoder::rotate_image(&image, &camera, width, height, dummy);
      Ok(RawImage {
        make: camera.make.clone(),
        model: camera.model.clone(),
        clean_make: camera.clean_make.clone(),
        clean_model: camera.clean_model.clone(),
        width: width,
        height: height,
        cpp: 1,
        wb_coeffs: self.get_wb()?,
        data: RawImageData::Integer(image),
        blacklevels: camera.blacklevels,
        whitelevels: camera.whitelevels,
        xyz_to_cam: camera.xyz_to_cam,
        cfa: camera.cfa.clone(),
        crops: [0,0,0,0],
        bps: bps,
        offset: offset,
        blackareas: Vec::new(),
        orientation: camera.orientation,
        encoding: Encoding::Fuji,
        // The tables and their frame describe the unrotated readout
        fuji_lens: None,
      })
    } else {
      ok_image(camera, width, height, self.get_wb()?, offset, bps, Encoding::Fuji, image)
        .map(|mut img| { img.fuji_lens = self.lens_tables(); img })
    }
  }

  fn fuji_strips(&self) -> Result<Option<FujiStrips>, String> {
    let raw = self.raw()?;
    let off = match raw.compressed_offset()? {
      Some(off) => off,
      None => return Ok(None),
    };
    // Byte ranges in the file, not in the compressed data
    let base = raw.offset + off;
    let strips = fuji_compressed::fuji_strips(&raw.src[off..], raw.width)?.into_iter().map(|strip| FujiStrip {
      data: strip.data.start + base..strip.data.end + base,
      q_bases: strip.q_bases.start + base..strip.q_bases.end + base,
      ..strip
    }).collect();
    Ok(Some(FujiStrips { width: raw.width, height: raw.height, header: base, strips }))
  }

  fn image_with_fuji_pixels(&self, pixels: Vec<u16>) -> Result<RawImage, String> {
    let raw = self.raw()?;
    if raw.compressed_offset()?.is_none() {
      return Err("RAF: not a compressed file".to_string());
    }
    if pixels.len() != raw.width * raw.height {
      return Err("RAF: the decoded pixels don't match the image size".to_string());
    }
    self.compressed_image(raw, pixels, false)
  }
}

impl<'a> RafDecoder<'a> {
  fn get_wb(&self) -> Result<[f32;4], String> {
    match self.tiff.find_entry(Tag::RafWBGRB) {
      Some(levels) => Ok([levels.get_f32(1), levels.get_f32(0), levels.get_f32(2), NAN]),
      None => {
        let levels = fetch_tag!(self.tiff, Tag::RafOldWB);
        Ok([levels.get_f32(1), levels.get_f32(0), levels.get_f32(3), NAN])
      },
    }
  }

  /// The camera's lens correction tables and the crop frame they refer to; None unless the
  /// file carries all three tables, as rationals with finite values
  fn lens_tables(&self) -> Option<FujiLensTables> {
    // get_f32 panics or reads out of bounds on most non-rational types, and a zero
    // denominator gives NaN or infinity, so either makes the table absent
    let values = |tag| self.tiff.find_entry(tag)
      .filter(|entry| entry.typ() == 5 || entry.typ() == 10)
      .map(|entry| (0..entry.count()).map(|i| entry.get_f32(i)).collect::<Vec<f32>>())
      .filter(|table| table.iter().all(|v| v.is_finite()));
    let distortion = values(Tag::RafGeometricDistortion)?;
    let ca = values(Tag::RafChromaticAberration)?;
    let vignetting = values(Tag::RafVignetting)?;

    // The frame comes from the RAF tag directory, which holds the full size (stored as
    // ImageWidth, height then width) next to the crop tags
    let mut frame = [0u16; 6];
    if let Some(dir) = self.tiff.find_first_ifd(Tag::RafCroppedSize) {
      for (i, tag) in [Tag::ImageWidth, Tag::RafCropTopLeft, Tag::RafCroppedSize].iter().enumerate() {
        if let Some(entry) = dir.find_entry(*tag) {
          frame[i*2] = entry.get_u16(0);
          frame[i*2+1] = entry.get_u16(1);
        }
      }
    }
    Some(FujiLensTables { distortion, ca, vignetting, frame })
  }

  fn rotate_image(src: &[u16], camera: &Camera, width: usize, height: usize, dummy: bool) -> (usize, usize, Vec<u16>) {
    let x = camera.crops[3];
    let y = camera.crops[0];
    let cropwidth = width - camera.crops[1] - x;
    let cropheight = height - camera.crops[2] - y;

    if camera.find_hint("fuji_rotation_alt") {
      let rotatedwidth = cropheight + cropwidth/2;
      let rotatedheight = rotatedwidth-1;

      let mut out: Vec<u16> = alloc_image_plain!(rotatedwidth, rotatedheight, dummy);
      if !dummy {
        for row in 0..cropheight {
          let inb = &src[(row+y)*width+x..];
          for col in 0..cropwidth {
            let out_row = rotatedwidth - (cropheight + 1 - row + (col >> 1));
            let out_col = ((col+1) >> 1) + row;
            out[out_row*rotatedwidth+out_col] = inb[col];
          }
        }
      }

      (rotatedwidth, rotatedheight, out)
    } else {
      let rotatedwidth = cropwidth + cropheight/2;
      let rotatedheight = rotatedwidth-1;

      let mut out: Vec<u16> = alloc_image_plain!(rotatedwidth, rotatedheight, dummy);
      if !dummy {
        for row in 0..cropheight {
          let inb = &src[(row+y)*width+x..];
          for col in 0..cropwidth {
            let out_row = cropwidth - 1 - col + (row>>1);
            let out_col = ((row+1) >> 1) + col;
            out[out_row*rotatedwidth+out_col] = inb[col];
          }
        }
      }

      (rotatedwidth, rotatedheight, out)
    }
  }
}


pub struct RafEncoder {
  original: Vec<u8>,
  bps: usize,
  offset: usize,
}

impl RafEncoder {
  pub fn new(original: Vec<u8>, bps: usize, offset: usize) -> RafEncoder {
    RafEncoder{
      original,
      bps,
      offset
    }
  }
}

impl Encoder for RafEncoder {
  fn encode(&self, new: Vec<u16>) -> Vec<u8> {
    let mut out = self.original.clone();
    // panic!("{}, {}", out.len(), new.len());
    let offset = self.offset;
    let n = new.len();
    let _ = n;
    for (i, value) in new.into_iter().enumerate() {
      let bytes = value.to_le_bytes();
      out[offset + i*2] = bytes[0];
      out[offset + i*2 + 1] = bytes[1];
    }
    out
  }
}
