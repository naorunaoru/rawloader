use std::env;

fn main() {
  let args: Vec<_> = env::args().collect();
  if args.len() != 2 {
    println!("Usage: {} <file>", args[0]);
    std::process::exit(2);
  }
  let file = &args[1];
  match rawloader::decode_file(file) {
    Ok(image)  => {
      if env::var("VERBOSE").is_ok() {
        println!("make: {}", image.make);
        println!("model: {}", image.model);
        println!("width: {}", image.width);
        println!("height: {}", image.height);
        println!("cfa: {}", image.cfa.name);
        println!("crops: {:?}", image.crops);
        println!("blacklevels: {:?}", image.blacklevels);
        println!("whitelevels: {:?}", image.whitelevels);
        println!("wb_coeffs: {:?}", image.wb_coeffs);
      }
      println!("OK file");
    },
    Err(e) => println!("FAILED file: {}", e),
  }
}
