import Foundation
import CoreImage
import ImageIO
import UniformTypeIdentifiers

let fm = FileManager.default
let args = CommandLine.arguments
guard args.count >= 3 else { fatalError("usage: proraw_decode input_dir output_dir") }
let input=URL(fileURLWithPath: args[1]), output=URL(fileURLWithPath: args[2])
try fm.createDirectory(at: output, withIntermediateDirectories: true)
let ctx=CIContext(options:[.cacheIntermediates:false])
let files=try fm.contentsOfDirectory(at: input, includingPropertiesForKeys:nil)
  .filter{ ["dng","DNG"].contains($0.pathExtension) }.sorted{$0.lastPathComponent<$1.lastPathComponent}
print("DNG count:", files.count)
for (i,u) in files.enumerated() {
  autoreleasepool {
    guard let raw=CIFilter(imageURL:u, options:[.allowDraftMode:false]) else { fatalError("Cannot open \(u.lastPathComponent)") }
    raw.setValue(1.0, forKey:"inputEV")
    guard let image=raw.outputImage else { fatalError("RAW decode failed \(u.lastPathComponent)") }
    let out=output.appendingPathComponent(u.deletingPathExtension().lastPathComponent+".tiff")
    guard let cs=CGColorSpace(name:CGColorSpace.extendedLinearSRGB) else { fatalError("colorspace") }
    do {
      try ctx.writeTIFFRepresentation(of:image, to:out, format:.RGBA16, colorSpace:cs, options:[:])
      print("[\(i+1)/\(files.count)]",u.lastPathComponent,"->",out.lastPathComponent,image.extent)
    } catch { fatalError("TIFF write failed: \(error)") }
  }
}