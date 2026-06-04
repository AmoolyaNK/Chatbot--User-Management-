import Foundation
import ImageIO
import Vision

guard CommandLine.arguments.count == 2 else {
    fputs("Usage: macos_ocr.swift <image-path>\n", stderr)
    exit(2)
}

let imageURL = URL(fileURLWithPath: CommandLine.arguments[1])
guard
    let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
    let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
else {
    fputs("Unable to load image.\n", stderr)
    exit(1)
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true

do {
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    for observation in request.results ?? [] {
        if let candidate = observation.topCandidates(1).first {
            print(candidate.string)
        }
    }
} catch {
    fputs("OCR failed: \(error)\n", stderr)
    exit(1)
}
