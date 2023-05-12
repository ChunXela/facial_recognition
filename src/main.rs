use opencv::{
    Result,
    prelude::*,
    objdetect,
    highgui,
    imgproc,
    core, 
    types,
    videoio,
};

/// Detects faces in an image captured by a camera using a Haar cascade classifier.
/// Returns a vector of rectangles representing the detected faces, or None if no faces were detected.
fn detect_faces(camera: &mut videoio::VideoCapture, face_detector: &mut objdetect::CascadeClassifier) -> Result<Option<types::VectorOfRect>> {
    // Capture an image from the camera
    let mut img = Mat::default();
    camera.read(&mut img)?;

    // Convert the image to grayscale
    let mut gray = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // Detect faces in the grayscale image
    let mut faces = types::VectorOfRect::new();
    face_detector.detect_multi_scale(
        &gray, 
        &mut faces, 
        1.4, 
        2,
        objdetect::CASCADE_SCALE_IMAGE, 
        core::Size::new(10,10), 
        core::Size::new(0,0)
    )?;

    // Return the detected faces, if any
    Ok(if faces.len() > 0 { Some(faces) } else { None })
}

/// Draws rectangles around the detected faces in an image and displays the image.
fn draw_faces(img: &mut Mat, faces: &types::VectorOfRect) -> Result<()> {
    // Draw a rectangle around each detected face
    for face in faces.iter() {
        imgproc::rectangle(
            img, 
            face, 
            core::Scalar::new(0f64, 255f64, 0f64, 0f64), 
            2, 
            imgproc::LINE_8, 
            0
        )?;
    }

    // Display the image
    highgui::imshow("gray", img)?;
    highgui::wait_key(1)?;
    Ok(())
}

fn main() -> Result<()> {
    // Create a new video capture object for camera 2
    let mut camera = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;

    // Load the Haar cascade classifier for frontal faces
    let xml = r"./haarcascade_frontalface_default.xml";
    let mut face_detector = objdetect::CascadeClassifier::new(xml)?;

    loop {
        // Detect faces in the captured image
        match detect_faces(&mut camera, &mut face_detector)? {
            Some(faces) => {
                // Draw rectangles around the detected faces and display the image
                let mut img = Mat::default();
                draw_faces(&mut img, &faces)?;
            },
            None => {}
        }
    }
}
