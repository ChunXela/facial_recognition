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

// This function reads a frame from the camera, detects faces in the frame, and returns their positions
fn detect_faces(camera: &mut videoio::VideoCapture, face_detector: &mut objdetect::CascadeClassifier) -> Result<Option<types::VectorOfRect>> {
    // Read a frame from the camera
    let mut img = Mat::default();
    camera.read(&mut img)?;

    // Convert the frame to grayscale
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

    // If at least one face was detected, return the positions of the faces
    Ok(if faces.len() > 0 { Some(faces) } else { None })
}

// This function draws rectangles around the detected faces in the image
fn draw_faces(img: &mut Mat, faces: &types::VectorOfRect) -> Result<()> {
    // Loop through all detected faces and draw a rectangle around each one
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
    Ok(())
}

// This function displays the image in a window
fn show_image(img: &Mat, name: &str) -> Result<()> {
    // Create a window with the given name
    highgui::named_window(name, highgui::WINDOW_NORMAL)?;

    // Resize the window to 800x600
    // Display the image in the window
    highgui::imshow(name, img)?;

    // Wait for a key press for 1ms
    highgui::wait_key(1)?;

    Ok(())
}

// The main function that sets up the camera and face detector and continuously detects faces
fn main() -> Result<()> {
    // Open the default camera (id = 0)
    let mut camera = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;

    // Load the face detector XML file
    let xml = r"./haarcascade_frontalface_default.xml";
    let mut face_detector = objdetect::CascadeClassifier::new(xml)?;

    loop {
        // Detect faces in the current camera frame
        match detect_faces(&mut camera, &mut face_detector)? {
            Some(faces) => {
                // If faces are detected, read the frame from the camera again and draw rectangles around the faces
                let mut img = Mat::default();
                camera.read(&mut img)?;
                draw_faces(&mut img, &faces)?;
                show_image(&img, "Face Detection")?;
            },
            None => {}
        }
    }
}
