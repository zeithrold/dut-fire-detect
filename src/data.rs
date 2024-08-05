use image::ImageReader;
use ndarray::{Array3, Array4};

pub fn proprocess_tensor(tensor: Array3<f32>) -> Array4<f32> {
    tensor
        .permuted_axes([2, 0, 1])
        .mapv(|x| x / 255.0)
        .insert_axis(ndarray::Axis(0))
}
