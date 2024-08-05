pub mod session;
pub mod data;

use ort::{
    GraphOptimizationLevel,
    Session,
    Tensor
};
use ndarray::Array4;
use anyhow;

fn main() -> anyhow::Result<()> {
    let input = Array4::<f32>::zeros((1, 3, 640, 640));
    let input = Tensor::from_array(input)?;
    let session = Session::builder()?
        .with_execution_providers([
            session::get_coreml_provider()
        ])?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file("model.onnx")?;
    let output = session.run(ort::inputs!["images" => input]?)?;
    println!("{:?}", output);
    Ok(())
}
