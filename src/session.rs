use ort::{
    CPUExecutionProvider,
    CUDAExecutionProvider,
    CoreMLExecutionProvider,
    DirectMLExecutionProvider,
    ExecutionProviderDispatch,
    GraphOptimizationLevel,
    Session
};

pub fn get_cpu_provider() -> ExecutionProviderDispatch {
    CPUExecutionProvider::default().build()
}

pub fn get_cuda_provider() -> ExecutionProviderDispatch {
    CUDAExecutionProvider::default().build()
}

pub fn get_coreml_provider() -> ExecutionProviderDispatch {
    CoreMLExecutionProvider::default().build()
}

pub fn get_directml_provider() -> ExecutionProviderDispatch {
    DirectMLExecutionProvider::default().build()
}

pub fn get_provider() -> impl IntoIterator<Item = ExecutionProviderDispatch> {
    #[cfg(feature = "cuda")]
    return [get_cpu_provider(), get_cuda_provider()];
    #[cfg(feature = "coreml")]
    return [get_cpu_provider(), get_coreml_provider()];
    #[cfg(feature = "directml")]
    return [get_cpu_provider(), get_directml_provider()];
    return [get_cpu_provider()];
}

pub fn get_session(filename: &str) -> anyhow::Result<Session> {
    let session = Session::builder()?
        .with_execution_providers(get_provider())?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(filename)?;
    Ok(session)
}
