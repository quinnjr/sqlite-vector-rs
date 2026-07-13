use sqlite_vector_rs::types::{VectorType, cast_blob, slice_to_blob};

#[test]
fn parse_type_names() {
    assert_eq!(VectorType::from_name("float4").unwrap(), VectorType::Float4);
    assert_eq!(VectorType::from_name("float2").unwrap(), VectorType::Float2);
    assert_eq!(VectorType::from_name("float8").unwrap(), VectorType::Float8);
    assert_eq!(VectorType::from_name("int1").unwrap(), VectorType::Int1);
    assert_eq!(VectorType::from_name("int2").unwrap(), VectorType::Int2);
    assert_eq!(VectorType::from_name("int4").unwrap(), VectorType::Int4);
    assert!(VectorType::from_name("float16").is_err());
    assert!(VectorType::from_name("").is_err());
}

#[test]
fn element_size_bytes() {
    assert_eq!(VectorType::Float2.element_size(), 2);
    assert_eq!(VectorType::Float4.element_size(), 4);
    assert_eq!(VectorType::Float8.element_size(), 8);
    assert_eq!(VectorType::Int1.element_size(), 1);
    assert_eq!(VectorType::Int2.element_size(), 2);
    assert_eq!(VectorType::Int4.element_size(), 4);
}

#[test]
fn validate_blob_size() {
    // 3-dim float4 = 12 bytes
    let vtype = VectorType::Float4;
    assert!(vtype.validate_blob(&[0u8; 12], 3).is_ok());
    assert!(vtype.validate_blob(&[0u8; 8], 3).is_err());
    assert!(vtype.validate_blob(&[0u8; 16], 3).is_err());
}

#[test]
fn blob_round_trip_float4() {
    let values: Vec<f32> = vec![1.0, 2.0, 3.0];
    let blob = slice_to_blob(&values);
    assert_eq!(blob.len(), 12);
    let restored = cast_blob::<f32>(&blob);
    assert_eq!(restored.as_ref(), &[1.0, 2.0, 3.0]);
}

#[test]
fn blob_round_trip_float2() {
    use half::f16;
    let values: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
    let blob = slice_to_blob(&values);
    assert_eq!(blob.len(), 4);
    let restored = cast_blob::<f16>(&blob);
    assert_eq!(restored[0].to_f32(), 1.0);
    assert_eq!(restored[1].to_f32(), 2.0);
}

#[test]
fn reject_nan_inf_float4() {
    let with_nan: Vec<f32> = vec![1.0, f32::NAN, 3.0];
    let blob = slice_to_blob(&with_nan);
    assert!(VectorType::Float4.validate_finite(&blob, 3).is_err());

    let with_inf: Vec<f32> = vec![1.0, f32::INFINITY, 3.0];
    let blob = slice_to_blob(&with_inf);
    assert!(VectorType::Float4.validate_finite(&blob, 3).is_err());
}

#[test]
fn validate_finite_skips_integer_types() {
    // Integer types don't have NaN/Inf, validate_finite should always pass
    let blob = slice_to_blob(&[1i32, 2, 3]);
    assert!(VectorType::Int4.validate_finite(&blob, 3).is_ok());
}
