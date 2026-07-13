use sqlite_vector_rs::json::{blob_to_json, json_to_blob};
use sqlite_vector_rs::types::{VectorType, cast_blob, slice_to_blob};

#[test]
fn json_to_float4_blob() {
    let blob = json_to_blob("[1.0, 2.0, 3.0]", VectorType::Float4).unwrap();
    let values = cast_blob::<f32>(&blob);
    assert_eq!(values.as_ref(), &[1.0, 2.0, 3.0]);
}

#[test]
fn json_to_float2_blob() {
    let blob = json_to_blob("[1.0, 2.0]", VectorType::Float2).unwrap();
    assert_eq!(blob.len(), 4); // 2 elements * 2 bytes
}

#[test]
fn json_to_int1_blob() {
    let blob = json_to_blob("[1, 2, -3]", VectorType::Int1).unwrap();
    let values = cast_blob::<i8>(&blob);
    assert_eq!(values.as_ref(), &[1, 2, -3]);
}

#[test]
fn blob_to_json_float4() {
    let values: Vec<f32> = vec![1.5, 2.5, 3.5];
    let blob = slice_to_blob(&values);
    let json = blob_to_json(&blob, VectorType::Float4).unwrap();
    assert_eq!(json, "[1.5,2.5,3.5]");
}

#[test]
fn json_rejects_non_array() {
    assert!(json_to_blob("{}", VectorType::Float4).is_err());
    assert!(json_to_blob("42", VectorType::Float4).is_err());
}

#[test]
fn json_rejects_non_numeric_elements() {
    assert!(json_to_blob("[1.0, \"hello\", 3.0]", VectorType::Float4).is_err());
}

#[test]
fn json_rejects_nan() {
    assert!(json_to_blob("[1.0, null, 3.0]", VectorType::Float4).is_err());
}

#[test]
fn json_empty_array() {
    let blob = json_to_blob("[]", VectorType::Float4).unwrap();
    assert!(blob.is_empty());
}
