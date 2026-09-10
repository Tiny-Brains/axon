//! The one thing in the dialect that is not JSON, and it is opaque: a tensor is produced by an
//! operator, consumed by an operator, and handed to the graph. A program can ask its shape and its
//! dtype and nothing else — docs/dialect.md §2.

use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType {
    I8,
    U8,
    I16,
    I32,
    F32,
}

impl DType {
    pub fn name(self) -> &'static str {
        match self {
            DType::I8 => "int8",
            DType::U8 => "uint8",
            DType::I16 => "int16",
            DType::I32 => "int32",
            DType::F32 => "float32",
        }
    }

    pub fn parse(s: &str) -> Option<DType> {
        Some(match s {
            "int8" => DType::I8,
            "uint8" => DType::U8,
            "int16" => DType::I16,
            "int32" => DType::I32,
            "float32" => DType::F32,
            _ => return None,
        })
    }

    pub fn bytes(self) -> usize {
        match self {
            DType::I8 | DType::U8 => 1,
            DType::I16 => 2,
            DType::I32 | DType::F32 => 4,
        }
    }
}

/// Elements are held as `f64` whatever the dtype and narrowed on the way to the graph. It costs
/// memory and buys the thing that matters more: every operator is written once instead of five
/// times, so `tb.stack` cannot be right for `int8` and subtly wrong for `float32`.
#[derive(Clone)]
pub struct Tensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Tensor {
    pub fn new(dtype: DType, shape: Vec<usize>, data: Vec<f64>) -> Tensor {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        Tensor { dtype, shape, data }
    }

    pub fn filled(dtype: DType, shape: Vec<usize>, v: f64) -> Tensor {
        let n = shape.iter().product::<usize>();
        Tensor { dtype, shape, data: vec![saturate(v, dtype); n] }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Row-major strides, in elements.
    pub fn strides(&self) -> Vec<usize> {
        let mut s = vec![1usize; self.shape.len()];
        for i in (0..self.shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * self.shape[i + 1];
        }
        s
    }

    /// The bytes the graph gets, little-endian, narrowed and saturated to the dtype.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.len() * self.dtype.bytes());
        for &v in &self.data {
            match self.dtype {
                DType::I8 => out.push((saturate(v, DType::I8) as i8) as u8),
                DType::U8 => out.push(saturate(v, DType::U8) as u8),
                DType::I16 => {
                    out.extend_from_slice(&(saturate(v, DType::I16) as i16).to_le_bytes())
                }
                DType::I32 => {
                    out.extend_from_slice(&(saturate(v, DType::I32) as i32).to_le_bytes())
                }
                DType::F32 => out.extend_from_slice(&(v as f32).to_le_bytes()),
            }
        }
        out
    }
}

/// Saturating narrowing, defined here once: scattering 300 into an `int8` plane gives 127, not a
/// wrapped -128, because a wrap would hide a competitor's arithmetic mistake in a plausible number.
pub fn saturate(v: f64, dtype: DType) -> f64 {
    if v.is_nan() {
        return 0.0;
    }
    let (lo, hi) = match dtype {
        DType::I8 => (-128.0, 127.0),
        DType::U8 => (0.0, 255.0),
        DType::I16 => (-32768.0, 32767.0),
        DType::I32 => (-2147483648.0, 2147483647.0),
        DType::F32 => return v,
    };
    v.clamp(lo, hi).trunc()
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor({} {:?}, {} elements)", self.dtype.name(), self.shape, self.len())
    }
}
